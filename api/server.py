"""
Serveur FastAPI : lance le pipeline async + SSE pour HITL et résultat final.

Lancement (depuis la racine du projet BPFragmentODRLProject) ::
    uvicorn api.server:app --reload --host 127.0.0.1 --port 8765

CORS (front Next.js) : par défaut ``localhost:3000`` et ``127.0.0.1:3000``.
Autre origine (ex. IP LAN) : variable d'environnement ``CORS_ORIGINS`` (liste séparée par des virgules)
ou ``CORS_ORIGIN_REGEX`` (ex. ``http://192\\.168\\..*``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for _p in (_ROOT, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")
load_dotenv(_SRC / ".env")

from fastapi import File
from fastapi import HTTPException
from fastapi import FastAPI
from fastapi import Query
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from api.hitl_bridge import WebHitlBridge
from baseline.bpmn_parser import BPMNParser
from baseline.semantic_fragmenter import (
    SemanticFragmenter,
    build_fragments_from_bp_model,
    companion_fragments_enhanced_path,
    save_fragments_enhanced_json,
    save_fragments_json,
)
from agents.policy_auditor import AuditReport

_logger = logging.getLogger(__name__)
from communication.acl import ACLEnvelope
from orchestration.async_pipeline import PipelineAsyncResult, run_pipeline_async
from orchestration.scenario_loader import load_scenario


def _serialize_result(r: PipelineAsyncResult) -> dict[str, Any]:
    rep = r.report
    if rep is None:
        report_out: Any = None
    elif isinstance(rep, AuditReport):
        report_out = rep.summary()
    elif isinstance(rep, dict):
        report_out = rep
    else:
        report_out = str(rep)
    out: dict[str, Any] = {
        "is_valid": r.is_valid,
        "status": r.status,
        "msg_type": r.msg_type,
        "syntax_score": r.syntax_score,
        "summary": r.summary or {},
        "manifest": r.manifest,
        "report_summary": report_out,
        "scenario_id": getattr(r, "scenario_id", "") or "",
    }
    oep = getattr(r, "odrl_export_paths", None)
    if oep:
        out["odrl_export_paths"] = oep
    return out


class StartRunBody(BaseModel):
    scenario_id: str = Field(
        default="scenario1",
        description="Dossier sous src/dataset/",
    )
    human_timeout_s: float = Field(default=120.0, ge=1.0, le=3600.0)
    process_title: Optional[str] = Field(
        default=None,
        description="Libellé métier affiché dans le HITL (ex. Credit Application BP).",
    )


class HitlBody(BaseModel):
    reply_with: str
    decision: str
    comment: str = ""

    @field_validator("decision")
    @classmethod
    def _norm_decision(cls, v: str) -> str:
        d = (v or "").strip().lower()
        if d not in ("agree", "refuse"):
            raise ValueError("decision doit être agree ou refuse")
        return d


RUNS: dict[str, WebHitlBridge] = {}
RUN_TASKS: dict[str, asyncio.Task] = {}

_DEFAULT_FRAGMENTS_FILENAME = "bpmn_fragments_export.json"


def _parse_and_fragment_bpmn_bytes(
    content: bytes,
    file_suffix: str,
    *,
    llm_temperature: float,
    llm_seed: int,
) -> dict[str, Any]:
    """Parse BPMN XML depuis des octets, fragmentation LLM, écrit ``fragments`` + ``fragments_enhanced``."""
    import tempfile

    suf = file_suffix if file_suffix.lower() in (".bpmn", ".xml") else ".bpmn"
    tmp_path: Optional[str] = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=suf, prefix="bpmn_upload_")
        os.write(fd, content)
        os.close(fd)

        parser = BPMNParser()
        model = parser.parse_file(tmp_path)
        if not model:
            raise ValueError("Échec du parsing BPMN (fichier vide, XML invalide ou sans processus).")
        model = parser.convert_ids_to_names(model)

        fragmenter = SemanticFragmenter(
            model,
            llm_temperature=llm_temperature,
            llm_seed=llm_seed,
        )
        fragmenter.fragment_process()

        normalized: list[dict[str, Any]] = []
        for i, frag in enumerate(fragmenter.fragments):
            d = dict(frag)
            d["id"] = f"f{i + 1}"
            normalized.append(d)

        _export_dir = os.environ.get("FRAGMENT_EXPORT_DIR", "").strip()
        out_dir = Path(_export_dir).expanduser() if _export_dir else Path.home() / "Downloads"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = os.environ.get("FRAGMENT_EXPORT_FILENAME", _DEFAULT_FRAGMENTS_FILENAME)
        out_path = out_dir / out_name
        save_fragments_json(normalized, str(out_path))
        enhanced_path = Path(companion_fragments_enhanced_path(str(out_path)))
        save_fragments_enhanced_json(model, normalized, str(enhanced_path))

        return {
            "ok": True,
            "output_path": str(out_path.resolve()),
            "enhanced_output_path": str(enhanced_path.resolve()),
            "fragment_count": len(normalized),
            "filename": out_name,
        }
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _next_scenario_id() -> str:
    dataset = _SRC / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in dataset.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"scenario(\d+)", p.name, re.IGNORECASE)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"scenario{max_n + 1}"


def _normalize_b2p_policies(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    raise ValueError("Le fichier policies doit être un JSON objet ou tableau d'objets.")


def _process_scenario_upload_sync(
    bpmn_bytes: bytes,
    bpmn_suffix: str,
    odrl_bytes: bytes,
    *,
    llm_temperature: float,
    llm_seed: int,
) -> dict[str, Any]:
    """Parse BPMN, fragmentation LLM, écrit ``scenarioN/`` avec bp_global.json, B2P.json, fragments.json, fragments_enhanced.json."""
    import tempfile

    _logger.info(
        "Scénario (sync) : %s octets BPMN, %s octets policies — parsing…",
        len(bpmn_bytes),
        len(odrl_bytes),
    )

    odrl_text = odrl_bytes.decode("utf-8-sig")
    try:
        raw_pol = json.loads(odrl_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Policies ODRL : JSON invalide ({e})") from e
    policies = _normalize_b2p_policies(raw_pol)
    if not policies:
        raise ValueError("Policies ODRL : aucun objet politique valide dans le JSON.")

    suf = bpmn_suffix if bpmn_suffix.lower() in (".bpmn", ".xml") else ".bpmn"
    tmp_path: Optional[str] = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=suf, prefix="bpmn_upload_")
        os.write(fd, bpmn_bytes)
        os.close(fd)

        parser = BPMNParser()
        model = parser.parse_file(tmp_path)
        if not model:
            raise ValueError("Échec du parsing BPMN (fichier vide, XML invalide ou sans processus).")

        _logger.info(
            "BPMN parsé OK — fragmentation LLM en cours (plusieurs appels possibles, souvent 1–5 min ; "
            "le terminal peut sembler « figé » pendant ce temps)."
        )
        fragments = build_fragments_from_bp_model(
            model,
            llm_temperature=llm_temperature,
            llm_seed=llm_seed,
        )

        scenario_id = _next_scenario_id()
        scenario_dir = _SRC / "dataset" / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)

        with open(scenario_dir / "bp_global.json", "w", encoding="utf-8") as f:
            json.dump(model, f, indent=2, ensure_ascii=False)

        with open(scenario_dir / "B2P.json", "w", encoding="utf-8") as f:
            json.dump(policies, f, indent=2, ensure_ascii=False)

        save_fragments_json(fragments, str(scenario_dir / "fragments.json"))
        save_fragments_enhanced_json(
            model,
            fragments,
            str(scenario_dir / "fragments_enhanced.json"),
        )

        _logger.info(
            "POST /scenario/process terminé — %s (%d fragments) → %s",
            scenario_id,
            len(fragments),
            scenario_dir.resolve(),
        )

        return {
            "ok": True,
            "scenario_id": scenario_id,
            "dataset_path": str(scenario_dir.resolve()),
            "fragment_count": len(fragments),
        }
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


app = FastAPI(title="BPFragment ODRL Pipeline API", version="0.1.0")

_default_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
_cors_regex = (os.environ.get("CORS_ORIGIN_REGEX") or "").strip()
_cors_kw: dict[str, Any] = {
    "allow_origins": [o.strip() for o in _default_origins if o.strip()],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}
if _cors_regex:
    _cors_kw["allow_origin_regex"] = _cors_regex
app.add_middleware(CORSMiddleware, **_cors_kw)


@app.post("/runs")
async def start_run(body: StartRunBody) -> dict[str, str]:
    run_id = uuid.uuid4().hex
    bridge = WebHitlBridge()
    RUNS[run_id] = bridge

    base_dir = str(_ROOT)

    async def _runner() -> None:
        try:
            await bridge.emit_log(f"Chargement du scénario « {body.scenario_id} »…")
            bp_model, fragments, b2p_policies = load_scenario(body.scenario_id, base_dir=base_dir)
            await bridge.emit_log("Pipeline async démarré (HITL via API).")

            _logged_agent5 = False

            async def _trace_acl(env: ACLEnvelope) -> None:
                nonlocal _logged_agent5
                if env.receiver == "agent5" and not _logged_agent5:
                    _logged_agent5 = True
                    await bridge.emit_log(
                        "Étape audit (agent 5) en cours — le LLM peut prendre du temps ; la connexion reste ouverte (keep-alive SSE)."
                    )
                await bridge.emit_acl(env)

            result = await run_pipeline_async(
                bp_model=bp_model,
                fragments=fragments,
                b2p_policies=b2p_policies,
                human_timeout_s=body.human_timeout_s,
                human_decision_bridge=bridge,
                on_acl_message=_trace_acl,
                scenario_id=body.scenario_id,
                process_title=body.process_title,
            )
            await bridge.emit_done(_serialize_result(result))
        except Exception as e:
            await bridge.emit_error(str(e))
        finally:
            RUN_TASKS.pop(run_id, None)

    RUN_TASKS[run_id] = asyncio.create_task(_runner())
    return {"run_id": run_id}


_RUN_LOST_MSG = (
    "Ce run_id n'existe pas en mémoire (serveur redémarré avec --reload, "
    "ou session trop ancienne). Relancez « Lancer le pipeline » depuis l'interface."
)


@app.get("/runs/{run_id}/ready")
async def run_session_ready(run_id: str) -> dict[str, Any]:
    """Vérifie que la session existe (utile au client pour distinguer 404 SSE)."""
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail=_RUN_LOST_MSG)
    return {"ok": True, "run_id": run_id}


@app.get("/runs/{run_id}/events")
async def stream_events(run_id: str) -> StreamingResponse:
    bridge = RUNS.get(run_id)
    if bridge is None:
        raise HTTPException(status_code=404, detail=_RUN_LOST_MSG)

    async def gen() -> Any:
        # Commentaires SSE (: …) pour éviter la coupure proxy/navigateur pendant les phases longues (ex. agent 5).
        ka_sec = float(os.environ.get("SSE_KEEPALIVE_SEC", "18"))
        while True:
            item = await bridge.pull_event_or_timeout(ka_sec)
            if item is None:
                yield ": sse-keepalive\n\n"
                continue
            line = json.dumps(item, ensure_ascii=False, default=str)
            yield f"data: {line}\n\n"
            if item.get("type") in ("done", "error"):
                break

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/runs/{run_id}/hitl")
async def submit_hitl(run_id: str, body: HitlBody) -> dict[str, Any]:
    bridge = RUNS.get(run_id)
    if bridge is None:
        raise HTTPException(status_code=404, detail=_RUN_LOST_MSG)
    ok = bridge.submit(body.reply_with.strip(), body.decision, body.comment)
    if not ok:
        print(
            f"[API] POST /hitl refusé — reply_with={body.reply_with!r} "
            f"(inconnu ou déjà répondu ; en attente : {bridge.pending_hitl_keys()!r})"
        )
        raise HTTPException(status_code=400, detail="reply_with inconnu ou déjà répondu")
    return {"ok": True}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/scenario/list")
async def list_scenarios() -> dict[str, Any]:
    dataset = _SRC / "dataset"
    out: list[str] = []
    if dataset.is_dir():
        for p in sorted(dataset.iterdir()):
            if p.is_dir() and re.fullmatch(r"scenario\d+", p.name, re.IGNORECASE):
                out.append(p.name)
    return {"scenarios": out}


def _scenario_id_ok(sid: str) -> bool:
    return bool(re.fullmatch(r"scenario\d+", sid, re.IGNORECASE))


def _odrl_output_root(scenario_id: str) -> Path:
    """``output/{scenarioN}/odrl_fragment_policies/`` à la racine du projet (hors ``dataset``)."""
    return (_ROOT / "output" / scenario_id / "odrl_fragment_policies").resolve()


def _ui_category_from_stem(stem: str) -> str:
    """
    Catégorie affichage dérivée du préfixe du fichier exporté (voir ``PolicyProjectionAgent.export``).
    Retourne : FPa | FPd-intra | FPd-inter | unsupported
    """
    s = stem.lower()
    if s.startswith("fpa_"):
        return "FPa"
    if s.startswith("fpd_message_"):
        return "FPd-inter"
    if (
        s.startswith("fpd_sequence_")
        or s.startswith("fpd_xor_")
        or s.startswith("fpd_and_")
        or s.startswith("fpd_or_")
    ):
        return "FPd-intra"
    if s.startswith("fpd_"):
        return "unsupported"
    return "unsupported"


@app.get("/scenario/{scenario_id}/odrl-index")
async def odrl_fragment_policies_index(scenario_id: str) -> dict[str, Any]:
    """
    Liste les exports JSON-LD par étape (``partial_template_only``, ``final_merged``),
    par fragment, fichiers plats ; ``ui_category`` est déduit du nom de fichier.
    """
    if not _scenario_id_ok(scenario_id):
        raise HTTPException(status_code=400, detail="Identifiant de scénario invalide.")
    base = _odrl_output_root(scenario_id)
    if not base.is_dir():
        return {"scenario_id": scenario_id, "stages": []}
    stages: list[dict[str, Any]] = []
    for stage_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        frag_map: dict[str, Any] = {}
        for frag in sorted(stage_dir.iterdir()):
            if not frag.is_dir():
                continue
            fid = frag.name
            files_meta: list[dict[str, str]] = []
            for p in sorted(frag.glob("*.jsonld")):
                rel = str(p.relative_to(stage_dir)).replace("\\", "/")
                files_meta.append(
                    {
                        "relative": rel,
                        "basename": p.name,
                        "ui_category": _ui_category_from_stem(p.stem),
                    }
                )
            if files_meta:
                frag_map[fid] = files_meta
        stages.append(
            {
                "name": stage_dir.name,
                "path": str(stage_dir),
                "fragments": frag_map,
            }
        )
    return {"scenario_id": scenario_id, "stages": stages}


@app.get("/scenario/{scenario_id}/odrl-file")
async def odrl_fragment_policy_file(
    scenario_id: str,
    rel: str = Query(
        ...,
        description="Chemin relatif depuis odrl_fragment_policies/ (ex. final_merged/f1/FPa_0_act.jsonld).",
    ),
) -> dict[str, Any]:
    if not _scenario_id_ok(scenario_id):
        raise HTTPException(status_code=400, detail="Identifiant de scénario invalide.")
    base = _odrl_output_root(scenario_id)
    if not base.is_dir():
        raise HTTPException(status_code=404, detail="Aucun export ODRL pour ce scénario.")
    rp = Path(rel.strip())
    if rp.is_absolute() or ".." in rp.parts:
        raise HTTPException(status_code=400, detail="Chemin invalide.")
    target = (base / rp).resolve()
    try:
        target.relative_to(base)
    except ValueError as e:
        raise HTTPException(status_code=403, detail="Chemin hors du dossier autorisé.") from e
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"JSON invalide : {e}") from e
    return {
        "path": str(target),
        "relative": rel.replace("\\", "/"),
        "data": data,
    }


@app.post("/scenario/process")
async def scenario_process(
    bpmn: UploadFile = File(...),
    odrl: UploadFile = File(...),
    temperature: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """
    BPMN + policies ODRL (JSON) : parsing, fragmentation LLM, création de ``src/dataset/scenarioN/``
    avec ``bp_global.json``, ``B2P.json``, ``fragments.json``, ``fragments_enhanced.json``.
    """
    _logger.info(
        "POST /scenario/process — fichiers reçus : bpmn=%r odrl=%r",
        bpmn.filename,
        odrl.filename,
    )
    raw_b = await bpmn.read()
    raw_o = await odrl.read()
    if not raw_b:
        raise HTTPException(status_code=400, detail="Fichier BPMN vide.")
    if not raw_o:
        raise HTTPException(status_code=400, detail="Fichier policies vide.")
    bpmn_name = (bpmn.filename or "upload.bpmn").lower()
    suffix = Path(bpmn_name).suffix.lower() if Path(bpmn_name).suffix else ".bpmn"
    try:
        return await asyncio.to_thread(
            _process_scenario_upload_sync,
            raw_b,
            suffix,
            raw_o,
            llm_temperature=temperature,
            llm_seed=seed,
        )
    except ValueError as e:
        _logger.warning("POST /scenario/process: %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Traitement scénario impossible : {e!s}",
        ) from e


@app.post("/bpmn/fragment")
async def fragment_uploaded_bpmn(
    file: UploadFile = File(...),
    temperature: float = 0.0,
    seed: int = 42,
    threshold: Optional[float] = None,
) -> dict[str, Any]:
    """
    Reçoit un BPMN (.bpmn / .xml), parse, fragmente (LLM), écrit un unique JSON
    dans le dossier Téléchargements de l'utilisateur Windows (``Path.home() / Downloads``),
    sauf si ``FRAGMENT_EXPORT_DIR`` est défini dans l'environnement.

    ``threshold`` est déprécié : si fourni il est utilisé comme température LLM pour compatibilité.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Fichier vide.")

    orig = (file.filename or "upload.bpmn").lower()
    suffix = Path(orig).suffix.lower() if Path(orig).suffix else ".bpmn"

    temp = threshold if threshold is not None else temperature
    try:
        return await asyncio.to_thread(
            _parse_and_fragment_bpmn_bytes,
            raw,
            suffix,
            llm_temperature=temp,
            llm_seed=seed,
        )
    except ValueError as e:
        _logger.warning("POST /bpmn/fragment validation: %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fragmentation impossible : {e!s}",
        ) from e
