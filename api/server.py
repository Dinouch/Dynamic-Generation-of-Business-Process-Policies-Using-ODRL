"""
FastAPI server: runs the async pipeline + SSE for HITL and final result.

Launch (from BPFragmentODRLProject root) ::
    uvicorn api.server:app --reload --host 127.0.0.1 --port 8765

CORS (Next.js frontend): defaults to ``localhost:3000`` and ``127.0.0.1:3000``.
Other origin (e.g. LAN IP): environment variable ``CORS_ORIGINS`` (comma-separated list)
or ``CORS_ORIGIN_REGEX`` (e.g. ``http://192\\.168\\..*``).
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
        description="Directory under src/dataset/",
    )
    human_timeout_s: float = Field(default=120.0, ge=1.0, le=3600.0)
    process_title: Optional[str] = Field(
        default=None,
        description="Business label shown in HITL (e.g. Credit Application BP).",
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
            raise ValueError("decision must be agree or refuse")
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
    """Parse BPMN XML from bytes, LLM fragmentation, write ``fragments`` + ``fragments_enhanced``."""
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
            raise ValueError("BPMN parsing failed (empty file, invalid XML, or no process).")
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
    raise ValueError("The policies file must be a JSON object or array of objects.")


def _process_scenario_upload_sync(
    bpmn_bytes: bytes,
    bpmn_suffix: str,
    odrl_bytes: bytes,
    *,
    llm_temperature: float,
    llm_seed: int,
) -> dict[str, Any]:
    """Parse BPMN, LLM fragmentation, write ``scenarioN/`` with bp_global.json, B2P.json, fragments.json, fragments_enhanced.json."""
    import tempfile

    _logger.info(
        "Scenario (sync): %s BPMN bytes, %s policy bytes — parsing…",
        len(bpmn_bytes),
        len(odrl_bytes),
    )

    odrl_text = odrl_bytes.decode("utf-8-sig")
    try:
        raw_pol = json.loads(odrl_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"ODRL policies: invalid JSON ({e})") from e
    policies = _normalize_b2p_policies(raw_pol)
    if not policies:
        raise ValueError("ODRL policies: no valid policy object in JSON.")

    suf = bpmn_suffix if bpmn_suffix.lower() in (".bpmn", ".xml") else ".bpmn"
    tmp_path: Optional[str] = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=suf, prefix="bpmn_upload_")
        os.write(fd, bpmn_bytes)
        os.close(fd)

        parser = BPMNParser()
        model = parser.parse_file(tmp_path)
        if not model:
            raise ValueError("BPMN parsing failed (empty file, invalid XML, or no process).")

        _logger.info(
            "BPMN parsed OK — LLM fragmentation in progress (multiple calls possible, often 1–5 min; "
            "the terminal may appear « frozen » during this time)."
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
            "POST /scenario/process completed — %s (%d fragments) → %s",
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
            await bridge.emit_log(f"Loading scenario « {body.scenario_id} »…")
            bp_model, fragments, b2p_policies = load_scenario(body.scenario_id, base_dir=base_dir)
            await bridge.emit_log("Async pipeline started (HITL via API).")

            _logged_agent5 = False

            async def _trace_acl(env: ACLEnvelope) -> None:
                nonlocal _logged_agent5
                if env.receiver == "agent5" and not _logged_agent5:
                    _logged_agent5 = True
                    await bridge.emit_log(
                        "Audit step (agent 5) in progress — the LLM may take a while; the connection stays open (SSE keep-alive)."
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
            oep = (result.odrl_export_paths or {}) if result else {}
            if oep:
                await bridge.emit_log(
                    "ODRL export written under output/"
                    f"{body.scenario_id}/odrl_fragment_policies/ "
                    f"({', '.join(sorted(oep.keys()))})."
                )
            else:
                await bridge.emit_log(
                    f"Warning: no ODRL export on disk for {body.scenario_id!r}. "
                    "Check uvicorn logs (Agent 4 / 5)."
                )
            await bridge.emit_done(_serialize_result(result))
        except Exception as e:
            await bridge.emit_error(str(e))
        finally:
            RUN_TASKS.pop(run_id, None)

    RUN_TASKS[run_id] = asyncio.create_task(_runner())
    return {"run_id": run_id}


_RUN_LOST_MSG = (
    "This run_id does not exist in memory (server restarted with --reload, "
    "or session too old). Restart « Run pipeline » from the interface."
)


@app.get("/runs/{run_id}/ready")
async def run_session_ready(run_id: str) -> dict[str, Any]:
    """Check that the session exists (useful for the client to distinguish SSE 404)."""
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail=_RUN_LOST_MSG)
    return {"ok": True, "run_id": run_id}


@app.get("/runs/{run_id}/events")
async def stream_events(run_id: str) -> StreamingResponse:
    bridge = RUNS.get(run_id)
    if bridge is None:
        raise HTTPException(status_code=404, detail=_RUN_LOST_MSG)

    async def gen() -> Any:
        # SSE comments (: …) to prevent proxy/browser disconnect during long phases (e.g. agent 5).
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
            f"[API] POST /hitl rejected — reply_with={body.reply_with!r} "
            f"(unknown or already answered; pending: {bridge.pending_hitl_keys()!r})"
        )
        raise HTTPException(status_code=400, detail="reply_with unknown or already answered")
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
    """``output/{scenarioN}/odrl_fragment_policies/`` at project root (outside ``dataset``)."""
    return (_ROOT / "output" / scenario_id / "odrl_fragment_policies").resolve()


_KNOWN_ODRL_STAGES = frozenset(
    {"partial_template_only", "final_merged", "final_validated", "generated"}
)


def _jsonld_files_in_dir(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.jsonld"))


def _collect_fragment_map(container: Path, base: Path) -> dict[str, list[dict[str, str]]]:
    """Group ``*.jsonld`` files by fragment subfolder; ``relative`` from ``base``."""
    frag_map: dict[str, list[dict[str, str]]] = {}
    if not container.is_dir():
        return frag_map
    for child in sorted(container.iterdir()):
        if not child.is_dir():
            continue
        files_meta: list[dict[str, str]] = []
        for policy_file in _jsonld_files_in_dir(child):
            rel = str(policy_file.relative_to(base)).replace("\\", "/")
            files_meta.append(
                {
                    "relative": rel,
                    "basename": policy_file.name,
                    "ui_category": _ui_category_from_stem(policy_file.stem),
                }
            )
        if files_meta:
            frag_map[child.name] = files_meta
    return frag_map


def _fragment_dirs_at_root(base: Path) -> bool:
    """
    Detect flat export ``odrl_fragment_policies/f1/*.jsonld`` (without stage subfolder).
    """
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if child.name in _KNOWN_ODRL_STAGES:
            return False
        if _jsonld_files_in_dir(child):
            return True
    return False


def _stage_sort_key(name: str) -> tuple[int, str]:
    order = {
        "final_validated": 0,
        "final_merged": 1,
        "partial_template_only": 2,
        "generated": 3,
    }
    return (order.get(name, 99), name)


def _list_odrl_export_stages(base: Path) -> list[dict[str, Any]]:
    if not base.is_dir():
        return []

    stages: list[dict[str, Any]] = []

    if _fragment_dirs_at_root(base):
        frag_map = _collect_fragment_map(base, base)
        if frag_map:
            stages.append(
                {
                    "name": "generated",
                    "path": str(base),
                    "fragments": frag_map,
                }
            )
        return stages

    for stage_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        frag_map = _collect_fragment_map(stage_dir, base)
        if frag_map:
            stages.append(
                {
                    "name": stage_dir.name,
                    "path": str(stage_dir),
                    "fragments": frag_map,
                }
            )

    stages.sort(key=lambda s: _stage_sort_key(str(s.get("name", ""))))
    return stages


def _ui_category_from_stem(stem: str) -> str:
    """
    Display category derived from exported file prefix (see ``PolicyProjectionAgent.export``).
    Returns: FPa | FPd-intra | FPd-inter | unmapped
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
        return "unmapped"
    return "unmapped"


@app.get("/scenario/{scenario_id}/odrl-index")
async def odrl_fragment_policies_index(scenario_id: str) -> dict[str, Any]:
    """
    List JSON-LD exports by stage (``final_validated``, ``partial_template_only``, …)
    or flat export ``fN/*.jsonld``; ``relative`` is relative to ``odrl_fragment_policies/``.
    """
    if not _scenario_id_ok(scenario_id):
        raise HTTPException(status_code=400, detail="Invalid scenario identifier.")
    base = _odrl_output_root(scenario_id)
    stages = _list_odrl_export_stages(base)
    return {"scenario_id": scenario_id, "stages": stages}


@app.get("/scenario/{scenario_id}/odrl-file")
async def odrl_fragment_policy_file(
    scenario_id: str,
    rel: str = Query(
        ...,
        description="Relative path from odrl_fragment_policies/ (e.g. final_merged/f1/FPa_0_act.jsonld).",
    ),
) -> dict[str, Any]:
    if not _scenario_id_ok(scenario_id):
        raise HTTPException(status_code=400, detail="Invalid scenario identifier.")
    base = _odrl_output_root(scenario_id)
    if not base.is_dir():
        raise HTTPException(status_code=404, detail="No ODRL export for this scenario.")
    rp = Path(rel.strip())
    if rp.is_absolute() or ".." in rp.parts:
        raise HTTPException(status_code=400, detail="Invalid path.")
    target = (base / rp).resolve()
    try:
        target.relative_to(base)
    except ValueError as e:
        raise HTTPException(status_code=403, detail="Path outside authorized directory.") from e
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {e}") from e
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
    BPMN + ODRL policies (JSON): parsing, LLM fragmentation, creation of ``src/dataset/scenarioN/``
    with ``bp_global.json``, ``B2P.json``, ``fragments.json``, ``fragments_enhanced.json``.
    """
    _logger.info(
        "POST /scenario/process — files received: bpmn=%r odrl=%r",
        bpmn.filename,
        odrl.filename,
    )
    raw_b = await bpmn.read()
    raw_o = await odrl.read()
    if not raw_b:
        raise HTTPException(status_code=400, detail="Empty BPMN file.")
    if not raw_o:
        raise HTTPException(status_code=400, detail="Empty policies file.")
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
            detail=f"Scenario processing failed: {e!s}",
        ) from e


@app.post("/bpmn/fragment")
async def fragment_uploaded_bpmn(
    file: UploadFile = File(...),
    temperature: float = 0.0,
    seed: int = 42,
    threshold: Optional[float] = None,
) -> dict[str, Any]:
    """
    Receives a BPMN (.bpmn / .xml), parses, fragments (LLM), writes a single JSON
    to the user's Windows Downloads folder (``Path.home() / Downloads``),
    unless ``FRAGMENT_EXPORT_DIR`` is set in the environment.

    ``threshold`` is deprecated: if provided it is used as LLM temperature for compatibility.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

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
            detail=f"Fragmentation failed: {e!s}",
        ) from e


class ExecutionRunBody(BaseModel):
    scenario_id: str = Field(
        default="scenario018",
        description="Directory under src/dataset/ (or relative/absolute path to a scenario)",
    )
    policies_dir: Optional[str] = Field(
        default=None,
        description="Post-pipeline ODRL export; otherwise ground truth / scenario odrl_policies",
    )
    mode: str = Field(default="simulation", description="simulation | camunda7 | camunda8")
    variables: dict[str, Any] = Field(default_factory=dict)
    max_steps: int = Field(default=64, ge=1, le=256)
    bpmn_path: Optional[str] = Field(default=None, description="Required for camunda mode")
    process_definition_key: Optional[str] = Field(
        default=None,
        description="Camunda process key (process id)",
    )


def _resolve_scenario_path(scenario_id: str) -> Path:
    sid = (scenario_id or "").strip()
    p = Path(sid)
    if p.is_dir():
        return p.resolve()
    under_src = _SRC / "dataset" / sid
    if under_src.is_dir():
        return under_src.resolve()
    raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_id}")


def _serialize_execution_result(result: Any) -> dict[str, Any]:
    return {
        "success": result.success,
        "mode": result.mode.value if hasattr(result.mode, "value") else str(result.mode),
        "error": result.error,
        "summary": result.summary,
        "camunda_process_instance_id": result.camunda_process_instance_id,
        "steps": [
            {
                "index": s.step_index,
                "activity": s.activity,
                "fragment_id": s.fragment_id,
                "allowed": s.decision.allowed,
                "reason": s.decision.reason,
                "enabled_rules": s.enabled_rules_after,
                "variables": s.variables_snapshot,
            }
            for s in (result.steps or [])
        ],
    }


@app.post("/execution/run")
async def execution_run(body: ExecutionRunBody) -> dict[str, Any]:
    """
    Run a scenario after fragment policy generation (FPa/FPd).

    ``simulation`` mode: local graph traversal + ODRL PDP.
    ``camunda7`` mode: BPMN deployment + Camunda instances (ODRL PEP on each task).
    """
    from execution.engine import ExecutionEngine
    from execution.models import RuntimeMode

    scenario_path = _resolve_scenario_path(body.scenario_id)
    mode_map = {
        "simulation": RuntimeMode.SIMULATION,
        "camunda7": RuntimeMode.CAMUNDA_7,
        "camunda8": RuntimeMode.CAMUNDA_8,
    }
    mode_key = (body.mode or "simulation").strip().lower()
    if mode_key not in mode_map:
        raise HTTPException(status_code=422, detail="mode must be simulation, camunda7 or camunda8")

    pol_dir = body.policies_dir
    if pol_dir and not Path(pol_dir).is_dir():
        pol_dir = str((_ROOT / pol_dir).resolve()) if (_ROOT / pol_dir).is_dir() else pol_dir

    try:
        engine = ExecutionEngine.from_scenario(
            str(scenario_path),
            policies_dir=pol_dir,
            mode=mode_map[mode_key],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    if mode_key == "simulation":
        result = await asyncio.to_thread(
            engine.run_simulation,
            initial_variables=body.variables,
            max_steps=body.max_steps,
        )
    else:
        if not body.bpmn_path or not body.process_definition_key:
            raise HTTPException(
                status_code=422,
                detail="bpmn_path and process_definition_key required for Camunda",
            )
        bpmn = Path(body.bpmn_path)
        if not bpmn.is_file():
            bpmn = _ROOT / body.bpmn_path
        if not bpmn.is_file():
            raise HTTPException(status_code=404, detail=f"BPMN not found: {body.bpmn_path}")
        result = await asyncio.to_thread(
            engine.run_camunda,
            bpmn_path=str(bpmn.resolve()),
            process_definition_key=body.process_definition_key,
            initial_variables=body.variables,
        )

    out = _serialize_execution_result(result)
    out["scenario_path"] = str(scenario_path)
    out["policies_loaded"] = sum(
        len(fps.all_policies()) for fps in engine.bundle.fp_results.values()
    )
    return out


@app.get("/execution/health")
async def execution_health(
    camunda_url: Optional[str] = Query(default=None, description="Camunda 7 engine-rest URL"),
) -> dict[str, Any]:
    """Check Camunda availability (optional)."""
    from execution.camunda_client import CamundaRestClient

    client = CamundaRestClient(base_url=camunda_url) if camunda_url else CamundaRestClient()
    return {"camunda_reachable": client.health(), "camunda_base_url": client.base_url}
