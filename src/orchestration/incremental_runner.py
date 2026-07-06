"""
Run **I0–I5** ablation profiles for incremental evaluation.

Profiles (cumulative)
---------------------
- **I0** — *Prompt-only* baseline: B2P + ``fragments_enhanced.json`` + ``bp_model`` → **one LLM prompt**
  generates the **entire** scenario (FPa + FPd). **No** Agent 1, **no** Agent 4 templates.
- **I1** — **Agent 1** (graph) + **Agent 4** (deterministic templates); **no** unmapped,
  **no** Agent 5.
- **I2** — I1 + **Agent 2** (unmapped formulation); proposals **auto-accepted** → FPd by Agent 4.
- **I3** — I2 + **syntax** loop Agent 5 ↔ Agent 4 (sequential, **without** Agent 3).
- **I4** — **Async** pipeline: I3 + **semantic** loops Agent 3; **without** HITL.
- **I5** — Full async pipeline + **HITL** (``i5_use_auto_agree=True`` = batch without input).

Output — ``<project_root>/output/<Ix>/<scenario_id>/`` (+ ``incremental_pipeline_run_meta.json``).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Project root (…/src/orchestration/incremental_runner.py → parents[2])
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_PROJECT_ROOT / "system_evaluation") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "system_evaluation"))

from agents.Agent_3.constraint_validator import (
    ConstraintValidator,
    ValidationDecision,
    ValidationReport,
    ValidationResult,
)
from agents.exception_handling_agent import UnmappedCaseFormulator, UnmappedCaseProposal
from agents.pipeline_registry import COVERED_PATTERNS
from agents.policy_auditor import PolicyAuditor
from agents.policy_projection_agent import PolicyProjectionAgent
from agents.structural_analyzer import (
    AgentMessage,
    EnrichedGraph,
    MessageType,
    StructuralAnalyzer,
)

from orchestration.incremental_i0_runner import run_i0_llm_baseline
from orchestration.incremental_sequential_syntax import run_syntax_correction_loop


@dataclass
class IncrementalRunResult:
    profile: str
    scenario_id: str
    output_dir: Path
    policies_json: Path
    mode: str  # "sequential" | "async"
    message: str = ""
    odrl_subdir: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)
    #: **Actual** pipeline run duration (analyze → project + audit / full async), wall-clock seconds.
    pipeline_wall_time_s: Optional[float] = None


def load_pipeline_run_meta(
    project_root: Path | str,
    profile: str,
    scenario_id: str,
) -> dict[str, Any] | None:
    """
    Read ``output/<Ix>/<scenario_id>/incremental_pipeline_run_meta.json`` if present
    (written after each successful ``run_incremental_profile`` / ``arun_async_incremental``).

    Expected return: ``pipeline_wall_time_s``, ``profile``, ``scenario_id``, ``mode``, …
    """
    root = Path(project_root).resolve()
    p = _output_base(root, profile, scenario_id) / "incremental_pipeline_run_meta.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def load_incremental_bert_fragments(
    project_root: Path | str,
    profile: str,
    scenario_id: str,
) -> dict[str, Any] | None:
    """
    Read ``output/<Ix>/<scenario_id>/incremental_bert_fragments.json`` if present.

    Implementation: ``metrics.incremental_bert_io`` (lazy import to avoid cycles).
    """
    from metrics.incremental_bert_io import load_incremental_bert_fragments as _load

    return _load(project_root, profile, scenario_id)


def save_incremental_bert_fragments(
    project_root: Path | str,
    profile: str,
    scenario_id: str,
    bert_fragments: dict[str, Any],
) -> Path:
    """Write ``incremental_bert_fragments.json`` under ``output/<Ix>/<scenario_id>/``."""
    from metrics.incremental_bert_io import save_incremental_bert_fragments as _save

    return _save(project_root, profile, scenario_id, bert_fragments)


def _write_pipeline_run_meta(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "incremental_pipeline_run_meta.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


_VALID_INCREMENT_PROFILES = frozenset(f"I{i}" for i in range(6))


def _validation_report_accept_all_raw_proposals(proposals: list[dict[str, Any]]) -> ValidationReport:
    """I2: Agent 2 proposals accepted as-is for projection (without A3)."""
    results: list[ValidationResult] = []
    for raw in proposals or []:
        if not isinstance(raw, dict):
            continue
        prop = UnmappedCaseProposal.from_dict(raw)
        results.append(
            ValidationResult(
                proposal=prop,
                decision=ValidationDecision.ACCEPTED,
                decision_level="incremental_I2_skip_A3",
                explanation="Auto-accept: Agent 3 disabled for this increment.",
            )
        )
    return ValidationReport(results=results)


def _api_keys_present() -> bool:
    """True if OpenAI key alone, or Azure key + endpoint (same logic as async pipeline)."""
    if (os.environ.get("OPENAI_API_KEY") or "").strip():
        return True
    azk = (os.environ.get("AZURE_OPENAI_KEY") or os.environ.get("AZURE_OPENAI_API_KEY") or "").strip()
    aze = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip()
    return bool(azk and aze)


def _ingest_env_file_manual(path: Path) -> None:
    """
    Fallback (same spirit as ``main._load_env``) if ``python-dotenv`` is not installed
    in the kernel: one ``KEY=VALUE`` line per entry, ``utf-8-sig`` encoding (BOM), optional
    quotes around the value.
    """
    if not path.is_file():
        return
    try:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            os.environ[k] = v
    except OSError:
        pass


def _load_project_dotenv(project_root: Path) -> None:
    """
    Load ``.env`` files like ``main``: project root then ``<root>/src/.env`` (later files
    override keys). Always includes the repo ``…/src`` ``.env``
    (``…/src/orchestration/`` → parent = ``…/src``), regardless of ``project_root``.

    **Without** ``python-dotenv`` in Jupyter, we previously returned immediately:
    nothing was loaded. Now uses manual reading (like ``main``).
    """
    _package_src = Path(__file__).resolve().parents[1]  # …/src
    seen: set[Path] = set()
    paths: list[Path] = []
    for candidate in (project_root / ".env", project_root / "src" / ".env", _package_src / ".env"):
        if not candidate.is_file():
            continue
        rp = candidate.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        paths.append(candidate)

    load_fn = None
    try:
        from dotenv import load_dotenv as load_fn
    except ImportError:  # pragma: no cover
        pass
    for p in paths:
        if load_fn is not None:
            load_fn(p, override=True)
        else:
            _ingest_env_file_manual(p)


def _resolve_async_llm_credentials(
    *,
    api_key: str | None,
    azure_endpoint: str | None,
    azure_api_version: str | None,
    azure_deployment: str | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """
    Resolve classic OpenAI key and Azure parameters (aligned with ``UnmappedCaseFormulator``).
    """
    ak = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip() or None
    az_end = (azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip() or None
    az_ver = (azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or "").strip() or None
    az_dep = (azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or "").strip() or None
    az_key = (os.environ.get("AZURE_OPENAI_KEY") or os.environ.get("AZURE_OPENAI_API_KEY") or "").strip()
    openai_ok = bool(ak)
    azure_ok = bool(az_key and az_end)
    if not openai_ok and not azure_ok:
        raise ValueError(
            "Missing LLM keys for I4/I5 (async pipeline). Set in the environment or in a `.env` "
            "(at repo root and/or under `src/`, like `main.py`): "
            "OPENAI_API_KEY — or — AZURE_OPENAI_KEY (or AZURE_OPENAI_API_KEY) + AZURE_OPENAI_ENDPOINT. "
            "You can also pass `api_key=...` to `arun_async_incremental`."
        )
    return ak, az_end or None, az_ver or None, az_dep or None


def _output_base(project_root: Path, profile: str, scenario_id: str) -> Path:
    sid = (scenario_id or "scenario1").strip()
    return (project_root / "output" / profile / sid).resolve()


def _fp_results_to_policy_list(fp_results: dict) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _fid, fps in (fp_results or {}).items():
        for pol in fps.all_policies():
            if isinstance(pol, dict):
                out.append(pol)
            else:
                out.append(dict(pol) if hasattr(pol, "items") else {})
    return out


def _write_policies_json(policies: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policies, ensure_ascii=False, indent=2), encoding="utf-8")


def _formulate_proposals_cfp(
    enriched: EnrichedGraph,
    *,
    api_key: str | None,
    azure_endpoint: str | None,
    azure_api_version: str | None,
    azure_deployment: str | None,
) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    def on_send(msg: AgentMessage) -> None:
        if msg.msg_type == MessageType.UNMAPPED_PROPOSALS:
            pl = msg.payload or {}
            captured.extend(list(pl.get("unmapped_proposals") or []))

    ag = UnmappedCaseFormulator(
        covered_patterns=COVERED_PATTERNS,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        azure_deployment=azure_deployment,
    )
    ag.register_send_callback(on_send)
    cfp = AgentMessage(
        sender="pipeline",
        recipient=ag.AGENT_NAME,
        msg_type=MessageType.CFP_UNMAPPED,
        payload={"enriched_graph": enriched, "utterance": "incremental I2/I3 CFP"},
        loop_turn=0,
    )
    ag.receive(cfp)
    return captured


def _run_sequential(
    profile: str,
    *,
    scenario_id: str,
    bp_model: dict[str, Any],
    fragments: list[dict[str, Any]],
    b2p_policies: list[dict[str, Any]],
    project_root: Path,
    api_key: str | None,
    azure_endpoint: str | None,
    azure_api_version: str | None,
    azure_deployment: str | None,
) -> IncrementalRunResult:
    wall_t0 = time.perf_counter()
    _load_project_dotenv(project_root)
    api_key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip() or None
    azure_endpoint = (azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip() or None
    azure_api_version = (azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or "").strip() or None
    azure_deployment = (azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or "").strip() or None

    out_dir = _output_base(project_root, profile, scenario_id)
    odrl_dir = out_dir / "odrl_fragment_policies"
    if odrl_dir.is_dir():
        shutil.rmtree(odrl_dir, ignore_errors=True)

    proposals: list[dict[str, Any]] = []
    fp_results: dict[str, Any]
    enriched: EnrichedGraph | None = None
    vr = ValidationReport(results=[])

    # ── I0: LLM baseline (single prompt, no Agent 4 templates) ──
    if profile == "I0":
        if not _api_keys_present():
            raise RuntimeError(
                "I0 (baseline prompt-only) requires OPENAI_API_KEY or Azure OpenAI "
                "(OPENAI / AZURE_OPENAI_KEY + AZURE_OPENAI_ENDPOINT)."
            )
        import importlib

        import agents.Agent_4.i0_llm_baseline as _i0_llm_mod
        import orchestration.incremental_i0_runner as _i0_run_mod

        importlib.reload(_i0_llm_mod)
        _i0_run_mod = importlib.reload(_i0_run_mod)
        fp_results, _i0_elapsed = _i0_run_mod.run_i0_llm_baseline(
            scenario_id=scenario_id,
            bp_model=bp_model,
            fragments=fragments,
            b2p_policies=b2p_policies,
            project_root=project_root,
            out_dir=out_dir,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            azure_deployment=azure_deployment,
        )
        policies = _fp_results_to_policy_list(fp_results)
        pj = out_dir / "policies.json"
        _write_policies_json(policies, pj)
        elapsed = round(time.perf_counter() - wall_t0, 4)
        _write_pipeline_run_meta(
            out_dir,
            {
                "profile": profile,
                "scenario_id": scenario_id,
                "mode": "i0_llm_baseline",
                "pipeline_wall_time_s": elapsed,
                "n_policies": len(policies),
                "n_unmapped_proposals": 0,
            },
        )
        return IncrementalRunResult(
            profile=profile,
            scenario_id=scenario_id,
            output_dir=out_dir,
            policies_json=pj,
            mode="i0_llm_baseline",
            message="ok",
            odrl_subdir=str(odrl_dir),
            raw={"n_policies": len(policies)},
            pipeline_wall_time_s=elapsed,
        )

    # ── I1–I3: Agent 1 + Agent 4 (± Agent 2, ± A5 syntax loop) ──
    analyzer = StructuralAnalyzer(
        bp_model,
        fragments,
        b2p_policies,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        azure_deployment=azure_deployment,
    )
    enriched = analyzer.analyze()

    if profile in ("I2", "I3") and (enriched.unmapped_patterns or []):
        if not _api_keys_present():
            print("[incremental_runner][WARN] I2/I3: API key missing — empty unmapped")
        else:
            try:
                proposals = _formulate_proposals_cfp(
                    enriched,
                    api_key=api_key,
                    azure_endpoint=azure_endpoint,
                    azure_api_version=azure_api_version,
                    azure_deployment=azure_deployment,
                )
                (out_dir / "proposals_unmapped.json").write_text(
                    json.dumps(proposals, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as e:
                print(f"[incremental_runner][WARN] Agent 2 : {e}")
                proposals = []

    if profile in ("I2", "I3"):
        vr = _validation_report_accept_all_raw_proposals(proposals)
    else:
        vr = ValidationReport(results=[])

    projector = PolicyProjectionAgent(
        enriched_graph=enriched,
        validation_report=vr,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        azure_deployment=azure_deployment,
    )
    projector._increment_profile = profile  # noqa: SLF001

    if profile == "I3":
        auditor = PolicyAuditor(
            fp_results={},
            enriched_graph=enriched,
            raw_b2p=b2p_policies,
            validation_report=vr,
        )
        fp_results = run_syntax_correction_loop(enriched, vr, projector, auditor, increment_profile="I3")
    else:
        fp_results = projector.project(enriched, vr)
    projector.export(fp_results, output_dir=str(odrl_dir))

    policies = _fp_results_to_policy_list(fp_results)
    pj = out_dir / "policies.json"
    _write_policies_json(policies, pj)

    elapsed = round(time.perf_counter() - wall_t0, 4)
    _write_pipeline_run_meta(
        out_dir,
        {
            "profile": profile,
            "scenario_id": scenario_id,
            "mode": "sequential",
            "pipeline_wall_time_s": elapsed,
            "n_policies": len(policies),
            "n_unmapped_proposals": len(proposals),
        },
    )

    return IncrementalRunResult(
        profile=profile,
        scenario_id=scenario_id,
        output_dir=out_dir,
        policies_json=pj,
        mode="sequential",
        message="ok",
        odrl_subdir=str(odrl_dir),
        raw={"n_policies": len(policies), "n_unmapped_proposals": len(proposals)},
        pipeline_wall_time_s=elapsed,
    )


def _async_pipeline_flags(
    profile: str,
    *,
    i5_use_auto_agree: bool,
) -> tuple[bool, bool]:
    """
    Return ``(accept_all_unmapped, enable_hitl)`` for I4 / I5.

    - **I4**: Agent 3 semantic validation (no blind acceptance), no HITL.
    - **I5** + ``i5_use_auto_agree``: batch without input (orchestrator acceptance).
    - **I5** without auto: interactive HITL (agree/refuse).
    """
    p = (profile or "").strip().upper()
    if p == "I4":
        return False, False
    if p == "I5":
        if i5_use_auto_agree:
            return True, False
        return False, True
    return False, False


async def _run_async_profile(
    profile: str,
    *,
    scenario_id: str,
    bp_model: dict[str, Any],
    fragments: list[dict[str, Any]],
    b2p_policies: list[dict[str, Any]],
    project_root: Path,
    human_timeout_s: float,
    i5_use_auto_agree: bool = False,
    api_key: str | None = None,
    azure_endpoint: str | None = None,
    azure_api_version: str | None = None,
    azure_deployment: str | None = None,
) -> IncrementalRunResult:
    from orchestration.async_pipeline import run_pipeline_async

    wall_t0 = time.perf_counter()
    accept_all, enable_hitl = _async_pipeline_flags(profile, i5_use_auto_agree=i5_use_auto_agree)
    print(
        f"[incremental_runner] profile={profile.strip().upper()} "
        f"accept_all_unmapped={accept_all} enable_hitl={enable_hitl}"
    )

    _load_project_dotenv(project_root)
    ak, az_end, az_ver, az_dep = _resolve_async_llm_credentials(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        azure_deployment=azure_deployment,
    )

    # Do not close the pipeline on a partial bundle while interactive HITL is open.
    os.environ["HITL_AUTOCLOSE_AFTER_PARTIAL"] = "0"

    result = await run_pipeline_async(
        bp_model=bp_model,
        fragments=fragments,
        b2p_policies=b2p_policies,
        human_timeout_s=human_timeout_s,
        api_key=ak,
        azure_endpoint=az_end,
        azure_api_version=az_ver,
        azure_deployment=az_dep,
        accept_all_unmapped=accept_all,
        enable_hitl=enable_hitl,
        scenario_id=scenario_id,
        increment_profile=profile.strip().upper(),
    )

    from benchmark.policy_collect import load_policies_from_export_directory

    paths = result.odrl_export_paths or {}
    edir = (
        paths.get("final_validated")
        or paths.get("final_merged")
        or paths.get("partial_template_only")
        or ""
    )
    policies: list[dict[str, Any]] = []
    edir_path = Path(edir) if edir else None
    if edir_path and edir_path.is_dir():
        policies = load_policies_from_export_directory(str(edir_path))

    dest = _output_base(project_root, profile, scenario_id)
    dest.mkdir(parents=True, exist_ok=True)
    odrl_dest = dest / "odrl_fragment_policies"
    if edir_path and edir_path.is_dir():
        if odrl_dest.is_dir():
            shutil.rmtree(odrl_dest, ignore_errors=True)
        shutil.copytree(edir_path, odrl_dest)

    pj = dest / "policies.json"
    _write_policies_json(policies, pj)

    elapsed = round(time.perf_counter() - wall_t0, 4)
    _write_pipeline_run_meta(
        dest,
        {
            "profile": profile,
            "scenario_id": scenario_id,
            "mode": "async",
            "pipeline_wall_time_s": elapsed,
            "n_policies": len(policies),
            "pipeline_status": str(result.status or ""),
            "is_valid": bool(result.is_valid),
            "increment_profile": profile.strip().upper(),
        },
    )

    return IncrementalRunResult(
        profile=profile,
        scenario_id=scenario_id,
        output_dir=dest,
        policies_json=pj,
        mode="async",
        message=str(result.status or ""),
        odrl_subdir=str(odrl_dest) if odrl_dest.is_dir() else (edir or None),
        raw={"pipeline_status": result.status, "is_valid": result.is_valid, "export": paths},
        pipeline_wall_time_s=elapsed,
    )


def run_incremental_profile(
    profile: str,
    *,
    scenario_id: str,
    bp_model: dict[str, Any],
    fragments: list[dict[str, Any]],
    b2p_policies: list[dict[str, Any]],
    project_root: Path | None = None,
    api_key: str | None = None,
    azure_endpoint: str | None = None,
    azure_api_version: str | None = None,
    azure_deployment: str | None = None,
    human_timeout_s: float = 120.0,
    i5_use_auto_agree: bool = False,
) -> IncrementalRunResult:
    """
    Run the requested profile (I0 … I5) and write ``output/<Ix>/<scenario_id>/policies.json``.

    I4: async pipeline + Agent 3 semantic loops (no HITL).
    I5: full pipeline + HITL; ``i5_use_auto_agree=True`` = batch without input.
    """
    root = (project_root or _PROJECT_ROOT).resolve()
    p = (profile or "I0").strip().upper()
    if p == "I6":
        raise ValueError(
            "Profile I6 no longer exists: the full chain + HITL corresponds to **I5**. "
            "Use i5_use_auto_agree=True to simulate human agreement in batch."
        )
    if p not in _VALID_INCREMENT_PROFILES:
        raise ValueError(f"Unknown profile: {profile!r} (expected I0..I5)")

    if p in ("I0", "I1", "I2", "I3"):
        return _run_sequential(
            p,
            scenario_id=scenario_id,
            bp_model=bp_model,
            fragments=fragments,
            b2p_policies=b2p_policies,
            project_root=root,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            azure_endpoint=azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            azure_api_version=azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION"),
            azure_deployment=azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        )

    # I4 / I5 — async
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "You are already in an asyncio loop (Jupyter): "
            "use `await arun_async_incremental(...)` instead of `run_incremental_profile` for I4/I5."
        )
    return asyncio.run(
        _run_async_profile(
            p,
            scenario_id=scenario_id,
            bp_model=bp_model,
            fragments=fragments,
            b2p_policies=b2p_policies,
            project_root=root,
            human_timeout_s=human_timeout_s,
            i5_use_auto_agree=i5_use_auto_agree,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            azure_deployment=azure_deployment,
        )
    )


async def arun_async_incremental(
    profile: str,
    *,
    scenario_id: str,
    bp_model: dict[str, Any],
    fragments: list[dict[str, Any]],
    b2p_policies: list[dict[str, Any]],
    project_root: Path | None = None,
    human_timeout_s: float = 120.0,
    i5_use_auto_agree: bool = False,
    api_key: str | None = None,
    azure_endpoint: str | None = None,
    azure_api_version: str | None = None,
    azure_deployment: str | None = None,
) -> IncrementalRunResult:
    """
    Same semantics as ``run_incremental_profile`` for **I4** and **I5**, as a coroutine (Jupyter / ``await``).

    - **I4**: default acceptance of unmapped (no HITL).
    - **I5**: real HITL if ``i5_use_auto_agree=False``; ``True`` = batch like I4.
    """
    p = (profile or "I0").strip().upper()
    if p == "I6":
        raise ValueError(
            "I6 removed: use I5 with i5_use_auto_agree=True for batch evaluation with simulated HITL."
        )
    if p not in ("I4", "I5"):
        raise ValueError(
            "arun_async_incremental applies only to I4 and I5 — use run_incremental_profile for I0–I3."
        )
    root = (project_root or _PROJECT_ROOT).resolve()
    return await _run_async_profile(
        p,
        scenario_id=scenario_id,
        bp_model=bp_model,
        fragments=fragments,
        b2p_policies=b2p_policies,
        project_root=root,
        human_timeout_s=human_timeout_s,
        i5_use_auto_agree=i5_use_auto_agree,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        azure_deployment=azure_deployment,
    )
