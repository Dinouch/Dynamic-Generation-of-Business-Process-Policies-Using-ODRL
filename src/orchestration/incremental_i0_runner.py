"""Run profile **I0** — LLM baseline (single prompt, no deterministic Agent 4)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agents.Agent_4.i0_llm_baseline import export_i0_fp_results, generate_i0_baseline_for_scenario
from orchestration.incremental_i0_local_projection import load_fragments_enhanced_optional


def run_i0_llm_baseline(
    *,
    scenario_id: str,
    bp_model: dict[str, Any],
    fragments: list[dict[str, Any]],
    b2p_policies: list[dict[str, Any]],
    project_root: Path,
    out_dir: Path,
    api_key: str | None,
    azure_endpoint: str | None,
    azure_api_version: str | None,
    azure_deployment: str | None,
) -> tuple[dict[str, Any], float]:
    """
    I0: one LLM prompt for the entire scenario (B2P + fragments + enhanced + bp_model).
    No ``PolicyProjectionAgent.project()`` / templates.
    """
    enhanced = load_fragments_enhanced_optional(project_root, scenario_id)
    if enhanced is None:
        print(
            f"[incremental_runner][I0][WARN] fragments_enhanced.json missing for {scenario_id} "
            "— falling back to fragments.json only."
        )

    wall_t0 = time.perf_counter()
    fp_results = generate_i0_baseline_for_scenario(
        scenario_id=scenario_id,
        fragments=fragments,
        fragments_enhanced=enhanced,
        b2p_policies=b2p_policies,
        bp_model=bp_model,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        azure_deployment=azure_deployment,
    )
    elapsed = time.perf_counter() - wall_t0
    odrl_dir = out_dir / "odrl_fragment_policies"
    export_i0_fp_results(fp_results, str(odrl_dir))
    return fp_results, elapsed
