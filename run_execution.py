#!/usr/bin/env python3
"""
CLI — ODRL-governed fragmented execution (simulation or Camunda).

Examples (from BPFragmentODRLProject root)::

    python run_execution.py --scenario src/dataset/scenario018 --vars product=complete
    python run_execution.py --scenario src/dataset/scenario018 --policies-dir output/scenario018/odrl_fragment_policies
    python run_execution.py --mode camunda7 --bpmn path/process.bpmn --process-key CreditApplication \\
        --scenario src/dataset/scenario018 --vars product=incomplete
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.dirname(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _parse_vars(pairs: list[str]) -> dict:
    out: dict = {}
    for p in pairs:
        if "=" not in p:
            continue
        k, _, v = p.partition("=")
        out[k.strip()] = v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="BP Fragment ODRL execution engine")
    ap.add_argument(
        "--scenario",
        required=True,
        help="Scenario directory (fragments_enhanced.json, bp_global.json)",
    )
    ap.add_argument(
        "--policies-dir",
        default=None,
        help="ODRL export directory (otherwise ground truth / scenario odrl_policies)",
    )
    ap.add_argument(
        "--mode",
        choices=("simulation", "camunda7", "camunda8"),
        default="simulation",
    )
    ap.add_argument("--vars", nargs="*", default=[], help="Business variables key=value")
    ap.add_argument("--bpmn", help="BPMN file for camunda mode")
    ap.add_argument("--process-key", help="Camunda process definition key")
    ap.add_argument("--max-steps", type=int, default=64)
    args = ap.parse_args()

    from execution.engine import ExecutionEngine
    from execution.models import RuntimeMode

    mode_map = {
        "simulation": RuntimeMode.SIMULATION,
        "camunda7": RuntimeMode.CAMUNDA_7,
        "camunda8": RuntimeMode.CAMUNDA_8,
    }
    engine = ExecutionEngine.from_scenario(
        args.scenario,
        policies_dir=args.policies_dir,
        mode=mode_map[args.mode],
    )
    variables = _parse_vars(args.vars)

    if args.mode == "simulation":
        result = engine.run_simulation(
            initial_variables=variables,
            max_steps=args.max_steps,
        )
    else:
        if not args.bpmn or not args.process_key:
            print("Error: --bpmn and --process-key required for Camunda", file=sys.stderr)
            return 2
        result = engine.run_camunda(
            bpmn_path=args.bpmn,
            process_definition_key=args.process_key,
            initial_variables=variables,
        )

    print(json.dumps(
        {
            "success": result.success,
            "mode": result.mode.value,
            "error": result.error,
            "summary": result.summary,
            "steps": [
                {
                    "index": s.step_index,
                    "activity": s.activity,
                    "fragment": s.fragment_id,
                    "allowed": s.decision.allowed,
                    "reason": s.decision.reason,
                    "enabled_rules": s.enabled_rules_after,
                }
                for s in result.steps
            ],
            "camunda_instance_id": result.camunda_process_instance_id,
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
