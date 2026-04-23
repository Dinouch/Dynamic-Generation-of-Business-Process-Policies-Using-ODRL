import json
import os
from typing import Any, Dict, List, Optional

from agents.structural_analyzer import StructuralAnalyzer


def run_looped_orchestration(
    *,
    bp_model: Dict[str, Any],
    fragments: List[Dict[str, Any]],
    b2p_policies: List[Dict[str, Any]],
    context_mode: str = "local",
) -> None:
    """Orchestration bout-en-bout : bus asyncio + enveloppes ACL (``run_pipeline_async``)."""
    _ = context_mode

    print("\n" + "═" * 70)
    print("  PIPELINE ASYNCHRONE — run_pipeline_async()")
    print("═" * 70)

    import asyncio

    from .async_pipeline import run_pipeline_async

    api_key = os.environ.get("OPENAI_API_KEY")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    azure_deploy = os.environ.get("AZURE_OPENAI_DEPLOYMENT")

    try:
        human_timeout_s = float(os.environ.get("HITL_TIMEOUT_S", "120") or 120)
        result = asyncio.run(
            run_pipeline_async(
                bp_model=bp_model,
                fragments=fragments,
                b2p_policies=b2p_policies,
                human_timeout_s=human_timeout_s,
                api_key=api_key,
                azure_endpoint=azure_endpoint or None,
                azure_api_version=azure_version or None,
                azure_deployment=azure_deploy or None,
            )
        )
        print(f"\n  Final status   : {result.status}")
        print(f"  Final signal   : {result.msg_type}")
        print(f"  Valid          : {result.is_valid}")
        print(f"  Syntax score   : {result.syntax_score:.2f}")
        if result.manifest:
            print(f"  Merge manifest : {json.dumps(result.manifest, indent=2)}")

        summary = result.summary or {}
        print(f"\n  Total issues   : {summary.get('total_issues', '?')}")
        print(f"    Critical     : {summary.get('critical', '?')}")
        print(f"    Warning      : {summary.get('warning', '?')}")
        print(f"    Info         : {summary.get('info', '?')}")

        by_layer = summary.get("by_layer", {})
        print("  Issues by layer:")
        print(f"    ODRL syntax       : {by_layer.get('syntax', '?')}")
        print(f"    FPa semantics     : {by_layer.get('fpa_semantic', '?')}")
        print(f"    FPd semantics     : {by_layer.get('fpd_semantic', '?')}")
        print(f"    Global coherence  : {by_layer.get('global', '?')}")

    except Exception as e:
        print(f"[ERROR run_pipeline_async] {e}")
        import traceback

        traceback.print_exc()


def run_sequential_agents(
    *,
    bp_model: Dict[str, Any],
    fragments: List[Dict[str, Any]],
    b2p_policies: List[Dict[str, Any]],
    context_mode: str = "local",
) -> Optional[Any]:
    """Runs the agents sequentially (demo / debug mode)."""

    print("\n" + "=" * 60)
    print("  AGENT 1 — Structural Analyzer")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    azure_deploy = os.environ.get("AZURE_OPENAI_DEPLOYMENT")

    analyzer = StructuralAnalyzer(
        bp_model=bp_model,
        fragments=fragments,
        b2p_policies=b2p_policies,
        api_key=api_key,
        azure_endpoint=azure_endpoint or None,
        azure_api_version=azure_version or None,
        azure_deployment=azure_deploy or None,
    )
    enriched_graph = analyzer.analyze()

    rep1 = getattr(enriched_graph, "structural_llm_report", None)
    if rep1 and rep1.get("llm_used"):
        print("\nAGENT 1 — Synthèse LLM (ambiguïtés B2P / graphe)")
        print("-" * 40)
        for n in rep1.get("notes_globales_fr") or []:
            if isinstance(n, str) and n.strip():
                print(f"  • {n.strip()}")
        g = rep1.get("graphe") or {}
        if g.get("conseil_fr"):
            print(f"  Conseil (graphe) : {g['conseil_fr']}")
    elif rep1 and not rep1.get("llm_used"):
        print(f"\n[Agent 1] LLM structurel : {rep1.get('reason', 'non utilisé')}")

    print("\nFORMAL GRAPH")
    print("-" * 40)
    print(enriched_graph.graph)
    for k, v in enriched_graph.graph.summary().items():
        print(f"  {k:20s} : {v}")

    print("\nSTRUCTURAL PATTERNS")
    print("-" * 40)
    for p in enriched_graph.patterns:
        print(f"  [{p.pattern_type.upper():15s}] {p.gateway_name}")
        print(f"    -> {p.description}")

    print("\nB2P MAPPINGS — COVERAGE")
    print("-" * 40)
    covered = 0
    for act_id, mapping in enriched_graph.b2p_mappings.items():
        if mapping.b2p_policy_ids:
            covered += 1
            print(f"  [OK] {mapping.activity_name} [{mapping.fragment_id}]")
            print(f"       Policies : {mapping.b2p_policy_ids}")
            print(f"       Rules    : {mapping.rule_types}")
        else:
            print(f"  [!!] {mapping.activity_name} [{mapping.fragment_id}] — NO B2P")
    print(f"\n  Coverage : {covered}/{len(enriched_graph.b2p_mappings)} activities")

    print("\nCONNECTIONS")
    print("-" * 40)
    for conn in enriched_graph.connections:
        inter = " [INTER]" if conn.is_inter else ""
        cond = f" (cond: {conn.condition})" if conn.condition else ""
        print(f"  {conn.from_activity} → {conn.to_activity}  [{conn.connection_type.upper()}]{cond}{inter}")

    # Exception handling agent & Agent 3 require an LLM
    has_llm = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY"))
    validation_report = None

    if not has_llm:
        print("\n[INFO] No LLM key detected — skipping Agents 2 and 3.")
    else:
        print("\n" + "=" * 60)
        print("  AGENT 2 & 3 — Unsupported formulation / validation (standalone stub)")
        print("=" * 60)
        try:
            from agents.Agent_3.constraint_validator import ConstraintValidator

            validator = ConstraintValidator(enriched_graph=enriched_graph)
            validation_report = validator.validate(enriched_graph, proposals=[])
            print(
                f"\n  ValidationReport: accepted_unsupported="
                f"{len(validation_report.accepted_unsupported_proposals)} "
                f"(use full pipeline for exception handling agent LLM formulation)"
            )
        except Exception as e:
            print(f"[ERROR Agent 3] {e}")
            import traceback

            traceback.print_exc()
            validation_report = None

    print("\n" + "=" * 60)
    print("  AGENT 4 — Policy Projection Agent")
    print("=" * 60)

    fp_results = None
    try:
        from agents.policy_projection_agent import PolicyProjectionAgent

        projector = PolicyProjectionAgent(
            enriched_graph=enriched_graph,
            validation_report=validation_report,
            api_key=api_key,
            azure_endpoint=azure_endpoint or None,
            azure_api_version=azure_version or None,
            azure_deployment=azure_deploy or None,
        )
        fp_results = projector.project(enriched_graph, validation_report)

        total_fpa = total_fpd = 0
        for frag_id, fps in fp_results.items():
            s = fps.summary()
            print(f"\n  Fragment '{frag_id}' : {s['fpa_count']} FPa + {s['fpd_count']} FPd = {s['total']} policies")
            total_fpa += s["fpa_count"]
            total_fpd += s["fpd_count"]

        print(f"\n  TOTAL : {total_fpa} FPa + {total_fpd} FPd = {total_fpa + total_fpd} policies")

        print("\n" + "-" * 60)
        print("  SAMPLE ODRL JSON-LD")
        print("-" * 60)

        # show one example per fragment if any
        for frag_id, fps in fp_results.items():
            any_policy = next(iter(fps.all_policies()), None)
            if any_policy:
                clean = {k: v for k, v in any_policy.items() if not k.startswith("_")}
                print(f"\n-- {frag_id} --")
                print(json.dumps(clean, indent=2))

        print("\n" + "-" * 60)
        print("  EXPORT JSON-LD")
        print("-" * 60)
        output_dir = os.path.abspath(
            os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                "..",
                "..",
                "output",
                "scenario1",
                "odrl_fragment_policies",
                "cli_run",
            )
        )
        exported = projector.export(fp_results, output_dir=output_dir)
        total_files = sum(len(v) for v in exported.values())
        print(f"  {total_files} .jsonld file(s) → {output_dir}")

    except Exception as e:
        print(f"[ERROR Agent 4] {e}")
        import traceback

        traceback.print_exc()

    if fp_results is not None:
        print("\n" + "=" * 60)
        print("  AGENT 5 — Policy Auditor")
        print("=" * 60)
        try:
            from agents.policy_auditor import PolicyAuditor

            auditor = PolicyAuditor(
                fp_results=fp_results,
                enriched_graph=enriched_graph,
                raw_b2p=b2p_policies,
                validation_report=validation_report,
            )
            audit_report = auditor.audit()
            auditor.print_report(audit_report)

            print(f"\n  Syntax score  : {audit_report.syntax_score():.2f}")
            print(f"  Syntax issues : {len(audit_report.syntax_issues)}")
            print(f"  FPa issues    : {len(audit_report.fpa_issues)}")
            print(f"  FPd issues    : {len(audit_report.fpd_issues)}")
            print(f"  Global issues : {len(audit_report.global_issues)}")

        except Exception as e:
            print(f"[ERROR Agent 5] {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("  END SEQUENTIAL RUN")
    print("=" * 60)
    return enriched_graph
