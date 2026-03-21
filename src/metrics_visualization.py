"""
metrics_visualization.py
═══════════════════════════════════════════════════════════════════════════════
Visualisation et export des métriques du pipeline ODRL.

Ce module affiche toutes les métriques (run summary, fidélité, boucles, temps,
tokens, policies par fragment, LLM vs déterministe) de façon lisible en console
et permet l'export JSON.

USAGE
────
    from pipeline_metrics import PipelineMetricsCollector
    from metrics_visualization import export_metrics_json, print_metrics_report

    collector = PipelineMetricsCollector(...)
    metrics = collector.compute_all()

    print_metrics_report(metrics)
    export_metrics_json(metrics, "rapport_metrics.json")
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union


# Constantes d'affichage
WIDTH = 72
SEP = "═" * WIDTH
SEP_THIN = "─" * WIDTH


def _fmt(v: Any) -> str:
    """Formate une valeur pour affichage (None → '—')."""
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}" if 0 < abs(v) < 1e-3 or abs(v) >= 1e4 else f"{v:.2f}"
    return str(v)


def _section_title(title: str) -> str:
    return f"\n  {title}\n  {SEP_THIN}"


def _print_run_summary(m: Dict[str, Any]) -> None:
    rs = m.get("run_summary") or {}
    print(_section_title("RUN SUMMARY"))
    print(f"  Succès pipeline       : {'Oui' if rs.get('success') else 'Non'}")
    print(f"  Signal final          : {rs.get('final_signal', '—')}")
    print(f"  Correction syntaxe    : {'Utilisée' if rs.get('syntax_correction_used') else 'Non utilisée'}")


def _print_fidelity(m: Dict[str, Any]) -> None:
    fid = m.get("fidelity") or {}
    print(_section_title("FIDÉLITÉ"))
    print(f"  Score syntaxe (ODRL)  : {_fmt(fid.get('syntax_fidelity'))}")
    print(f"  Valide syntaxe        : {fid.get('is_syntactically_valid', '—')}")
    print(f"  Valide sémantique     : {fid.get('is_semantically_valid', '—')}")
    print(f"  Taux qualité FPa      : {_fmt(fid.get('fpa_quality_rate'))}")
    print(f"  Taux complétude FPd   : {_fmt(fid.get('fpd_completeness_rate'))}")
    by_layer = fid.get("by_layer_issues") or {}
    if by_layer:
        print("  Issues par couche     :")
        for layer, count in by_layer.items():
            print(f"    • {layer}: {count}")


def _print_loops(m: Dict[str, Any]) -> None:
    loops = m.get("loops") or {}
    print(_section_title("BOUCLES"))
    print(f"  Correction syntaxe (5↔4) : {_fmt(loops.get('syntax_correction_loops'))}")
    print(f"  Reformulation (2↔3)      : {_fmt(loops.get('reformulation_loops'))}")
    print(f"  Structurelles (3→1)      : {_fmt(loops.get('structural_loops'))}")
    print(f"  Total messages (hops)    : {_fmt(loops.get('message_hops_total'))}")


def _print_time(m: Dict[str, Any]) -> None:
    time_m = m.get("time") or {}
    print(_section_title("TEMPS"))
    print(f"  Temps total (s)       : {_fmt(time_m.get('wall_clock_seconds'))}")
    print(f"  Temps LLM (s)         : {_fmt(time_m.get('llm_time_seconds'))}")
    per_agent = time_m.get("time_per_agent") or {}
    if per_agent:
        print("  Par agent (s)         :")
        for agent, sec in sorted(per_agent.items()):
            print(f"    • {agent}: {_fmt(sec)}")


def _print_tokens(m: Dict[str, Any]) -> None:
    tok = m.get("tokens") or {}
    print(_section_title("TOKENS"))
    print(f"  Total tokens          : {_fmt(tok.get('total_tokens'))}")
    print(f"  Prompt                : {_fmt(tok.get('total_prompt_tokens'))}")
    print(f"  Completion            : {_fmt(tok.get('total_completion_tokens'))}")
    by_agent = tok.get("tokens_by_agent") or {}
    if by_agent:
        print("  Par agent             :")
        for agent, usage in sorted(by_agent.items()):
            if isinstance(usage, dict):
                p = usage.get("prompt_tokens", 0)
                c = usage.get("completion_tokens", 0)
                print(f"    • {agent}: prompt={p}, completion={c}, total={p + c}")


def _print_policies_per_fragment(m: Dict[str, Any]) -> None:
    pol = m.get("policies_per_fragment") or {}
    print(_section_title("POLICIES PAR FRAGMENT"))
    print(f"  Total FPa             : {pol.get('total_fpa', '—')}")
    print(f"  Total FPd             : {pol.get('total_fpd', '—')}")
    print(f"  Total règles ODRL     : {pol.get('total_rules', '—')}")
    print(f"  Taux expansion global : {_fmt(pol.get('global_expansion_rate'))}")
    per_frag = pol.get("per_fragment") or {}
    if per_frag:
        print("  Détail par fragment   :")
        for fid, data in sorted(per_frag.items()):
            if isinstance(data, dict):
                print(f"    • {fid}: FPa={data.get('fpa_count', 0)}, FPd={data.get('fpd_count', 0)}, "
                      f"règles={data.get('rules_count', 0)}, expansion={data.get('expansion_rate', '—')}")


def _print_llm_vs_deterministic(m: Dict[str, Any]) -> None:
    llm = m.get("llm_vs_deterministic") or {}
    print(_section_title("LLM VS DÉTERMINISTE"))
    if not llm.get("available"):
        print("  (Données non disponibles — ValidationReport non fourni)")
        return
    print(f"  Acceptations LLM      : {llm.get('llm_acceptances', '—')}")
    print(f"  Rejets LLM            : {llm.get('llm_rejections', '—')}")
    print(f"  Rejets déterministes  : {llm.get('deterministic_rejections', '—')}")
    print(f"  Reformulations        : {llm.get('reformulate_count', '—')}")
    print(f"  Erreurs structurelles : {llm.get('structural_errors_count', '—')}")
    print(f"  % décisions par LLM   : {_fmt((llm.get('pct_decisions_by_llm') or 0) * 100)} %")
    print(f"  FPa issues de B2P     : {llm.get('fpa_from_b2p_count', '—')}")
    print(f"  FPa minimales         : {llm.get('fpa_minimal_count', '—')}")
    print(f"  % FPa depuis B2P      : {_fmt((llm.get('pct_fpa_from_b2p') or 0) * 100)} %")
    print(f"  Taux reformulation    : {_fmt(llm.get('reformulation_rate'))}")


def print_metrics_report(metrics: Dict[str, Any]) -> None:
    """
    Affiche en console un rapport complet de toutes les métriques,
    section par section (run summary, fidélité, boucles, temps, tokens,
    policies par fragment, LLM vs déterministe).
    """
    if not metrics:
        print("  [Aucune métrique à afficher]")
        return
    print("\n" + SEP)
    print("  MÉTRIQUES PIPELINE ODRL — RAPPORT COMPLET".center(WIDTH))
    print(SEP)
    _print_run_summary(metrics)
    _print_fidelity(metrics)
    _print_loops(metrics)
    _print_time(metrics)
    _print_tokens(metrics)
    _print_policies_per_fragment(metrics)
    _print_llm_vs_deterministic(metrics)
    print("\n" + SEP + "\n")


def export_metrics_json(metrics: Dict[str, Any], filepath: str) -> None:
    """
    Exporte les métriques au format JSON (indentation et UTF-8).
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def get_metrics_collector_metrics(collector: Any) -> Dict[str, Any]:
    """
    Récupère le dictionnaire de métriques depuis un PipelineMetricsCollector.
    Si compute_all() n'a pas encore été appelé, l'appelle.
    """
    if hasattr(collector, "_metrics") and collector._metrics:
        return collector._metrics
    if hasattr(collector, "compute_all"):
        return collector.compute_all()
    raise TypeError("Attendu un PipelineMetricsCollector ou un dict de métriques.")


def print_report_from_collector(collector: Union[Any, Dict[str, Any]]) -> None:
    """
    Affiche le rapport complet à partir d'un PipelineMetricsCollector
    ou d'un dictionnaire de métriques déjà calculées.
    """
    if isinstance(collector, dict):
        metrics = collector
    else:
        metrics = get_metrics_collector_metrics(collector)
    print_metrics_report(metrics)


def export_from_collector(
    collector: Union[Any, Dict[str, Any]],
    filepath: str,
) -> None:
    """
    Exporte les métriques en JSON à partir d'un PipelineMetricsCollector
    ou d'un dictionnaire de métriques.
    """
    if isinstance(collector, dict):
        metrics = collector
    else:
        metrics = get_metrics_collector_metrics(collector)
    export_metrics_json(metrics, filepath)
