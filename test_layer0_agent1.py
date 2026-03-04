"""
test_layer0_agent1.py — Pipeline complet : Agent 1 + Agent 2 + Agent 3 + Agent 4

Usage :
    python test_layer0_agent1.py                         # Agent 1 seul
    OPENAI_API_KEY=sk-... python test_layer0_agent1.py   # Agent 1 + 2 + 3 + 4
"""

import os
import sys
import json

_ROOT = os.path.abspath(os.path.dirname(__file__))
_SRC  = os.path.join(_ROOT, "src")
sys.path.append(_SRC)


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_ROOT, ".env"), override=True)
        load_dotenv(os.path.join(_ROOT, "src", ".env"), override=True)
        return
    except ImportError:
        pass
    for path in [os.path.join(_ROOT, ".env"), os.path.join(_ROOT, "src", ".env")]:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, _, v = line.partition("=")
                            os.environ[k.strip()] = v.strip()
            except Exception:
                pass

_load_env()

from agents.structural_analyzer import StructuralAnalyzer


# ─────────────────────────────────────────────
#  Données — Credit Application BP (section 2.3)
# ─────────────────────────────────────────────

BP_MODEL = {
    "activities": [
        {"name": "check-for-completeness",          "start": True, "role": "staff"},
        {"name": "ask-for-more-information",                        "role": "staff"},
        {"name": "check-credit-amount",                             "role": "staff"},
        {"name": "perform-checks-for-large-amount",                 "role": "central-bank"},
        {"name": "perform-checks-for-small-amount",                 "role": "staff"},
        {"name": "make-decision",                                   "role": "manager"},
        {"name": "notify-rejection-by-email",                       "role": "staff"},
        {"name": "notify-acceptance-by-email",                      "role": "staff"},
        {"name": "request-credit-card",             "end": True,   "role": "staff"},
    ],
    "gateways": [
        {"name": "application-completeness", "type": "XOR"},
        {"name": "credit-amount",            "type": "XOR"},
        {"name": "decision-outcome",         "type": "XOR"},
    ],
    "flows": [
        {"from": "check-for-completeness",    "to": "application-completeness",        "type": "sequence"},
        {"from": "application-completeness",  "to": "ask-for-more-information",        "type": "sequence",
         "gateway": "application-completeness", "condition": "incomplete"},
        {"from": "application-completeness",  "to": "check-credit-amount",             "type": "sequence",
         "gateway": "application-completeness", "condition": "complete"},
        {"from": "ask-for-more-information",  "to": "check-for-completeness",          "type": "sequence"},
        {"from": "check-credit-amount",       "to": "credit-amount",                   "type": "sequence"},
        {"from": "credit-amount",             "to": "perform-checks-for-large-amount", "type": "sequence",
         "gateway": "credit-amount", "condition": ">500"},
        {"from": "credit-amount",             "to": "perform-checks-for-small-amount", "type": "sequence",
         "gateway": "credit-amount", "condition": "<=500"},
        {"from": "perform-checks-for-large-amount", "to": "make-decision",             "type": "sequence"},
        {"from": "perform-checks-for-small-amount", "to": "make-decision",             "type": "sequence"},
        {"from": "make-decision",             "to": "decision-outcome",                "type": "sequence"},
        {"from": "decision-outcome",          "to": "notify-rejection-by-email",       "type": "sequence",
         "gateway": "decision-outcome", "condition": "rejected"},
        {"from": "decision-outcome",          "to": "notify-acceptance-by-email",      "type": "sequence",
         "gateway": "decision-outcome", "condition": "accepted"},
        {"from": "notify-acceptance-by-email","to": "request-credit-card",             "type": "sequence"},
    ],
}

B2P_POLICIES = [
    {
        "@context": "http://www.w3.org/ns/odrl.jsonld",
        "uid": "http://example.com/policy:101",
        "@type": "Agreement",
        "obligation": [{
            "uid": "http://example.com/rules/largeAmountRule",
            "target": "http://example.com/asset/perform-checks-for-large-amount",
            "assigner": "http://example.com/CentralBankDBOwner:org:CBDB",
            "assignee": "http://example.com/staff/FinancialBank:org:FB",
            "action": [{"rdf:value": {"@id": "odrl:trigger"},
                        "refinement": [{"leftOperand": "Spatial", "operator": "eq", "rightOperand": "X"}]}],
        }],
    },
    {
        "@context": "http://www.w3.org/ns/odrl.jsonld",
        "uid": "http://example.com/policy:102",
        "@type": "Agreement",
        "permission": [{
            "uid": "http://example.com/rules/emailNotifyingRule",
            "target": "http://example.com/asset/notify-acceptance-by-email",
            "assigner": "http://example.com/FinancialBank:org:FB",
            "assignee": "http://example.com/customer/creditApplicant",
            "action": [{"rdf:value": {"@id": "odrl:trigger"},
                        "refinement": [{"leftOperand": "systemDevice", "operator": "eq",
                                        "rightOperand": "https://www.wikidata.org/wiki/Q80911"}]}],
            "constraint": [{"leftOperand": "event", "operator": "eq", "rightOperand": "credit acceptance"}],
        }],
    },
]

FRAGMENTS = [
    {
        "id": "f1",
        "activities": ["check-for-completeness", "ask-for-more-information"],
        "gateways": [{"name": "application-completeness", "type": "XOR"}],
    },
    {
        "id": "f2",
        "activities": ["check-credit-amount", "perform-checks-for-large-amount",
                       "perform-checks-for-small-amount"],
        "gateways": [{"name": "credit-amount", "type": "XOR"}],
    },
    {
        "id": "f3",
        "activities": ["make-decision", "notify-rejection-by-email",
                       "notify-acceptance-by-email", "request-credit-card"],
        "gateways": [{"name": "decision-outcome", "type": "XOR"}],
    },
]


# ─────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────

def main():

    # ══════════════════════════════════════════
    #  AGENT 1 — Structural Analyzer
    # ══════════════════════════════════════════
    print("=" * 60)
    print("  COUCHE 0 + AGENT 1 — Structural Analyzer")
    print("  Case Study : Credit Application BP")
    print("=" * 60)

    analyzer = StructuralAnalyzer(bp_model=BP_MODEL, fragments=FRAGMENTS, b2p_policies=B2P_POLICIES)
    result   = analyzer.analyze()

    print("\nGRAPHE FORMEL")
    print("-" * 40)
    print(result.graph)
    for k, v in result.graph.summary().items():
        print(f"  {k:20s} : {v}")

    print("\nPATTERNS STRUCTURELS")
    print("-" * 40)
    for p in result.patterns:
        print(f"  [{p.pattern_type.upper():15s}] {p.gateway_name}")
        print(f"    -> {p.description}")

    print("\nB2P MAPPINGS (activités avec policies)")
    print("-" * 40)
    for act_id, mapping in result.b2p_mappings.items():
        if mapping.b2p_policy_ids:
            print(f"  [OK] {mapping.activity_name} [{mapping.fragment_id}]")
            print(f"     Policies : {mapping.b2p_policy_ids}")
            print(f"     Règles   : {mapping.rule_types}")
        else:
            print(f"  [  ] {mapping.activity_name} [{mapping.fragment_id}] — pas de B2P policy")

    print("\nCONNEXIONS ANALYSÉES")
    print("-" * 40)
    for conn in result.connections:
        inter = " [INTER-FRAGMENT]" if conn.is_inter else ""
        cond  = f" (cond: {conn.condition})" if conn.condition else ""
        gw    = f" via {conn.gateway_name}" if conn.gateway_name else ""
        print(f"  {conn.from_activity} -> {conn.to_activity}")
        print(f"    type={conn.connection_type.upper()}{gw}{cond}{inter}")

    print("\nCONTEXTE LOCAL CL(f2)")
    print("-" * 40)
    cl_f2 = result.fragment_contexts.get("f2")
    if cl_f2:
        print(f"  Activités   : {cl_f2.activities}")
        print(f"  Gateways    : {cl_f2.gateways}")
        print(f"  Upstream    : {[(u.from_activity, u.from_fragment) for u in cl_f2.upstream_deps]}")
        print(f"  Downstream  : {[(d.to_activity, d.to_fragment) for d in cl_f2.downstream_deps]}")
        print(f"  Vue globale : {cl_f2.is_global}")

    print("\nCONTEXTE GLOBAL CG(f2)")
    print("-" * 40)
    cg_f2 = result.global_contexts.get("f2")
    if cg_f2:
        print(f"  Vue globale activée : {cg_f2.is_global}")
        print(f"  Résumé global       : {cg_f2.global_graph_summary}")
        print(f"  Arêtes inter (all)  : {len(cg_f2.all_inter_edges or [])} connexions inter-fragments")

    print("\nAnalyse Agent 1 terminée. EnrichedGraph prêt pour les agents suivants.")

    # ══════════════════════════════════════════
    #  AGENT 2 — Implicit Dependency Detector
    # ══════════════════════════════════════════
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_azure  = bool(os.environ.get("AZURE_OPENAI_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"))
    agent2_results = None

    if not (has_openai or has_azure):
        print("\n[INFO] Aucune clé LLM définie — Agents 2, 3 et 4 (dépendances implicites) non exécutés.")
        print("       L'Agent 4 sera lancé sans dépendances implicites validées.")
    else:
        print("\n" + "=" * 60)
        print("  AGENT 2 — Implicit Dependency Detector")
        print("=" * 60)
        try:
            from agents.implicit_dependency_detector import ImplicitDependencyDetector

            detector       = ImplicitDependencyDetector(enriched_graph=result, context_mode="local")
            agent2_results = detector.analyze()

            print("\nDÉPENDANCES IMPLICITES (candidats par fragment)")
            print("-" * 40)
            for frag_id, analysis in agent2_results.items():
                print(f"\n  Fragment {frag_id}: {len(analysis.candidates)} candidat(s)")
                for c in analysis.candidates:
                    conf = "haute" if c.is_high_confidence else "basse"
                    print(f"    - {c} [{conf}]")
                    j = c.justification[:80] + ("..." if len(c.justification) > 80 else "")
                    print(f"      Justification: {j}")
                if not analysis.candidates:
                    print("    (aucun)")

        except Exception as e:
            print(f"[ERREUR Agent 2] {e}")
            agent2_results = None

    # ══════════════════════════════════════════
    #  AGENT 3 — Constraint Validator
    # ══════════════════════════════════════════
    report = None

    if agent2_results and (has_openai or has_azure):
        print("\n" + "=" * 60)
        print("  AGENT 3 — Constraint Validator")
        print("=" * 60)
        try:
            from agents.constraint_validator import ConstraintValidator

            validator = ConstraintValidator(enriched_graph=result, analysis_results=agent2_results)
            report    = validator.validate()

            print("\nDÉTAIL DES DÉCISIONS")
            print("-" * 40)
            icons = {"accepted": "[OK]", "rejected": "[KO]", "reformulate": "[~~]"}
            for r in report.results:
                d     = r.decision.value
                inter = " [INTER]" if r.candidate.is_inter else ""
                print(f"\n  {icons.get(d,'?')} [{d.upper():12s}] "
                      f"{r.candidate.source_activity} -> {r.candidate.target_activity}{inter}")
                print(f"     Niveau      : {r.decision_level}")
                if r.reason:
                    print(f"     Raison      : {r.reason.value}")
                print(f"     Explication : {r.explanation[:80]}...")
                if r.reformulation_hint:
                    print(f"     Hint        : {r.reformulation_hint[:80]}...")

            print("\n" + "-" * 40)
            print("  RAPPORT FINAL AGENT 3")
            print("-" * 40)
            print(f"  Acceptés   : {len(report.accepted)}")
            print(f"  Rejetés    : {len(report.rejected)} "
                  f"(dont {report.deterministic_rejections} déterministes, "
                  f"{report.llm_rejections} LLM)")
            print(f"  Reformuler : {len(report.reformulate)}")

            print("\n  Candidats validés -> Agent 4 :")
            validated = report.validated_candidates()
            if validated:
                for c in validated:
                    inter = " [INTER]" if c.is_inter else ""
                    print(f"    [{c.dep_type.value:12s}] "
                          f"{c.source_activity} -> {c.target_activity}"
                          f"{inter} | ODRL: {c.suggested_odrl_rule}")
            else:
                print("    (aucun candidat validé)")

        except Exception as e:
            print(f"[ERREUR Agent 3] {e}")
            report = None

    # ══════════════════════════════════════════
    #  AGENT 4 — Policy Projection Agent
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  AGENT 4 — Policy Projection Agent")
    print("=" * 60)
    try:
        from agents.policy_projection_agent import PolicyProjectionAgent

        projector  = PolicyProjectionAgent(
            enriched_graph=result,
            validation_report=report,   # None si pas de clé LLM — Agent 4 tourne quand même
        )
        # Génération des policies (parallèle si supporté)
        if hasattr(projector, "generate_parallel"):
            fp_results = projector.generate_parallel()
        else:
            fp_results = projector.generate()

        # ══════════════════════════════════════════
        #  AGENT 5 — Consistency Auditor
        # ══════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  AGENT 5 — Consistency Auditor")
        print("=" * 60)
        try:
            from agents.consistency_auditor import ConsistencyAuditor

            auditor = ConsistencyAuditor(fp_results=fp_results)
            consistency_report = auditor.audit()

            s = consistency_report.summary()
            print("\nRÉSUMÉ COHÉRENCE")
            print("-" * 40)
            print(f"  Cohérent     : {s['is_consistent']}")
            print(f"  Total issues : {s['total_issues']}")
            print(f"    - Critical : {s['critical']}")
            print(f"    - Warning  : {s['warning']}")
            print(f"    - Info     : {s['info']}")
            print(f"  Intra issues : {s['intra_issues']}")
            print(f"  Inter issues : {s['inter_issues']}")

            if consistency_report.all_issues:
                print("\nDÉTAIL (premières issues)")
                print("-" * 40)
                for issue in consistency_report.all_issues[:10]:
                    print(f"  {issue}")
                if len(consistency_report.all_issues) > 10:
                    print(f"  ... ({len(consistency_report.all_issues) - 10} autres)")
            else:
                print("\nAucune issue détectée.")

        except Exception as e:
            print(f"[ERREUR Agent 5] {e}")

        # ── Synthèse des policies générées ──
        print("\nFRAGMENT POLICIES GÉNÉRÉES")
        print("-" * 40)
        total_fpa = total_fpd = 0
        for frag_id, fps in fp_results.items():
            s = fps.summary()
            print(f"\n  Fragment '{frag_id}' : "
                  f"{s['fpa_count']} FPa + {s['fpd_count']} FPd = {s['total']} policies")
            for p in fps.fpa_policies:
                src = "(B2P)   " if p.get("_source_b2p") else "(minimal)"
                print(f"    [FPa] {src} {p.get('_activity')}")
            for p in fps.fpd_policies:
                gw    = p.get("_gateway", p.get("_flow", p.get("_dep_type", "?")))
                acts  = p.get("_activities", [])
                conds = p.get("_conditions", [])
                impl  = " [IMPLICIT]" if p.get("_implicit") else ""
                cond_str = f" [{conds[0]} | {conds[1]}]" if len(conds) == 2 else ""
                a1 = acts[0] if acts else "?"
                a2 = acts[1] if len(acts) > 1 else "?"
                print(f"    [FPd] {gw:10s} {a1} -> {a2}{cond_str}{impl}")
            total_fpa += s["fpa_count"]
            total_fpd += s["fpd_count"]

        print(f"\n  TOTAL : {total_fpa} FPa + {total_fpd} FPd "
              f"= {total_fpa + total_fpd} policies générées")

        # ── Exemples ODRL JSON-LD ──
        print("\n" + "-" * 40)
        print("  EXEMPLES ODRL JSON-LD GÉNÉRÉS")
        print("-" * 40)

        examples = [
            ("FPa avec B2P (projection directe)",
             lambda p: p.get("_type") == "FPa" and p.get("_source_b2p")),
            ("FPa minimal",
             lambda p: p.get("_type") == "FPa" and not p.get("_source_b2p")),
            ("FPd XOR (Listing 7/8)",
             lambda p: p.get("_gateway") == "XOR"),
            ("FPd sequence (Listing 9)",
             lambda p: p.get("_flow") == "sequence"),
            ("FPd message inter-fragment (Listing 10)",
             lambda p: p.get("_flow") == "message"),
            ("FPd implicite (Agent 3)",
             lambda p: p.get("_implicit")),
        ]

        for label, predicate in examples:
            found = None
            for fps in fp_results.values():
                found = next((p for p in fps.all_policies() if predicate(p)), None)
                if found:
                    break
            if found:
                clean = {k: v for k, v in found.items() if not k.startswith("_")}
                print(f"\n── {label} ──")
                print(json.dumps(clean, indent=2))
            else:
                print(f"\n── {label} ── (non généré ce run)")

        # ── Export fichiers .jsonld ──
        print("\n" + "-" * 40)
        print("  EXPORT JSON-LD")
        print("-" * 40)
        output_dir = os.path.join(_ROOT, "odrl_policies")
        exported   = projector.export(fp_results, output_dir=output_dir)
        total_files = sum(len(v) for v in exported.values())
        print(f"  {total_files} fichiers .jsonld exportés → {output_dir}")
        for frag_id, files in exported.items():
            print(f"    {frag_id}/ : {len(files)} fichiers")

    except Exception as e:
        print(f"[ERREUR Agent 4] {e}")
        import traceback; traceback.print_exc()

    print("\n" + "=" * 60)
    return result


if __name__ == "__main__":
    main()