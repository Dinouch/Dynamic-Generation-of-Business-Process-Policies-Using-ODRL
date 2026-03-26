"""
evaluation.py
═══════════════════════════════════════════════════════════════════════════════
Évaluation du pipeline de génération de Fragment Policies
Rapport technique : « An Approach for Generating Policies of Fragmented BPs »

USAGE :
    from evaluation import PipelineEvaluator

    evaluator = PipelineEvaluator(
        bp_model           = BP_MODEL,
        fragments          = FRAGMENTS,
        b2p_policies       = B2P_POLICIES,
        enriched_graph     = result,
        fp_results         = fp_results,
        consistency_report = consistency_report,
    )
    evaluator.evaluate()
    evaluator.print_report()
    evaluator.export_json("evaluation_results.json")

5 MÉTRIQUES :
  M1 — FPa Quality          : FPa générées avec le bon type de règle B2P / total activités
  M2 — FPd Completeness     : FPd générées / connexions BPMN attendues
  M3 — ODRL Conformity      : policies valides JSON-LD ODRL / total policies
  M4 — Consistency          : cohérence intra/inter-fragments (Agent 5)
  M5 — Policy Expansion Rate: FP générées (FPa+FPd) / B2P en entrée

  --Time + inspiration from ODRL articles +generation LLM vs algo + Difference selon le nombre de fragments
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional


class PipelineEvaluator:

    def __init__(
        self,
        bp_model:            Dict,
        fragments:           List[Dict],
        b2p_policies:        List[Dict],
        enriched_graph:      Any,
        fp_results:          Dict,
        consistency_report:  Optional[Any] = None,
    ):
        self.bp_model           = bp_model
        self.fragments          = fragments
        self.b2p_policies       = b2p_policies
        self.enriched_graph     = enriched_graph
        self.fp_results         = fp_results
        self.consistency_report = consistency_report
        self.metrics: Dict[str, Dict] = {}

    # ─────────────────────────────────────────────────────────────

    def evaluate(self) -> Dict:
        self.metrics = {
            "M1_fpa_quality":      self._m1_fpa_quality(),
            "M2_fpd_completeness": self._m2_fpd_completeness(),
            "M3_odrl_conformity":  self._m3_odrl_conformity(),
            "M4_consistency":      self._m4_consistency(),
            "M5_expansion_rate":   self._m5_expansion_rate(),
        }
        return self.metrics

    # ─────────────────────────────────────────────────────────────
    #  M1 — FPa QUALITY
    #
    #  Fusionne couverture + fidélité :
    #  Pour chaque activité, une FPa a-t-elle été générée
    #  avec le bon type de règle issu de la B2P source ?
    #
    #  Numérateur   : activités dont la FPa a le même type de règle
    #                 que la B2P matchée (permission/prohibition/obligation)
    #  Dénominateur : total des activités dans tous les fragments
    # ─────────────────────────────────────────────────────────────

    def _m1_fpa_quality(self) -> Dict:
        all_activities = [a for f in self.fragments for a in f["activities"]]
        total          = len(all_activities)

        # Type de règle attendu depuis les B2P (Agent 1)
        # b2p_mappings est indexé par activity_id (ex: "act_check-for-completeness")
        # mais mapping.activity_name contient le nom réel ("check-for-completeness")
        b2p_mappings = getattr(self.enriched_graph, "b2p_mappings", {})
        expected: Dict[str, str] = {}
        for _act_id, mapping in b2p_mappings.items():
            rule_types = getattr(mapping, "rule_types", [])
            if rule_types:
                expected[mapping.activity_name] = rule_types[0]

        # Type de règle obtenu dans les FPa générées (Agent 4)
        obtained: Dict[str, str] = {}
        for fps in self.fp_results.values():
            for p in fps.fpa_policies:
                act = p.get("_activity", "")
                for rtype in ("permission", "prohibition", "obligation"):
                    if rtype in p:
                        obtained[act] = rtype
                        break

        # Activités correctement couvertes : FPa présente + bon type
        correct    = sum(1 for act in all_activities
                         if act in expected and obtained.get(act) == expected[act])
        rate       = correct / total if total else 0
        mismatches = [
            f"{act}: attendu={expected[act]}, obtenu={obtained.get(act, '—')}"
            for act in all_activities
            if act in expected and obtained.get(act) != expected[act]
        ]
        no_b2p = [act for act in all_activities if act not in expected]

        return {
            "label":       "FPa Quality",
            "description": ("Activités dont la FPa générée a le bon type de règle B2P "
                            "(permission / prohibition / obligation) / total activités"),
            "numerator":   correct,
            "denominator": total,
            "rate":        rate,
            "mismatches":  mismatches,
            "no_b2p":      no_b2p,
            "status":      "ok"   if rate == 1.0 else
                           "warn" if rate >= 0.7 else "error",
        }

    # ─────────────────────────────────────────────────────────────
    #  M2 — FPd COMPLETENESS
    #
    #  Pour chaque connexion BPMN (gateway XOR/AND/OR + flow
    #  sequence + flow message inter-fragments), une FPd
    #  a-t-elle été générée ?
    #
    #  Numérateur   : connexions BPMN couvertes par une FPd
    #  Dénominateur : total connexions BPMN dans le graphe
    # ─────────────────────────────────────────────────────────────

    def _m2_fpd_completeness(self) -> Dict:
        connections = getattr(self.enriched_graph, "connections", [])

        # Catégoriser les connexions attendues
        gw_conns    = [c for c in connections
                       if getattr(c, "connection_type", "").upper() in ("XOR", "AND", "OR")]
        seq_conns   = [c for c in connections
                       if getattr(c, "connection_type", "").lower() == "sequence"
                       and not getattr(c, "is_inter", False)]
        msg_conns   = [c for c in connections
                       if getattr(c, "is_inter", False)]

        expected_gw  = len(set(getattr(c, "gateway_name", "") for c in gw_conns
                               if getattr(c, "gateway_name", "")))
        expected_seq = len(seq_conns)
        expected_msg = len(msg_conns)
        expected_total = expected_gw + expected_seq + expected_msg

        # FPd générées par type
        fpd_gw  = sum(sum(1 for p in fps.fpd_policies
                          if p.get("_gateway") in ("XOR", "AND", "OR"))
                      for fps in self.fp_results.values())
        fpd_seq = sum(sum(1 for p in fps.fpd_policies if p.get("_flow") == "sequence")
                      for fps in self.fp_results.values())
        fpd_msg = sum(sum(1 for p in fps.fpd_policies if p.get("_flow") == "message")
                      for fps in self.fp_results.values())
        generated_total = fpd_gw + fpd_seq + fpd_msg

        rate = min(generated_total / expected_total, 1.0) if expected_total else 0

        return {
            "label":       "FPd Completeness",
            "description": ("FPd générées couvrant les connexions BPMN "
                            "(gateways + séquences + messages inter-fragments) / total connexions"),
            "generated_total":  generated_total,
            "expected_total":   expected_total,
            "detail": {
                "gateways":  {"expected": expected_gw,  "generated": fpd_gw},
                "sequences": {"expected": expected_seq, "generated": fpd_seq},
                "messages":  {"expected": expected_msg, "generated": fpd_msg},
            },
            "rate":   rate,
            "status": "ok"   if rate >= 0.9 else
                      "warn" if rate >= 0.6 else "error",
        }

    # ─────────────────────────────────────────────────────────────
    #  M3 — ODRL CONFORMITY
    #
    #  Chaque policy générée (FPa + FPd) respecte-t-elle
    #  le schéma JSON-LD ODRL ?
    #  Critères : @context ODRL + uid + @type + au moins une règle
    #
    #  Numérateur   : policies valides
    #  Dénominateur : total policies générées
    # ─────────────────────────────────────────────────────────────

    def _m3_odrl_conformity(self) -> Dict:
        all_policies = [p for fps in self.fp_results.values()
                          for p in fps.all_policies()]
        total = len(all_policies)

        def is_valid(p: Dict) -> bool:
            return (
                "odrl.jsonld" in str(p.get("@context", ""))
                and bool(p.get("uid"))
                and bool(p.get("@type"))
                and any(k in p for k in ("permission", "prohibition", "obligation"))
            )

        valid = sum(1 for p in all_policies if is_valid(p))
        rate  = valid / total if total else 0

        return {
            "label":       "ODRL Conformity",
            "description": ("Policies conformes JSON-LD ODRL "
                            "(@context + uid + @type + règle) / total policies générées"),
            "numerator":   valid,
            "denominator": total,
            "rate":        rate,
            "status":      "ok"   if rate == 1.0 else
                           "warn" if rate >= 0.8 else "error",
        }

    # ─────────────────────────────────────────────────────────────
    #  M4 — CONSISTENCY
    #
    #  Le jeu complet de policies est-il cohérent ?
    #  Résultat direct de l'Agent 5.
    #  Score : 1.0 si aucune issue, pénalité par critical (-20%)
    #  et par warning (-5%).
    # ─────────────────────────────────────────────────────────────

    def _m4_consistency(self) -> Dict:
        if self.consistency_report is None:
            return {
                "label":       "Consistency",
                "description": "Cohérence intra- et inter-fragments (Agent 5)",
                "available":   False,
                "rate":        0,
                "status":      "info",
            }

        s             = self.consistency_report.summary()
        n_critical    = s.get("critical", 0)
        n_warning     = s.get("warning",  0)
        is_consistent = s.get("is_consistent", False)
        rate          = max(0.0, 1.0 - (n_critical * 0.20) - (n_warning * 0.05))

        return {
            "label":          "Consistency",
            "description":    ("Cohérence des policies générées — "
                               "pénalité : -20% par issue critique, -5% par warning"),
            "is_consistent":  is_consistent,
            "critical":       n_critical,
            "warnings":       n_warning,
            "total_policies": sum(fps.summary()["total"] for fps in self.fp_results.values()),
            "rate":           rate,
            "available":      True,
            "status":         "ok"   if n_critical == 0 and n_warning == 0 else
                              "warn" if n_critical == 0 else "error",
        }

    # ─────────────────────────────────────────────────────────────
    #  M5 — POLICY EXPANSION RATE
    #
    #  Combien de fragment policies le pipeline génère-t-il
    #  par B2P en entrée ? Mesure l'enrichissement apporté
    #  par la fragmentation.
    #
    #  Global  : (FPa + FPd totales) / nombre de B2P
    #  Par fragment : FP du fragment / B2P des activités du fragment
    # ─────────────────────────────────────────────────────────────

    def _m5_expansion_rate(self) -> Dict:
        n_b2p       = len(self.b2p_policies)
        total_fpa   = sum(len(fps.fpa_policies) for fps in self.fp_results.values())
        total_fpd   = sum(len(fps.fpd_policies) for fps in self.fp_results.values())
        total_fp    = total_fpa + total_fpd
        global_rate = total_fp / n_b2p if n_b2p else 0

        # Expansion par fragment
        b2p_mappings = getattr(self.enriched_graph, "b2p_mappings", {})

        # Index nom → mapping pour éviter le bug activity_id vs activity_name
        name_to_mapping = {
            getattr(m, "activity_name", ""): m
            for m in b2p_mappings.values()
        }

        per_fragment = {}
        for frag in self.fragments:
            fid  = frag["id"]
            fps  = self.fp_results.get(fid)
            if fps is None:
                continue

            # B2P du fragment : activités ayant un mapping (via activity_name)
            n_frag_b2p = sum(
                1 for act in frag["activities"]
                if act in name_to_mapping
                and getattr(name_to_mapping[act], "b2p_policy_ids", [])
            )
            fp_generated = fps.summary()["total"]
            expansion    = fp_generated / n_frag_b2p if n_frag_b2p else 0

            per_fragment[fid] = {
                "b2p_count":     n_frag_b2p,
                "fp_generated":  fp_generated,
                "fpa":           fps.summary()["fpa_count"],
                "fpd":           fps.summary()["fpd_count"],
                "expansion":     round(expansion, 2),
            }

        return {
            "label":        "Policy Expansion Rate",
            "description":  ("FP générées (FPa + FPd) / B2P en entrée — "
                             "mesure l'enrichissement apporté par la fragmentation"),
            "n_b2p":        n_b2p,
            "total_fpa":    total_fpa,
            "total_fpd":    total_fpd,
            "total_fp":     total_fp,
            "global_rate":  round(global_rate, 2),
            "per_fragment": per_fragment,
            # pas de rate 0-1 ici : métrique descriptive, non notée
            "rate":         min(global_rate / 3.0, 1.0),  # normalisé pour la barre (ref=3x)
            "status":       "info",
        }

    # ─────────────────────────────────────────────────────────────
    #  AFFICHAGE CONSOLE
    # ─────────────────────────────────────────────────────────────

    def print_report(self):
        W    = 68
        ICON = {"ok": "✅", "warn": "⚠️ ", "error": "❌", "info": "ℹ️ "}

        print("\n" + "═" * W)
        print("  ÉVALUATION DU PIPELINE — Credit Application BP".center(W))
        print("═" * W)

        order = [
            "M1_fpa_quality",
            "M2_fpd_completeness",
            "M3_odrl_conformity",
            "M4_consistency",
            "M5_expansion_rate",
        ]

        scored_rates = []

        for i, key in enumerate(order, 1):
            m = self.metrics.get(key, {})
            if not m:
                continue

            rate   = m.get("rate", 0)
            status = m.get("status", "info")
            icon   = ICON.get(status, " ")
            label  = m.get("label", key)
            desc   = m.get("description", "")

            filled = int(rate * 30)
            bar    = "█" * filled + "░" * (30 - filled)
            pct    = f"{rate * 100:.1f}%"

            print(f"\n  {icon} [M{i}] {label}")
            print(f"       {desc}")
            print(f"       [{bar}] {pct}", end="")

            if key == "M1_fpa_quality":
                print(f"   → {m['numerator']}/{m['denominator']} activités")
                for mm in m.get("mismatches", []):
                    print(f"             ⚠️  Mismatch : {mm}")
                if m.get("no_b2p"):
                    print(f"             ℹ️  Sans B2P : {m['no_b2p']}")

            elif key == "M2_fpd_completeness":
                d = m["detail"]
                print(f"   → {m['generated_total']}/{m['expected_total']} FPd")
                print(f"             gateways  : {d['gateways']['generated']}/{d['gateways']['expected']}"
                      f"   séquences : {d['sequences']['generated']}/{d['sequences']['expected']}"
                      f"   messages  : {d['messages']['generated']}/{d['messages']['expected']}")

            elif key == "M3_odrl_conformity":
                print(f"   → {m['numerator']}/{m['denominator']} policies valides")

            elif key == "M4_consistency":
                if m.get("available"):
                    flag = "COHÉRENT ✓" if m["is_consistent"] else "INCOHÉRENT ✗"
                    print(f"   → {flag}  "
                          f"(critical={m['critical']}, warnings={m['warnings']}, "
                          f"total={m['total_policies']} policies)")
                else:
                    print("   → Agent 5 non exécuté")

            elif key == "M5_expansion_rate":
                print(f"   → {m['total_fp']} FP / {m['n_b2p']} B2P"
                      f"  =  ×{m['global_rate']} en moyenne")
                print(f"             (FPa={m['total_fpa']}, FPd={m['total_fpd']})")
                print()
                print(f"       {'Fragment':<12} {'B2P':>5} {'FPa':>5} "
                      f"{'FPd':>5} {'Total FP':>10} {'Expansion':>11}")
                print(f"       {'─'*12} {'─'*5} {'─'*5} {'─'*5} {'─'*10} {'─'*11}")
                for fid, fd in m["per_fragment"].items():
                    print(f"       {fid:<12} {fd['b2p_count']:>5} {fd['fpa']:>5} "
                          f"{fd['fpd']:>5} {fd['fp_generated']:>10} "
                          f"{'×'+str(fd['expansion']):>11}")

            if status != "info":
                scored_rates.append(rate)

        # ── Score global ────────────────────────────────────────
        overall = sum(scored_rates) / len(scored_rates) * 100 if scored_rates else 0
        grade   = ("A" if overall >= 90 else
                   "B" if overall >= 75 else
                   "C" if overall >= 60 else "D")

        print("\n\n" + "─" * W)
        print(f"  SCORE GLOBAL : {overall:.1f} / 100   [Grade {grade}]")
        print(f"  (calculé sur M1, M2, M3, M4 — M5 est descriptive)")
        print("─" * W + "\n")

    # ─────────────────────────────────────────────────────────────
    #  EXPORT JSON
    # ─────────────────────────────────────────────────────────────

    def export_json(self, path: str = "evaluation_results.json"):
        scored = [m.get("rate", 0)
                  for m in self.metrics.values()
                  if m.get("status") != "info"]
        overall = sum(scored) / len(scored) * 100 if scored else 0

        out = {
            "pipeline":      "Credit Application BP — Fragment Policy Generation",
            "overall_score": round(overall, 1),
            "metrics":       self.metrics,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  [Évaluation] Résultats exportés → {path}")