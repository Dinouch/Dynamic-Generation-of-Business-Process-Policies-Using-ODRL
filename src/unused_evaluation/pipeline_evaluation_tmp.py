"""
evaluation.py
═══════════════════════════════════════════════════════════════════════════════
Module d'évaluation du pipeline de génération de Fragment Policies (FP)
Rapport technique : « An Approach for Generating Policies of Fragmented BPs »

USAGE — intégration en fin de pipeline :

    from evaluation import PipelineEvaluator

    evaluator = PipelineEvaluator(
        bp_model          = BP_MODEL,
        fragments         = FRAGMENTS,
        b2p_policies      = B2P_POLICIES,
        enriched_graph    = result,          # sortie Agent 1
        agent2_results    = agent2_results,  # sortie Agent 2  (None si pas de LLM)
        validation_report = report,          # sortie Agent 3  (None si pas de LLM)
        fp_results        = fp_results,      # sortie Agent 4
        consistency_report= consistency_report, # sortie Agent 5
    )
    evaluator.evaluate()
    evaluator.print_report()
    evaluator.export_json("evaluation_results.json")

MÉTRIQUES CALCULÉES :
  Bloc 1 — Couverture B2P              (B2P Coverage)
  Bloc 2 — Complétude structurelle     (Structural Completeness)
  Bloc 3 — Cohérence                   (Consistency)
  Bloc 4 — Performance du pipeline     (Pipeline Performance)
  Bloc 5 — Conformité ODRL             (ODRL Compliance)
  Bloc 6 — Distribution des règles     (Rule Type Distribution)
  Bloc 7 — Score global                (Overall Score)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════
#  STRUCTURES DE DONNÉES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Metric:
    """Représente une métrique individuelle avec sa valeur et son interprétation."""
    name:        str
    value:       float          # valeur numérique brute
    label:       str            # valeur formatée pour l'affichage  (ex: "9/9 = 100%")
    description: str            # ce que mesure la métrique
    status:      str            # "ok" | "warn" | "error" | "info"
    weight:      float = 1.0    # poids pour le score global (0 = non noté)

    @property
    def icon(self) -> str:
        return {"ok": "✅", "warn": "⚠️ ", "error": "❌", "info": "ℹ️ "}.get(self.status, " ")


@dataclass
class MetricBlock:
    """Groupe de métriques appartenant au même thème."""
    title:   str
    metrics: List[Metric] = field(default_factory=list)

    def score(self) -> Optional[float]:
        """Score moyen pondéré du bloc (None si aucune métrique notable)."""
        weighted = [(m.value * m.weight) for m in self.metrics if m.weight > 0]
        weights  = [m.weight             for m in self.metrics if m.weight > 0]
        if not weights:
            return None
        return sum(weighted) / sum(weights)


@dataclass
class EvaluationReport:
    """Rapport complet d'évaluation."""
    blocks:       List[MetricBlock] = field(default_factory=list)
    overall_score: float = 0.0
    grade:         str   = ""
    summary:       str   = ""

    # Données brutes pour export JSON
    raw: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
#  ÉVALUATEUR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

class PipelineEvaluator:

    def __init__(
        self,
        bp_model:            Dict,
        fragments:           List[Dict],
        b2p_policies:        List[Dict],
        enriched_graph:      Any,                   # EnrichedGraph (Agent 1)
        fp_results:          Dict,                  # {frag_id: FragmentPolicies} (Agent 4)
        agent2_results:      Optional[Any] = None,  # {frag_id: Analysis} (Agent 2)
        validation_report:   Optional[Any] = None,  # ValidationReport (Agent 3)
        consistency_report:  Optional[Any] = None,  # ConsistencyReport (Agent 5)
    ):
        self.bp_model           = bp_model
        self.fragments          = fragments
        self.b2p_policies       = b2p_policies
        self.enriched_graph     = enriched_graph
        self.fp_results         = fp_results
        self.agent2_results     = agent2_results
        self.validation_report  = validation_report
        self.consistency_report = consistency_report
        self.report             = EvaluationReport()

    # ─────────────────────────────────────────────────────────────
    #  POINT D'ENTRÉE
    # ─────────────────────────────────────────────────────────────

    def evaluate(self) -> EvaluationReport:
        self.report.blocks = [
            self._eval_b2p_coverage(),
            self._eval_structural_completeness(),
            self._eval_consistency(),
            self._eval_pipeline_performance(),
            self._eval_odrl_compliance(),
            self._eval_rule_distribution(),
        ]
        self._compute_overall_score()
        self._build_raw()
        return self.report

    # ─────────────────────────────────────────────────────────────
    #  BLOC 1 — COUVERTURE B2P
    # ─────────────────────────────────────────────────────────────

    def _eval_b2p_coverage(self) -> MetricBlock:
        block = MetricBlock("Bloc 1 — Couverture B2P (B2P Coverage)")

        # Nombre total d'activités dans tous les fragments
        all_activities = [a for f in self.fragments for a in f["activities"]]
        total_act      = len(all_activities)

        # Activités avec au moins une B2P matchée
        b2p_mappings   = getattr(self.enriched_graph, "b2p_mappings", {})
        covered        = sum(1 for m in b2p_mappings.values() if getattr(m, "b2p_policy_ids", []))
        b2p_rate       = covered / total_act if total_act else 0

        block.metrics.append(Metric(
            name        = "B2P Activity Coverage",
            value       = b2p_rate,
            label       = f"{covered}/{total_act} = {b2p_rate*100:.1f}%",
            description = "Activités ayant au moins une B2P matchée / total activités",
            status      = "ok" if b2p_rate == 1.0 else ("warn" if b2p_rate >= 0.5 else "error"),
            weight      = 2.0,
        ))

        # Nombre de B2P chargées vs nombre d'activités
        n_b2p    = len(self.b2p_policies)
        b2p_dens = n_b2p / total_act if total_act else 0
        block.metrics.append(Metric(
            name        = "B2P Input Density",
            value       = b2p_dens,
            label       = f"{n_b2p} B2P pour {total_act} activités (ratio={b2p_dens:.2f})",
            description = "Nombre de B2P en entrée / nombre d'activités (idéal ≥ 1.0)",
            status      = "ok" if b2p_dens >= 1.0 else ("warn" if b2p_dens >= 0.5 else "error"),
            weight      = 1.0,
        ))

        # FPa générées avec source B2P vs FPa minimales
        total_fpa  = sum(len(fps.fpa_policies) for fps in self.fp_results.values())
        fpa_b2p    = sum(
            sum(1 for p in fps.fpa_policies if p.get("_source_b2p"))
            for fps in self.fp_results.values()
        )
        fpa_b2p_rate = fpa_b2p / total_fpa if total_fpa else 0
        block.metrics.append(Metric(
            name        = "FPa B2P Source Rate",
            value       = fpa_b2p_rate,
            label       = f"{fpa_b2p}/{total_fpa} FPa issues de B2P = {fpa_b2p_rate*100:.1f}%",
            description = "FPa projetées depuis une B2P réelle vs FPa minimales sans source",
            status      = "ok" if fpa_b2p_rate == 1.0 else ("warn" if fpa_b2p_rate >= 0.6 else "error"),
            weight      = 1.5,
        ))

        return block

    # ─────────────────────────────────────────────────────────────
    #  BLOC 2 — COMPLÉTUDE STRUCTURELLE
    # ─────────────────────────────────────────────────────────────

    def _eval_structural_completeness(self) -> MetricBlock:
        block = MetricBlock("Bloc 2 — Complétude Structurelle (Structural Completeness)")

        # Gateways couverts par des FPd
        patterns      = getattr(self.enriched_graph, "patterns", [])
        gw_patterns   = [p for p in patterns if "xor" in getattr(p, "pattern_type", "").lower()
                         or "and" in getattr(p, "pattern_type", "").lower()
                         or "or"  in getattr(p, "pattern_type", "").lower()]
        n_gw_patterns = len(gw_patterns)

        fpd_xor = sum(
            sum(1 for p in fps.fpd_policies if p.get("_gateway") in ("XOR", "AND", "OR"))
            for fps in self.fp_results.values()
        )
        gw_cov  = fpd_xor / n_gw_patterns if n_gw_patterns else 1.0
        block.metrics.append(Metric(
            name        = "Gateway FPd Coverage",
            value       = gw_cov,
            label       = f"{fpd_xor} FPd gateway / {n_gw_patterns} gateways = {gw_cov*100:.1f}%",
            description = "Gateways BPMN traduits en FPd / total gateways détectés",
            status      = "ok" if gw_cov == 1.0 else ("warn" if gw_cov >= 0.5 else "error"),
            weight      = 2.0,
        ))

        # Connexions inter-fragments couvertes
        connections   = getattr(self.enriched_graph, "connections", [])
        inter_conns   = [c for c in connections if getattr(c, "is_inter", False)]
        n_inter       = len(inter_conns)
        fpd_msg       = sum(
            sum(1 for p in fps.fpd_policies if p.get("_flow") == "message")
            for fps in self.fp_results.values()
        )
        inter_cov     = fpd_msg / n_inter if n_inter else 1.0
        block.metrics.append(Metric(
            name        = "Inter-Fragment Coverage",
            value       = inter_cov,
            label       = f"{fpd_msg} FPd message / {n_inter} inter-edges = {inter_cov*100:.1f}%",
            description = "Connexions inter-fragments couvertes par des FPd de type message",
            status      = "ok" if inter_cov == 1.0 else ("warn" if inter_cov >= 0.5 else "error"),
            weight      = 1.5,
        ))

        # Séquences intra-fragment couvertes
        intra_seq  = [c for c in connections
                      if not getattr(c, "is_inter", False)
                      and getattr(c, "connection_type", "").lower() == "sequence"]
        n_seq      = len(intra_seq)
        fpd_seq    = sum(
            sum(1 for p in fps.fpd_policies if p.get("_flow") == "sequence")
            for fps in self.fp_results.values()
        )
        seq_cov    = fpd_seq / n_seq if n_seq else 1.0
        block.metrics.append(Metric(
            name        = "Sequence FPd Coverage",
            value       = seq_cov,
            label       = f"{fpd_seq} FPd sequence / {n_seq} séquences intra = {seq_cov*100:.1f}%",
            description = "Séquences intra-fragment couvertes par des FPd de type sequence",
            status      = "ok" if seq_cov >= 0.8 else ("warn" if seq_cov >= 0.4 else "error"),
            weight      = 1.0,
        ))

        # Ratio moyen FPd par fragment
        n_frags  = len(self.fragments)
        total_fpd = sum(len(fps.fpd_policies) for fps in self.fp_results.values())
        avg_fpd  = total_fpd / n_frags if n_frags else 0
        block.metrics.append(Metric(
            name        = "Avg FPd per Fragment",
            value       = avg_fpd / 10,   # normalisé pour le score (max théorique ~10)
            label       = f"{avg_fpd:.1f} FPd/fragment (total={total_fpd})",
            description = "Nombre moyen de FPd générées par fragment",
            status      = "info",
            weight      = 0,   # informatif, non noté
        ))

        # Ratio moyen FPa par fragment
        total_fpa = sum(len(fps.fpa_policies) for fps in self.fp_results.values())
        avg_fpa   = total_fpa / n_frags if n_frags else 0
        block.metrics.append(Metric(
            name        = "Avg FPa per Fragment",
            value       = avg_fpa / 10,
            label       = f"{avg_fpa:.1f} FPa/fragment (total={total_fpa})",
            description = "Nombre moyen de FPa générées par fragment",
            status      = "info",
            weight      = 0,
        ))

        # Ratio total policies par activité
        all_act   = [a for f in self.fragments for a in f["activities"]]
        total_pol = total_fpa + total_fpd
        pol_per_act = total_pol / len(all_act) if all_act else 0
        block.metrics.append(Metric(
            name        = "Policies per Activity",
            value       = pol_per_act / 5,  # normalisé
            label       = f"{pol_per_act:.2f} policies/activité ({total_pol} total / {len(all_act)} activités)",
            description = "Densité de policies générées relativement au nombre d'activités",
            status      = "info",
            weight      = 0,
        ))

        return block

    # ─────────────────────────────────────────────────────────────
    #  BLOC 3 — COHÉRENCE
    # ─────────────────────────────────────────────────────────────

    def _eval_consistency(self) -> MetricBlock:
        block = MetricBlock("Bloc 3 — Cohérence (Consistency)")

        if self.consistency_report is None:
            block.metrics.append(Metric(
                name="Consistency Check", value=0,
                label="N/A — Agent 5 non exécuté",
                description="", status="info", weight=0,
            ))
            return block

        s = self.consistency_report.summary()

        # Cohérence globale
        is_consistent = s.get("is_consistent", False)
        block.metrics.append(Metric(
            name        = "Global Consistency",
            value       = 1.0 if is_consistent else 0.0,
            label       = "COHÉRENT ✓" if is_consistent else "INCOHÉRENT ✗",
            description = "Le pipeline produit un ensemble de policies sans conflit critique",
            status      = "ok" if is_consistent else "error",
            weight      = 3.0,
        ))

        # Taux d'issues critiques
        total_pol  = sum(fps.summary()["total"] for fps in self.fp_results.values())
        n_critical = s.get("critical", 0)
        crit_rate  = n_critical / total_pol if total_pol else 0
        block.metrics.append(Metric(
            name        = "Critical Issue Rate",
            value       = 1.0 - crit_rate,   # on note l'absence de critiques
            label       = f"{n_critical} critical / {total_pol} policies = {crit_rate*100:.1f}%",
            description = "Taux de policies sans issue critique (plus c'est bas, mieux c'est)",
            status      = "ok" if n_critical == 0 else "error",
            weight      = 2.0,
        ))

        # Taux de warnings (explicables = ok)
        n_warn    = s.get("warning", 0)
        warn_rate = n_warn / total_pol if total_pol else 0
        block.metrics.append(Metric(
            name        = "Warning Rate",
            value       = 1.0 - min(warn_rate, 1.0),
            label       = f"{n_warn} warning(s) / {total_pol} policies = {warn_rate*100:.1f}%",
            description = "Taux de warnings (ex: orphan FPd inter-fragment = attendu et normal)",
            status      = "ok" if n_warn == 0 else ("warn" if warn_rate < 0.15 else "error"),
            weight      = 1.0,
        ))

        # Cohérence intra vs inter
        n_intra = s.get("intra_issues", 0)
        n_inter = s.get("inter_issues", 0)
        block.metrics.append(Metric(
            name        = "Intra-Fragment Issues",
            value       = 1.0 if n_intra == 0 else 0.5,
            label       = f"{n_intra} issue(s) intra-fragment",
            description = "Conflits détectés à l'intérieur d'un même fragment",
            status      = "ok" if n_intra == 0 else "warn",
            weight      = 1.5,
        ))
        block.metrics.append(Metric(
            name        = "Inter-Fragment Issues",
            value       = 1.0 if n_inter == 0 else 0.5,
            label       = f"{n_inter} issue(s) inter-fragment",
            description = "Conflits détectés entre fragments distincts",
            status      = "ok" if n_inter == 0 else "warn",
            weight      = 1.5,
        ))

        return block

    # ─────────────────────────────────────────────────────────────
    #  BLOC 4 — PERFORMANCE DU PIPELINE
    # ─────────────────────────────────────────────────────────────

    def _eval_pipeline_performance(self) -> MetricBlock:
        block = MetricBlock("Bloc 4 — Performance du Pipeline (Pipeline Performance)")

        # ── Agent 2 ─────────────────────────────────────────────
        if self.agent2_results is None:
            block.metrics.append(Metric(
                name="Implicit Dep. Detection", value=0,
                label="N/A — Agent 2 non exécuté (pas de clé LLM)",
                description="", status="info", weight=0,
            ))
        else:
            all_candidates  = [c for a in self.agent2_results.values()
                                 for c in getattr(a, "candidates", [])]
            n_candidates    = len(all_candidates)
            n_high_conf     = sum(1 for c in all_candidates if getattr(c, "is_high_confidence", False))
            inter_cands     = sum(1 for c in all_candidates if getattr(c, "is_inter", False))
            avg_conf        = (sum(getattr(c, "confidence", 0) for c in all_candidates) / n_candidates
                               if n_candidates else 0)

            block.metrics.append(Metric(
                name        = "Implicit Dep. Candidates",
                value       = n_candidates / max(n_candidates, 1),
                label       = f"{n_candidates} candidats détectés ({n_high_conf} haute confiance)",
                description = "Nombre de dépendances implicites proposées par l'Agent 2",
                status      = "info",
                weight      = 0,
            ))
            block.metrics.append(Metric(
                name        = "Avg Candidate Confidence",
                value       = avg_conf,
                label       = f"{avg_conf:.3f} (min=0, max=1)",
                description = "Confiance moyenne des candidats de dépendances implicites",
                status      = "ok" if avg_conf >= 0.75 else ("warn" if avg_conf >= 0.5 else "error"),
                weight      = 1.0,
            ))
            block.metrics.append(Metric(
                name        = "High Confidence Rate",
                value       = n_high_conf / n_candidates if n_candidates else 0,
                label       = f"{n_high_conf}/{n_candidates} = {n_high_conf/n_candidates*100:.1f}%" if n_candidates else "N/A",
                description = "Proportion de candidats à haute confiance (> seuil)",
                status      = "ok" if n_high_conf / max(n_candidates, 1) >= 0.7 else "warn",
                weight      = 0.5,
            ))
            block.metrics.append(Metric(
                name        = "Inter-Fragment Candidates",
                value       = inter_cands / n_candidates if n_candidates else 0,
                label       = f"{inter_cands}/{n_candidates} candidats inter-fragments",
                description = "Part des dépendances implicites qui traversent des fragments",
                status      = "info",
                weight      = 0,
            ))

        # ── Agent 3 ─────────────────────────────────────────────
        if self.validation_report is None:
            block.metrics.append(Metric(
                name="Constraint Validation", value=0,
                label="N/A — Agent 3 non exécuté (pas de clé LLM)",
                description="", status="info", weight=0,
            ))
        else:
            n_accepted    = len(getattr(self.validation_report, "accepted",    []))
            n_rejected    = len(getattr(self.validation_report, "rejected",    []))
            n_reformulate = len(getattr(self.validation_report, "reformulate", []))
            n_total_val   = n_accepted + n_rejected + n_reformulate

            det_rej  = getattr(self.validation_report, "deterministic_rejections", 0)
            llm_rej  = getattr(self.validation_report, "llm_rejections",           0)

            # Precision : parmi les candidats, combien ont survécu à la validation
            precision = (n_accepted + n_reformulate) / n_total_val if n_total_val else 0
            block.metrics.append(Metric(
                name        = "Implicit Dep. Precision",
                value       = precision,
                label       = (f"{n_accepted} acceptés + {n_reformulate} reformulés "
                               f"/ {n_total_val} = {precision*100:.1f}%"),
                description = ("Part des dépendances implicites validées par l'Agent 3 "
                               "(conservateur = peu de faux positifs)"),
                status      = "info",   # pas de bon/mauvais absolu ici
                weight      = 0,
            ))

            # Rejection breakdown
            det_rate = det_rej / n_rejected if n_rejected else 0
            block.metrics.append(Metric(
                name        = "Deterministic Rejection Rate",
                value       = det_rate,
                label       = f"{det_rej}/{n_rejected} rejets déterministes = {det_rate*100:.1f}%",
                description = ("Proportion de rejets décidés sans LLM "
                               "(duplicate, gateway violation) — plus c'est haut, plus le pipeline "
                               "est efficace et transparent"),
                status      = "ok" if det_rate >= 0.6 else "warn",
                weight      = 1.0,
            ))
            block.metrics.append(Metric(
                name        = "LLM Rejection Count",
                value       = 1 - (llm_rej / max(n_total_val, 1)),
                label       = f"{llm_rej} rejet(s) LLM / {n_total_val} candidats",
                description = "Rejets nécessitant le LLM (coûteux) — idéalement minimal",
                status      = "ok" if llm_rej <= 2 else "warn",
                weight      = 0.5,
            ))

        return block

    # ─────────────────────────────────────────────────────────────
    #  BLOC 5 — CONFORMITÉ ODRL
    # ─────────────────────────────────────────────────────────────

    def _eval_odrl_compliance(self) -> MetricBlock:
        block = MetricBlock("Bloc 5 — Conformité ODRL (ODRL Compliance)")

        all_policies = [p for fps in self.fp_results.values()
                          for p in fps.all_policies()]
        n_total = len(all_policies)

        # Présence des champs obligatoires ODRL
        required_fields = ["@context", "uid", "@type"]
        valid_schema = sum(
            1 for p in all_policies
            if all(f in p for f in required_fields)
            and "odrl.jsonld" in str(p.get("@context", ""))
        )
        schema_rate = valid_schema / n_total if n_total else 0
        block.metrics.append(Metric(
            name        = "ODRL Schema Validity",
            value       = schema_rate,
            label       = f"{valid_schema}/{n_total} policies valides JSON-LD ODRL = {schema_rate*100:.1f}%",
            description = "Policies ayant @context ODRL + uid + @type correctement renseignés",
            status      = "ok" if schema_rate == 1.0 else ("warn" if schema_rate >= 0.8 else "error"),
            weight      = 2.0,
        ))

        # Diversité des types ODRL utilisés
        odrl_types = set(p.get("@type", "") for p in all_policies)
        n_types    = len(odrl_types)
        block.metrics.append(Metric(
            name        = "ODRL Type Diversity",
            value       = min(n_types / 3, 1.0),   # Agreement, Set, Offer = 3 types max
            label       = f"{n_types} type(s) ODRL distinct(s) : {sorted(odrl_types)}",
            description = "Variété des types de policies ODRL (Agreement, Set, Offer)",
            status      = "ok" if n_types >= 2 else "warn",
            weight      = 0.5,
        ))

        # Constructs ODRL couverts (richesse expressive)
        constructs_found = set()
        for p in all_policies:
            for key in ("permission", "prohibition", "obligation"):
                if key in p:
                    constructs_found.add(key)
            for rules in (p.get("permission") or p.get("prohibition") or p.get("obligation") or []):
                if isinstance(rules, dict):
                    if "constraint"  in rules: constructs_found.add("constraint")
                    if "refinement"  in str(rules): constructs_found.add("refinement")
                    if "duty"        in rules: constructs_found.add("duty")
                    if "consequence" in rules: constructs_found.add("consequence")

        n_constructs     = len(constructs_found)
        construct_score  = min(n_constructs / 6, 1.0)  # 6 constructs cibles
        block.metrics.append(Metric(
            name        = "ODRL Construct Coverage",
            value       = construct_score,
            label       = f"{n_constructs}/6 constructs ODRL : {sorted(constructs_found)}",
            description = ("Constructs ODRL distincts présents dans les policies générées "
                           "(permission, prohibition, obligation, constraint, refinement, duty)"),
            status      = "ok" if n_constructs >= 4 else ("warn" if n_constructs >= 2 else "error"),
            weight      = 1.5,
        ))

        # Présence d'une action dans toutes les règles
        rules_with_action = 0
        total_rules       = 0
        for p in all_policies:
            for rule_key in ("permission", "prohibition", "obligation"):
                rules = p.get(rule_key, [])
                if isinstance(rules, list):
                    for rule in rules:
                        total_rules += 1
                        if "action" in rule:
                            rules_with_action += 1
        action_rate = rules_with_action / total_rules if total_rules else 0
        block.metrics.append(Metric(
            name        = "Action Completeness",
            value       = action_rate,
            label       = f"{rules_with_action}/{total_rules} règles avec action = {action_rate*100:.1f}%",
            description = "Proportion de règles ODRL ayant un champ 'action' renseigné",
            status      = "ok" if action_rate == 1.0 else ("warn" if action_rate >= 0.8 else "error"),
            weight      = 1.0,
        ))

        # Présence du champ target dans toutes les règles
        rules_with_target = sum(
            1 for p in all_policies
            for rule_key in ("permission", "prohibition", "obligation")
            for rule in (p.get(rule_key) or [])
            if isinstance(rule, dict) and "target" in rule
        )
        target_rate = rules_with_target / total_rules if total_rules else 0
        block.metrics.append(Metric(
            name        = "Target Completeness",
            value       = target_rate,
            label       = f"{rules_with_target}/{total_rules} règles avec target = {target_rate*100:.1f}%",
            description = "Proportion de règles ODRL ayant un champ 'target' (asset) renseigné",
            status      = "ok" if target_rate == 1.0 else ("warn" if target_rate >= 0.8 else "error"),
            weight      = 1.0,
        ))

        return block

    # ─────────────────────────────────────────────────────────────
    #  BLOC 6 — DISTRIBUTION DES RÈGLES
    # ─────────────────────────────────────────────────────────────

    def _eval_rule_distribution(self) -> MetricBlock:
        block = MetricBlock("Bloc 6 — Distribution des Règles (Rule Type Distribution)")

        b2p_mappings = getattr(self.enriched_graph, "b2p_mappings", {})

        # Distribution des types de règles dans les B2P d'entrée
        b2p_rule_counts: Dict[str, int] = {}
        for m in b2p_mappings.values():
            for rt in getattr(m, "rule_types", []):
                b2p_rule_counts[rt] = b2p_rule_counts.get(rt, 0) + 1

        total_b2p_rules = sum(b2p_rule_counts.values())
        for rtype, count in sorted(b2p_rule_counts.items()):
            pct = count / total_b2p_rules * 100 if total_b2p_rules else 0
            block.metrics.append(Metric(
                name        = f"B2P Rule: {rtype}",
                value       = count / total_b2p_rules if total_b2p_rules else 0,
                label       = f"{count}/{total_b2p_rules} = {pct:.1f}%",
                description = f"Proportion de règles de type '{rtype}' dans les B2P d'entrée",
                status      = "info",
                weight      = 0,
            ))

        # Distribution dans les FPa générées
        fpa_rule_counts: Dict[str, int] = {}
        for fps in self.fp_results.values():
            for p in fps.fpa_policies:
                for rtype in ("permission", "prohibition", "obligation"):
                    if rtype in p:
                        fpa_rule_counts[rtype] = fpa_rule_counts.get(rtype, 0) + 1

        total_fpa_rules = sum(fpa_rule_counts.values())
        for rtype, count in sorted(fpa_rule_counts.items()):
            pct = count / total_fpa_rules * 100 if total_fpa_rules else 0
            block.metrics.append(Metric(
                name        = f"FPa Rule: {rtype}",
                value       = count / total_fpa_rules if total_fpa_rules else 0,
                label       = f"{count}/{total_fpa_rules} = {pct:.1f}%",
                description = f"Proportion de FPa avec règle '{rtype}' (doit refléter les B2P)",
                status      = "info",
                weight      = 0,
            ))

        # Fidélité de projection : les types B2P sont-ils conservés dans les FPa ?
        b2p_types = set(b2p_rule_counts.keys())
        fpa_types = set(fpa_rule_counts.keys())
        fidelity  = len(b2p_types & fpa_types) / len(b2p_types) if b2p_types else 1.0
        block.metrics.append(Metric(
            name        = "Rule Type Fidelity",
            value       = fidelity,
            label       = f"Types B2P préservés dans FPa : {sorted(b2p_types & fpa_types)} / {sorted(b2p_types)} = {fidelity*100:.1f}%",
            description = "Les types de règles des B2P sont fidèlement reportés dans les FPa générées",
            status      = "ok" if fidelity == 1.0 else ("warn" if fidelity >= 0.6 else "error"),
            weight      = 2.0,
        ))

        # Distribution des types de FPd
        fpd_type_counts: Dict[str, int] = {}
        for fps in self.fp_results.values():
            for p in fps.fpd_policies:
                gw   = p.get("_gateway")
                flow = p.get("_flow")
                key  = f"gateway:{gw}" if gw else (f"flow:{flow}" if flow else "other")
                fpd_type_counts[key] = fpd_type_counts.get(key, 0) + 1

        total_fpd = sum(fpd_type_counts.values())
        for dtype, count in sorted(fpd_type_counts.items()):
            pct = count / total_fpd * 100 if total_fpd else 0
            block.metrics.append(Metric(
                name        = f"FPd Type: {dtype}",
                value       = count / total_fpd if total_fpd else 0,
                label       = f"{count}/{total_fpd} = {pct:.1f}%",
                description = f"Proportion de FPd de type '{dtype}'",
                status      = "info",
                weight      = 0,
            ))

        return block

    # ─────────────────────────────────────────────────────────────
    #  SCORE GLOBAL
    # ─────────────────────────────────────────────────────────────

    def _compute_overall_score(self):
        """Calcule le score global pondéré sur 100."""
        scores  = []
        weights = []
        for block in self.report.blocks:
            s = block.score()
            if s is not None:
                # Poids des blocs pour le score global
                block_weight = {
                    "Bloc 1": 2.5,
                    "Bloc 2": 2.0,
                    "Bloc 3": 3.0,
                    "Bloc 4": 1.0,
                    "Bloc 5": 2.0,
                    "Bloc 6": 1.5,
                }.get(block.title[:6], 1.0)
                scores.append(s * block_weight)
                weights.append(block_weight)

        overall = sum(scores) / sum(weights) * 100 if weights else 0
        self.report.overall_score = round(overall, 1)

        if overall >= 90:
            self.report.grade   = "A"
            self.report.summary = "Pipeline excellent — couverture et cohérence optimales."
        elif overall >= 75:
            self.report.grade   = "B"
            self.report.summary = "Pipeline bon — quelques points d'amélioration mineurs."
        elif overall >= 60:
            self.report.grade   = "C"
            self.report.summary = "Pipeline acceptable — améliorations recommandées."
        else:
            self.report.grade   = "D"
            self.report.summary = "Pipeline insuffisant — révisions importantes nécessaires."

    # ─────────────────────────────────────────────────────────────
    #  EXPORT JSON
    # ─────────────────────────────────────────────────────────────

    def _build_raw(self):
        self.report.raw = {
            "overall_score": self.report.overall_score,
            "grade":         self.report.grade,
            "summary":       self.report.summary,
            "blocks": {
                block.title: {
                    "block_score": round((block.score() or 0) * 100, 1),
                    "metrics": [
                        {
                            "name":        m.name,
                            "value":       round(m.value, 4),
                            "label":       m.label,
                            "description": m.description,
                            "status":      m.status,
                        }
                        for m in block.metrics
                    ]
                }
                for block in self.report.blocks
            }
        }

    def export_json(self, path: str = "evaluation_results.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.report.raw, f, indent=2, ensure_ascii=False)
        print(f"\n[Évaluation] Résultats exportés → {path}")

    # ─────────────────────────────────────────────────────────────
    #  AFFICHAGE CONSOLE
    # ─────────────────────────────────────────────────────────────

    def print_report(self):
        W = 72
        print("\n" + "═" * W)
        print("  RAPPORT D'ÉVALUATION DU PIPELINE".center(W))
        print("  Credit Application BP — Fragment Policy Generation".center(W))
        print("═" * W)

        for block in self.report.blocks:
            print(f"\n┌─ {block.title} " + "─" * max(0, W - len(block.title) - 4) + "┐")
            for m in block.metrics:
                # ligne principale
                status_icon = m.icon
                name_trunc  = m.name[:32].ljust(33)
                val_str     = m.label[:34].ljust(35)
                print(f"│ {status_icon} {name_trunc} {val_str}│")
                # ligne description (grisée)
                desc = m.description
                for i in range(0, len(desc), W - 6):
                    chunk = desc[i:i + W - 6]
                    print(f"│   {'':33s} {chunk:{W-38}s} │")
            block_score = block.score()
            if block_score is not None:
                score_str = f"Score bloc : {block_score*100:.1f}%"
                print(f"│{'':>{W-len(score_str)-3}}{score_str}  │")
            print("└" + "─" * (W - 2) + "┘")

        # ── Score global ────────────────────────────────────────
        print("\n" + "═" * W)
        print(f"  SCORE GLOBAL : {self.report.overall_score:.1f} / 100   "
              f"[Grade {self.report.grade}]".center(W))
        print(f"  {self.report.summary}".center(W))
        print("═" * W)

        # ── Tableau synthétique ─────────────────────────────────
        print("\n  TABLEAU SYNTHÉTIQUE DES MÉTRIQUES CLÉS")
        print("  " + "-" * 68)
        key_metrics = [
            ("B2P Activity Coverage",     "Couverture activités ↔ B2P"),
            ("FPa B2P Source Rate",        "FPa issues de B2P réelles"),
            ("Gateway FPd Coverage",       "Gateways → FPd"),
            ("Inter-Fragment Coverage",    "Inter-edges → FPd message"),
            ("Global Consistency",         "Cohérence globale Agent 5"),
            ("Critical Issue Rate",        "Absence d'issues critiques"),
            ("ODRL Schema Validity",       "Validité schéma JSON-LD"),
            ("ODRL Construct Coverage",    "Richesse expressive ODRL"),
            ("Rule Type Fidelity",         "Fidélité types B2P → FPa"),
            ("Avg Candidate Confidence",   "Confiance moy. Agent 2"),
            ("Deterministic Rejection Rate","Rejets déterministes Agent 3"),
        ]
        all_metrics_flat = {m.name: m for b in self.report.blocks for m in b.metrics}
        for mname, mlabel in key_metrics:
            m = all_metrics_flat.get(mname)
            if m:
                bar_len = int(m.value * 20)
                bar     = "█" * bar_len + "░" * (20 - bar_len)
                print(f"  {m.icon} {mlabel:38s} {bar} {m.label}")
        print()