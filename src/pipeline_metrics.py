"""
pipeline_metrics.py
═══════════════════════════════════════════════════════════════════════════════
Collecte et calcul de toutes les métriques d'évaluation du pipeline
multi-agent ODRL (Agent 1 → 5, boucles de reformulation et de correction).

Chaque métrique est documentée par un commentaire justifiant son choix
(pertinence pour le rapport).

USAGE
────
    from pipeline_metrics import PipelineMetricsCollector
    from metrics_visualization import print_metrics_report, export_metrics_json

    collector = PipelineMetricsCollector(...)
    metrics = collector.compute_all()
    print_metrics_report(metrics)
    export_metrics_json(metrics, "metrics.json")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _count_odrl_rules(policy_dict: dict) -> int:
    """Nombre de règles ODRL (permission, prohibition, obligation) dans une policy."""
    count = 0
    for rule_type in ("permission", "prohibition", "obligation"):
        if rule_type in policy_dict and isinstance(policy_dict[rule_type], list):
            count += len(policy_dict[rule_type])
    return count


def _get_summary(obj: Any) -> dict:
    """Accès au summary (AuditReport ou dict)."""
    if hasattr(obj, "summary") and callable(obj.summary):
        return obj.summary()
    if isinstance(obj, dict):
        return obj
    return {}


# Seuil de reformulation (Agent 2↔3) — aligné sur ConstraintValidator.MAX_REFORMULATE.
# Convergence rate = % de runs où reformulation_loops < MAX_REFORMULATE.
MAX_REFORMULATE = 3

# ─────────────────────────────────────────────────────────────────────────────
#  COLLECTEUR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class PipelineMetricsCollector:
    """
    Calcule l'ensemble des métriques d'évaluation du pipeline à partir
    du résultat final, des fp_results, du graphe enrichi et d'un contexte
    de run optionnel (temps, tokens, compteurs de boucles).
    """

    def __init__(
        self,
        pipeline_result: Any,
        fp_results: Dict[str, Any],
        enriched_graph: Any,
        fragments: List[dict],
        b2p_policies: List[dict],
        validation_report: Optional[Any] = None,
        run_context: Optional[dict] = None,
    ):
        """
        pipeline_result : résultat de run_pipeline() (is_valid, report, summary,
                           syntax_score, msg_type, loop_turns).
        fp_results       : dict[fragment_id → FragmentPolicySet] en sortie Agent 4.
        enriched_graph   : EnrichedGraph (connections, b2p_mappings, etc.).
        fragments        : liste des fragments (id, activities, gateways).
        b2p_policies     : liste des B2P en entrée.
        validation_report: ValidationReport d'Agent 3 (optionnel, pour LLM vs déterministe).
        run_context      : dict optionnel avec :
                           - wall_clock_seconds : float
                           - time_per_agent    : {"agent1": float, ...}
                           - tokens            : {"agent2": {"prompt_tokens": int, "completion_tokens": int}, ...}
                           - reformulation_loops: int (tours Agent 2↔3)
                           - structural_loops   : int (tours Agent 3→1)
                           - message_hops      : int (total messages échangés)
        """
        self.pipeline_result = pipeline_result
        self.fp_results = fp_results or {}
        self.enriched_graph = enriched_graph
        self.fragments = fragments or []
        self.b2p_policies = b2p_policies or []
        self.validation_report = validation_report
        self.run_context = run_context or {}
        self._metrics: Dict[str, Any] = {}

    def compute_all(self) -> Dict[str, Any]:
        """
        Calcule uniquement les métriques essentielles pour l'évaluation globale du
        pipeline (version simplifiée pour l'étude expérimentale).

        Métriques retournées (toutes au niveau d'un run) :
        - critical_issue_count, total_issue_count, semantic_issue_count
        - semantic_validity_score : 1 - (semantic_issues / total_semantic_checks)
                * semantic_total_checks   : (FPa + FPd + global) checks
        - syntax_fidelity         : dict, score global + détail C1..C9
        - fpa_quality_rate        : activités avec expected_types ⊆ obtained_types
        - fpd_completeness_rate   : FPd générées / FPd attendues (connexions pertinentes)
        - policies_per_fragment   : dict, moyennes / dispersion + détail par fragment
        - wall_clock_seconds, total_tokens
        - reformulation_loops, converged (convergence rate = % runs avec converged=True)
        here convergence means reaching correct result before reaching max_loop
        - implicit_acceptance_rate : acceptées / proposées (Agent 2→3)
        """
        run_summary = self._run_summary()
        fidelity = self._fidelity_metrics()
        time_metrics = self._time_metrics()
        token_metrics = self._token_metrics()
        policies_stats = self._policies_per_fragment_metrics()
        implicit_metrics = self._implicit_acceptance_metrics()

        # Construction d'un dictionnaire centré sur les critères essentiels.
        self._metrics = {
            "syntax_fidelity": fidelity.get("syntax_fidelity_detailed"),
            "critical_issue_count": run_summary.get("critical_issue_count", 0),
            "total_issue_count": run_summary.get("total_issue_count", 0),
            "semantic_issue_count": fidelity.get("semantic_issue_count", 0),
            "semantic_validity_score": fidelity.get("semantic_validity_score", 0.0),
            "semantic_total_checks": fidelity.get("semantic_total_checks"),
            "fpa_quality_rate": fidelity.get("fpa_quality_rate", 0.0),
            "fpd_completeness_rate": fidelity.get("fpd_completeness_rate", 0.0),
            "fpd_expected": fidelity.get("fpd_expected"),
            "fpd_generated": fidelity.get("fpd_generated"),
            "policies_per_fragment": policies_stats,
            "wall_clock_seconds": time_metrics.get("wall_clock_seconds"),
            "total_tokens": token_metrics.get("total_tokens", 0),
            "reformulation_loops": run_summary.get("reformulation_loops"),
            "converged": run_summary.get("converged"),
            "implicit_acceptance_rate": implicit_metrics.get("implicit_acceptance_rate"),
            "implicit_proposed": implicit_metrics.get("implicit_proposed"),
            "implicit_accepted": implicit_metrics.get("implicit_accepted"),
        }
        return self._metrics

    # ══════════════════════════════════════════════════════════════════════════
    #  RUN SUMMARY
    #  Indicateurs globaux du run : succès, signal final, convergence.
    #  Justification : nécessaire pour filtrer les runs valides en évaluation
    #  et pour rapporter la décision binaire (valide / invalide) du pipeline.
    # ══════════════════════════════════════════════════════════════════════════

    def _run_summary(self) -> dict:
        pr = self.pipeline_result
        msg_type = getattr(pr, "msg_type", None)
        msg_value = getattr(msg_type, "value", str(msg_type)) if msg_type else "unknown"
        summary = _get_summary(getattr(pr, "report", None) or pr)
        by_layer = summary.get("by_layer", {})

        reformulation_loops = self.run_context.get("reformulation_loops")
        converged = (
            reformulation_loops is None
            or (isinstance(reformulation_loops, int) and reformulation_loops < MAX_REFORMULATE)
        )

        return {
            # Nombre d'issues critiques — proxy pour le "succès" global.
            "critical_issue_count": int(summary.get("critical", 0)),

            # Nombre total d'issues, toutes couches confondues.
            "total_issue_count": int(
                by_layer.get("syntax", 0)
                + by_layer.get("fpa_semantic", 0)
                + by_layer.get("fpd_semantic", 0)
                + by_layer.get("global", 0)
            ),

            # Signal final émis par Agent 5 (odrl_valid | odrl_syntax_error).
            "final_signal": msg_value,

            # La boucle de correction syntaxique 5↔4 a-t-elle été utilisée ?
            "syntax_correction_used": (getattr(pr, "loop_turns", 0) or 0) > 0,

            # Convergence : run terminé sans avoir atteint la limite de reformulation.
            # Convergence rate (agrégé) = % de runs avec converged=True.
            "reformulation_loops": reformulation_loops,
            "converged": converged,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  FIDELITY METRICS
    #  Fidélité syntaxique (ODRL) et sémantique (FPa/FPd vs BPMN et B2P).
    #  Justification : cœur de l'évaluation qualité — alignement avec la spec
    #  et avec les références (oracle, B2P, graphe).
    # ══════════════════════════════════════════════════════════════════════════

    def _fidelity_metrics(self) -> dict:
        pr = self.pipeline_result
        summary = _get_summary(getattr(pr, "report", None) or pr)
        by_layer = summary.get("by_layer", {})

        # Syntax fidelity : score [0–1] de conformité ODRL (Agent 5).
        # Justification : métrique standard pour évaluer la validité des policies
        # générées ; permet de comparer avec d’autres générateurs ODRL.
        syntax_score = getattr(pr, "syntax_score", 1.0)
        if not isinstance(syntax_score, (int, float)):
            syntax_score = 1.0

        # Répartition des issues par couche (syntax, fpa_semantic, fpd_semantic, global).
        # Justification : distinguer erreurs corrigeables (syntax) des erreurs
        # sémantiques pour cibler les améliorations (prompts, règles déterministes).
        by_layer_counts = {
            "syntax":       by_layer.get("syntax", 0),
            "fpa_semantic": by_layer.get("fpa_semantic", 0),
            "fpd_semantic": by_layer.get("fpd_semantic", 0),
            "global":       by_layer.get("global", 0),
        }

        # Indicateurs binaires de validité par niveau (Agent 5).
        # Justification : rapport technique et tableaux récapitulatifs.
        is_syntactically_valid = summary.get("is_syntactically_valid", True)
        is_semantically_valid  = summary.get("is_semantically_valid", True)

        # Comptage des issues sémantiques (FPa + FPd + global).
        semantic_issue_count = int(
            by_layer_counts["fpa_semantic"]
            + by_layer_counts["fpd_semantic"]
            + by_layer_counts["global"]
        )

        # Nombre total de FPa et FPd (pour total_checks et FPd completeness).
        total_fpa = sum(
            len(getattr(fps, "fpa_policies", fps.get("fpa_policies", [])))
            for fps in self.fp_results.values()
        )
        total_fpd = sum(
            len(getattr(fps, "fpd_policies", fps.get("fpd_policies", [])))
            for fps in self.fp_results.values()
        )

        # Semantic validity score interprétable : 1 - (issues / total_checks).
        # total_checks = nombre d'entités vérifiées (chaque FPa, chaque FPd, + couche globale).
        total_semantic_checks = total_fpa + total_fpd + 1
        if total_semantic_checks > 0:
            semantic_validity_score = max(
                0.0,
                1.0 - (float(semantic_issue_count) / total_semantic_checks),
            )
        else:
            semantic_validity_score = 1.0 if semantic_issue_count == 0 else 0.0

        # ── Décomposition de la fidelity syntaxique selon les 9 critères
        #    de Mustafa Daham et al. (C1..C9).
        #
        # On s'attend idéalement à ce que le rapport de validation (Agent 5 ou
        # SHACL) expose un dictionnaire "syntax_criteria" contenant des booléens
        # par critère. Exemple attendu :
        #
        #   summary["syntax_criteria"] = {
        #       "C1_uid": True,
        #       "C2_datatype": False,
        #       ...
        #   }
        #
        # Si ces informations ne sont pas disponibles, on se replie sur le
        # score global existant et/ou sur is_syntactically_valid.
        raw_criteria = summary.get("syntax_criteria", {})
        syntax_criteria = {
            "C1_uid": raw_criteria.get("C1_uid"),
            "C2_datatype": raw_criteria.get("C2_datatype"),
            "C3_meta_info": raw_criteria.get("C3_meta_info"),
            "C4_function_party": raw_criteria.get("C4_function_party"),
            "C5_relation_target": raw_criteria.get("C5_relation_target"),
            "C6_action": raw_criteria.get("C6_action"),
            "C7_rule": raw_criteria.get("C7_rule"),
            "C8_constraint": raw_criteria.get("C8_constraint"),
            "C9_odrl_extension": raw_criteria.get("C9_odrl_extension"),
        }

        # Calcul d'un score moyen sur les critères effectivement renseignés.
        bool_values = [v for v in syntax_criteria.values() if isinstance(v, bool)]
        if bool_values:
            criteria_score = sum(1 for v in bool_values if v) / len(bool_values)
        else:
            # Fallback : utiliser le score global existant.
            criteria_score = syntax_score

        # FPa quality : pour chaque activité, les types de règles générés couvrent-ils
        # les types attendus (B2P) ? Comparaison d'ensembles : expected ⊆ obtained.
        # Une activité peut avoir plusieurs types (ex. permission + obligation).
        all_activities = [a for f in self.fragments for a in f.get("activities", [])]
        b2p_mappings = getattr(self.enriched_graph, "b2p_mappings", {})
        expected_types: Dict[str, set] = {}
        for _aid, m in b2p_mappings.items():
            name = getattr(m, "activity_name", "")
            rtypes = getattr(m, "rule_types", [])
            if rtypes:
                expected_types[name] = set(rtypes)
        obtained_types: Dict[str, set] = {}
        for fps in self.fp_results.values():
            for p in getattr(fps, "fpa_policies", fps.get("fpa_policies", [])):
                act = p.get("_activity", "")
                obtained_types.setdefault(act, set())
                for rt in ("permission", "prohibition", "obligation"):
                    if p.get(rt):
                        obtained_types[act].add(rt)
        fpa_correct = sum(
            1 for a in all_activities
            if expected_types.get(a) and expected_types[a] <= obtained_types.get(a, set())
        )
        fpa_quality_rate = fpa_correct / len(all_activities) if all_activities else 0.0

        # FPd completeness : FPd générées / FPd attendues.
        # Seules les connexions qui doivent produire une FPd sont comptées : sequence
        # (intra), gateways XOR/AND/OR (par gateway unique), message (inter-fragment).
        connections = getattr(self.enriched_graph, "connections", [])
        gw_conns = [
            c for c in connections
            if getattr(c, "connection_type", "").upper() in ("XOR", "AND", "OR")
        ]
        seq_conns = [
            c for c in connections
            if getattr(c, "connection_type", "").lower() == "sequence"
            and not getattr(c, "is_inter", False)
        ]
        msg_conns = [c for c in connections if getattr(c, "is_inter", False)]
        expected_gw = len(set(getattr(c, "gateway_name", "") for c in gw_conns if getattr(c, "gateway_name", "")))
        expected_seq = len(seq_conns)
        expected_msg = len(msg_conns)
        expected_fpd = expected_gw + expected_seq + expected_msg
        fpd_completeness_rate = (
            min(total_fpd / expected_fpd, 1.0) if expected_fpd else 0.0
        )

        return {
            "syntax_fidelity": round(syntax_score, 4),
            "syntax_fidelity_detailed": {
                "overall_score": round(criteria_score, 4),
                "criteria": syntax_criteria,
                "legacy_score": round(syntax_score, 4),
                "is_syntactically_valid": is_syntactically_valid,
            },
            "by_layer_issues": by_layer_counts,
            "is_syntactically_valid": is_syntactically_valid,
            "is_semantically_valid": is_semantically_valid,
            "semantic_issue_count": semantic_issue_count,
            "semantic_total_checks": total_semantic_checks,
            "semantic_validity_score": round(semantic_validity_score, 4),
            "fpa_quality_rate": round(fpa_quality_rate, 4),
            "fpd_completeness_rate": round(fpd_completeness_rate, 4),
            "fpd_expected": expected_fpd,
            "fpd_generated": total_fpd,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  IMPLICIT DEPENDENCY ACCEPTANCE (Agent 2 + Agent 3)
    #  Taux d'acceptation des dépendances implicites proposées par l'Agent 2
    #  et validées par l'Agent 3. Métrique centrale pour évaluer la boucle 2↔3.
    # ══════════════════════════════════════════════════════════════════════════

    def _implicit_acceptance_metrics(self) -> dict:
        """Acceptance rate = implicit_accepted / implicit_proposed (si ValidationReport)."""
        vr = self.validation_report
        if vr is None:
            return {}
        results = getattr(vr, "results", [])
        accepted = getattr(vr, "accepted", [])
        n_proposed = len(results)
        n_accepted = len(accepted)
        if n_proposed == 0:
            rate = None  # pas de candidats proposés
        else:
            rate = round(n_accepted / n_proposed, 4)
        return {
            "implicit_proposed": n_proposed,
            "implicit_accepted": n_accepted,
            "implicit_acceptance_rate": rate,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  LOOP METRICS
    #  Nombre de tours des boucles multi-agents (reformulation, structurelle, syntaxe).
    #  Justification : coût de coopération et stabilité — moins de boucles
    #  peut indiquer un bon premier passage ou un abandon prématuré.
    # ══════════════════════════════════════════════════════════════════════════

    def _loop_metrics(self) -> dict:
        pr = self.pipeline_result
        ctx = self.run_context

        # Boucle de correction syntaxique Agent 5 ↔ Agent 4 (déjà dans PipelineResult).
        # Justification : mesure directe de l’effort de correction ODRL.
        syntax_loops = getattr(pr, "loop_turns", 0) or 0

        # Boucles de reformulation Agent 2 ↔ Agent 3 (si fourni par run_context).
        # Justification : coût de la boucle courte et stabilité des candidats.
        reformulation_loops = ctx.get("reformulation_loops")

        # Boucles structurelles Agent 3 → Agent 1 (si fourni).
        # Justification : nombre de mises à jour du graphe demandées par Agent 3.
        structural_loops = ctx.get("structural_loops")

        # Nombre total de messages échangés (si instrumenté).
        # Justification : coût en communication et complexité du run.
        message_hops = ctx.get("message_hops")

        return {
            "syntax_correction_loops": syntax_loops,
            "syntax_loops_rationale": "Tours de la boucle Agent 5 → Agent 4 pour corrections ODRL.",
            "reformulation_loops":    reformulation_loops,
            "reformulation_rationale": "Tours Agent 3 → Agent 2 (REFORMULATE) — qualité des candidats implicites.",
            "structural_loops":       structural_loops,
            "structural_rationale":   "Tours Agent 3 → Agent 1 (STRUCTURAL_UPDATE) — enrichissement du graphe.",
            "message_hops_total":     message_hops,
            "message_hops_rationale": "Nombre total de messages SEND pour coût et reproductibilité.",
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  TIME METRICS
    #  Temps total et par agent (si fournis via run_context).
    #  Justification : latence utilisateur et comparaison de performance
    #  (pipeline vs baseline, impact des boucles).
    # ══════════════════════════════════════════════════════════════════════════

    def _time_metrics(self) -> dict:
        ctx = self.run_context

        # Temps total du run (wall-clock).
        # Justification : métrique principale pour l’expérience et les benchmarks.
        wall_clock = ctx.get("wall_clock_seconds")

        # Temps par agent (agent1..agent5).
        # Justification : identifier les goulots d’étranglement (souvent Agent 2/3 LLM).
        time_per_agent = ctx.get("time_per_agent") or {}

        # Temps cumulé dans les appels LLM (si fourni).
        # Justification : séparer coût API du reste du pipeline.
        llm_time_seconds = ctx.get("llm_time_seconds")

        return {
            "wall_clock_seconds":   wall_clock,
            "wall_clock_rationale": "Temps total du pipeline (depuis analyse jusqu’au signal final).",
            "time_per_agent":       time_per_agent,
            "time_per_agent_rationale": "Répartition du temps par agent pour profilage.",
            "llm_time_seconds":    llm_time_seconds,
            "llm_time_rationale":  "Temps passé dans les appels API LLM (Agent 2, Agent 3).",
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  TOKEN METRICS
    #  Tokens consommés par agent (prompt + completion) si run_context les fournit.
    #  Justification : coût API et comparaison d’efficacité (prompt design, modèle).
    # ══════════════════════════════════════════════════════════════════════════

    def _token_metrics(self) -> dict:
        ctx = self.run_context
        tokens_by_agent = ctx.get("tokens") or {}

        total_prompt = 0
        total_completion = 0
        for agent_usage in tokens_by_agent.values():
            if isinstance(agent_usage, dict):
                total_prompt    += agent_usage.get("prompt_tokens", 0)
                total_completion += agent_usage.get("completion_tokens", 0)

        return {
            "tokens_by_agent":     tokens_by_agent,
            "tokens_rationale":   "Usage par agent (Agent 2, Agent 3) — coût et analyse par étape.",
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens":       total_prompt + total_completion,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  POLICIES PER FRAGMENT
    #  Nombre de policies (FPa, FPd, règles) par fragment.
    #  Justification : équilibrage de la charge par fragment et expansion
    #  (M5) — utile pour rapports et comparaison selon le nombre de fragments.
    # ══════════════════════════════════════════════════════════════════════════

    def _policies_per_fragment_metrics(self) -> dict:
        per_frag = {}
        total_fpa = total_fpd = total_rules = 0
        expansion_values: List[float] = []
        total_policies_values: List[int] = []

        for frag in self.fragments:
            fid = frag.get("id", "")
            fps = self.fp_results.get(fid)
            if fps is None:
                continue
            fpa_list = getattr(fps, "fpa_policies", fps.get("fpa_policies", []))
            fpd_list = getattr(fps, "fpd_policies", fps.get("fpd_policies", []))
            n_fpa = len(fpa_list)
            n_fpd = len(fpd_list)
            rules = sum(_count_odrl_rules(p) for p in fpa_list + fpd_list)
            total_fpa += n_fpa
            total_fpd += n_fpd
            total_rules += rules
            n_activities = len(frag.get("activities", []))
            # Taux d’expansion : policies générées / activités (proxy B2P par fragment).
            expansion = (n_fpa + n_fpd) / n_activities if n_activities else 0
            expansion_values.append(expansion)
            total_policies_values.append(n_fpa + n_fpd)
            per_frag[fid] = {
                "fpa_count":    n_fpa,
                "fpd_count":    n_fpd,
                "total_policies": n_fpa + n_fpd,
                "rules_count":  rules,
                "activities_count": n_activities,
                "expansion_rate": round(expansion, 2),
            }

        # Statistiques globales (moyenne / dispersion) sur le nombre de policies
        # et le taux d'expansion par fragment.
        def _mean(values: List[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        def _variance(values: List[float]) -> float:
            if len(values) <= 1:
                return 0.0
            m = _mean(values)
            return sum((v - m) ** 2 for v in values) / (len(values) - 1)

        total_policies_mean = _mean([float(v) for v in total_policies_values])
        total_policies_var = _variance([float(v) for v in total_policies_values])
        expansion_mean = _mean(expansion_values)
        expansion_var = _variance(expansion_values)

        return {
            "per_fragment": per_frag,
            "total_fpa": total_fpa,
            "total_fpd": total_fpd,
            "total_rules": total_rules,
            "stats": {
                "total_policies_mean": round(total_policies_mean, 4),
                "total_policies_variance": round(total_policies_var, 4),
                "expansion_rate_mean": round(expansion_mean, 4),
                "expansion_rate_variance": round(expansion_var, 4),
            },
        }
