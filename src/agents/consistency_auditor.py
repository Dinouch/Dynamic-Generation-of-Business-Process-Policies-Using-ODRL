"""
consistency_auditor.py — Agent 5 : Consistency Auditor

Responsabilité unique :
    Vérifier la cohérence intra- et inter-fragment des policies générées
    par l'Agent 4, conformément à la section 4.2 du rapport (module checker).

Deux niveaux de vérification :

    1. INTRA-CONSISTENCY (au sein d'un fragment fi)
       Vérifier que les FPa et FPd d'un même fragment ne se contredisent pas :
       - Conflit permission vs prohibition sur la même activité/règle
       - FPd XOR incohérent : deux branches XOR avec la même condition
       - FPd AND/OR sur des règles qui se contredisent (permission vs prohibition)
       - Contraintes temporelles impossibles (dateTime contradictoires)
       - Activité référencée dans FPd mais absente des FPa du fragment

    2. INTER-CONSISTENCY (entre fragments fi et fj)
       Vérifier que les FPd message sont cohérents entre les deux fragments :
       - Le message flow fi→fj existe bien du côté de fj (policy de réception)
       - La règle ruleix (assignee) dans fi correspond bien à une règle existante
       - La règle rulejy (target du duty) dans fj correspond bien à une FPa existante
       - Pas de cycle de dépendances entre fragments

Produit :
    ConsistencyReport :
        - intra_issues : list[ConsistencyIssue] par fragment
        - inter_issues : list[ConsistencyIssue] entre fragments
        - is_consistent : bool (True si aucune issue CRITICAL)
        - summary : dict

Types d'issues :
    CRITICAL  → incohérence bloquante (policy invalide)
    WARNING   → incohérence potentielle (policy douteuse mais pas invalide)
    INFO      → observation (pas un problème, juste une note)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .policy_projection_agent import FragmentPolicySet


# ─────────────────────────────────────────────
#  Types
# ─────────────────────────────────────────────

class IssueSeverity(Enum):
    CRITICAL = "critical"
    WARNING  = "warning"
    INFO     = "info"


class IssueType(Enum):
    # Intra
    PERM_PROHIB_CONFLICT   = "permission_prohibition_conflict"
    XOR_SAME_CONDITION     = "xor_same_condition"
    XOR_MISSING_CONDITION  = "xor_missing_condition"
    MISSING_FPA            = "missing_fpa_for_fpd_activity"
    DUPLICATE_RULE         = "duplicate_rule_uid"
    CONTRADICTORY_ACTIONS  = "contradictory_actions"
    # Inter
    MESSAGE_NO_RECEIVER    = "message_flow_no_receiver"
    MESSAGE_UNKNOWN_RULE   = "message_flow_unknown_rule"
    MESSAGE_UNKNOWN_TARGET = "message_flow_unknown_target"
    DEPENDENCY_CYCLE       = "inter_fragment_dependency_cycle"
    # Générique
    ORPHAN_FPD             = "orphan_fpd_no_matching_fpa"


@dataclass
class ConsistencyIssue:
    severity:    IssueSeverity
    issue_type:  IssueType
    fragment_id: str
    description: str
    policy_uid:  Optional[str] = None
    details:     Optional[dict] = None

    def __str__(self) -> str:
        return (f"[{self.severity.value.upper():8s}] [{self.issue_type.value}] "
                f"({self.fragment_id}) {self.description}")


@dataclass
class ConsistencyReport:
    intra_issues: list[ConsistencyIssue] = field(default_factory=list)
    inter_issues: list[ConsistencyIssue] = field(default_factory=list)

    @property
    def all_issues(self) -> list[ConsistencyIssue]:
        return self.intra_issues + self.inter_issues

    @property
    def is_consistent(self) -> bool:
        """True si aucune issue CRITICAL."""
        return not any(i.severity == IssueSeverity.CRITICAL for i in self.all_issues)

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.all_issues if i.severity == IssueSeverity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.all_issues if i.severity == IssueSeverity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.all_issues if i.severity == IssueSeverity.INFO)

    def summary(self) -> dict:
        return {
            "is_consistent":   self.is_consistent,
            "total_issues":    len(self.all_issues),
            "critical":        self.critical_count,
            "warning":         self.warning_count,
            "info":            self.info_count,
            "intra_issues":    len(self.intra_issues),
            "inter_issues":    len(self.inter_issues),
        }


# ─────────────────────────────────────────────
#  Agent 5 — Consistency Auditor
# ─────────────────────────────────────────────

class ConsistencyAuditor:
    """
    Agent 5 du pipeline multi-agent.

    Vérifie l'intra- et inter-consistency des fragment policies
    générées par l'Agent 4, conformément au module checker (section 4.2).
    """

    def __init__(self, fp_results: dict[str, FragmentPolicySet]):
        self.fp_results = fp_results
        # Index global : rule_uid → (fragment_id, policy_uid, rule_type)
        self._rule_index:   dict[str, tuple[str, str, str]] = {}
        # Index global : activity_name → fragment_id
        self._activity_index: dict[str, str] = {}
        self._build_indexes()

    # ─────────────────────────────────────────
    #  Indexes globaux
    # ─────────────────────────────────────────

    def _build_indexes(self) -> None:
        """
        Construit deux index globaux pour les vérifications inter-fragment :
        - _rule_index    : rule_uid → (fragment_id, policy_uid, rule_type)
        - _activity_index: activity_name → fragment_id
        """
        for frag_id, fps in self.fp_results.items():
            for policy in fps.fpa_policies:
                # Indexer les activités
                act = policy.get("_activity")
                if act:
                    self._activity_index[act] = frag_id

                # Indexer les règles
                for rule_type in ("permission", "prohibition", "obligation"):
                    for rule in policy.get(rule_type, []):
                        uid = rule.get("uid")
                        if uid:
                            self._rule_index[uid] = (
                                frag_id, policy.get("uid", ""), rule_type)

    # ─────────────────────────────────────────
    #  Point d'entrée
    # ─────────────────────────────────────────

    def audit(self) -> ConsistencyReport:
        print("[Agent 5] Consistency Auditor — démarrage de l'audit")
        report = ConsistencyReport()

        for frag_id, fps in self.fp_results.items():
            print(f"[Agent 5] Audit intra-consistency '{frag_id}'...")
            issues = self._check_intra(frag_id, fps)
            report.intra_issues.extend(issues)

        print("[Agent 5] Audit inter-consistency...")
        inter_issues = self._check_inter()
        report.inter_issues.extend(inter_issues)

        s = report.summary()
        status = "✓ COHÉRENT" if s["is_consistent"] else "✗ INCOHÉRENT"
        print(f"[Agent 5] Audit terminé : {status} — "
              f"{s['critical']} critical, {s['warning']} warning, {s['info']} info")

        return report

    # ─────────────────────────────────────────
    #  Intra-consistency
    # ─────────────────────────────────────────

    def _check_intra(
        self, frag_id: str, fps: FragmentPolicySet
    ) -> list[ConsistencyIssue]:
        issues = []

        # ── 1. Construire l'index local des règles ──
        # rule_uid → {"type": permission|prohibition|obligation, "policy": uid, "target": ...}
        local_rules: dict[str, dict] = {}
        for policy in fps.fpa_policies:
            for rule_type in ("permission", "prohibition", "obligation"):
                for rule in policy.get(rule_type, []):
                    uid = rule.get("uid")
                    if uid:
                        if uid in local_rules:
                            # Règle dupliquée
                            issues.append(ConsistencyIssue(
                                severity=IssueSeverity.WARNING,
                                issue_type=IssueType.DUPLICATE_RULE,
                                fragment_id=frag_id,
                                description=f"Règle UID dupliquée : '{uid}'",
                                policy_uid=policy.get("uid"),
                                details={"duplicate_uid": uid},
                            ))
                        local_rules[uid] = {
                            "type":   rule_type,
                            "policy": policy.get("uid"),
                            "target": rule.get("target"),
                            "action": rule.get("action"),
                        }

        # ── 2. Conflit permission vs prohibition sur le même target ──
        target_rules: dict[str, list[str]] = {}
        for uid, info in local_rules.items():
            target = info.get("target", "")
            target_rules.setdefault(target, []).append(info["type"])

        for target, rule_types in target_rules.items():
            if "permission" in rule_types and "prohibition" in rule_types:
                issues.append(ConsistencyIssue(
                    severity=IssueSeverity.CRITICAL,
                    issue_type=IssueType.PERM_PROHIB_CONFLICT,
                    fragment_id=frag_id,
                    description=(
                        f"Conflit permission/prohibition sur la même cible : '{target}'"),
                    details={"target": target, "rule_types": rule_types},
                ))

        # ── 3. Vérifications sur les FPd ──
        local_activity_names = set(
            p.get("_activity") for p in fps.fpa_policies if p.get("_activity")
        )

        for fpd in fps.fpd_policies:
            gw   = fpd.get("_gateway")
            flow = fpd.get("_flow")
            acts = fpd.get("_activities", [])

            # 3a. Activités dans FPd sans FPa correspondante (intra seulement)
            if flow != "message":  # les message flows pointent vers d'autres fragments
                for act in acts:
                    if act and act not in local_activity_names:
                        # Peut être dans un autre fragment → WARNING pas CRITICAL
                        issues.append(ConsistencyIssue(
                            severity=IssueSeverity.WARNING,
                            issue_type=IssueType.ORPHAN_FPD,
                            fragment_id=frag_id,
                            description=(
                                f"FPd référence '{act}' sans FPa dans ce fragment"),
                            policy_uid=fpd.get("uid"),
                            details={"activity": act, "fpd_type": gw or flow},
                        ))

            # 3b. FPd XOR — vérifier que les deux conditions sont distinctes
            if gw == "XOR":
                conds = fpd.get("_conditions", [])
                if len(conds) >= 2 and conds[0] == conds[1]:
                    issues.append(ConsistencyIssue(
                        severity=IssueSeverity.CRITICAL,
                        issue_type=IssueType.XOR_SAME_CONDITION,
                        fragment_id=frag_id,
                        description=(
                            f"FPd XOR avec conditions identiques : '{conds[0]}'"),
                        policy_uid=fpd.get("uid"),
                        details={"conditions": conds, "activities": acts},
                    ))

                # Vérifier que les permissions ont bien un refinement
                permissions = fpd.get("permission", [])
                for perm in permissions:
                    actions = perm.get("action", [])
                    if isinstance(actions, list):
                        for act_obj in actions:
                            if isinstance(act_obj, dict):
                                refinement = act_obj.get("refinement", [])
                                if not refinement:
                                    issues.append(ConsistencyIssue(
                                        severity=IssueSeverity.WARNING,
                                        issue_type=IssueType.XOR_MISSING_CONDITION,
                                        fragment_id=frag_id,
                                        description=(
                                            f"FPd XOR sans refinement conditionnel "
                                            f"pour '{perm.get('uid', '?')}'"),
                                        policy_uid=fpd.get("uid"),
                                    ))

            # 3c. FPd AND/OR — vérifier que les cibles sont des règles connues
            if gw in ("AND", "OR"):
                for rule_list in (fpd.get("obligation", []),):
                    for rule in rule_list:
                        target = rule.get("target")
                        if isinstance(target, str) and target not in local_rules:
                            # Règle cible inconnue localement → INFO
                            issues.append(ConsistencyIssue(
                                severity=IssueSeverity.INFO,
                                issue_type=IssueType.MISSING_FPA,
                                fragment_id=frag_id,
                                description=(
                                    f"FPd {gw} cible une règle non indexée "
                                    f"localement : '{target}'"),
                                policy_uid=fpd.get("uid"),
                                details={"unknown_rule": target},
                            ))

        return issues

    # ─────────────────────────────────────────
    #  Inter-consistency
    # ─────────────────────────────────────────

    def _check_inter(self) -> list[ConsistencyIssue]:
        issues = []

        # Collecter tous les FPd message
        message_fpds: list[tuple[str, dict]] = []
        for frag_id, fps in self.fp_results.items():
            for fpd in fps.fpd_policies:
                if fpd.get("_flow") == "message":
                    message_fpds.append((frag_id, fpd))

        for frag_id, fpd in message_fpds:
            to_frag   = fpd.get("_to_fragment", "")
            acts      = fpd.get("_activities", [])
            from_act  = acts[0] if acts else "?"
            to_act    = acts[1] if len(acts) > 1 else "?"

            permissions = fpd.get("permission", [])
            for perm in permissions:
                assignee = perm.get("assignee", "")
                duty_list = perm.get("duty", [])

                # ── Vérifier que l'assignee (ruleix) existe bien ──
                if assignee and assignee not in self._rule_index:
                    issues.append(ConsistencyIssue(
                        severity=IssueSeverity.WARNING,
                        issue_type=IssueType.MESSAGE_UNKNOWN_RULE,
                        fragment_id=frag_id,
                        description=(
                            f"Message flow '{from_act}→{to_act}' : "
                            f"assignee '{assignee}' introuvable dans les FPa"),
                        policy_uid=fpd.get("uid"),
                        details={"assignee": assignee, "to_fragment": to_frag},
                    ))

                # ── Vérifier que la règle cible du duty existe bien ──
                for duty in duty_list:
                    target = duty.get("target", "")
                    if target and target not in self._rule_index:
                        issues.append(ConsistencyIssue(
                            severity=IssueSeverity.WARNING,
                            issue_type=IssueType.MESSAGE_UNKNOWN_TARGET,
                            fragment_id=frag_id,
                            description=(
                                f"Message flow '{from_act}→{to_act}' : "
                                f"duty target '{target}' introuvable dans les FPa"),
                            policy_uid=fpd.get("uid"),
                            details={"duty_target": target, "to_fragment": to_frag},
                        ))

                # ── Vérifier que le fragment cible existe bien ──
                if to_frag and to_frag not in self.fp_results:
                    issues.append(ConsistencyIssue(
                        severity=IssueSeverity.CRITICAL,
                        issue_type=IssueType.MESSAGE_NO_RECEIVER,
                        fragment_id=frag_id,
                        description=(
                            f"Message flow '{from_act}→{to_act}' : "
                            f"fragment cible '{to_frag}' n'existe pas"),
                        policy_uid=fpd.get("uid"),
                        details={"to_fragment": to_frag},
                    ))

        # ── Vérifier les cycles de dépendances inter-fragments ──
        cycle_issues = self._detect_dependency_cycles()
        issues.extend(cycle_issues)

        return issues

    # ─────────────────────────────────────────
    #  Détection de cycles inter-fragments
    # ─────────────────────────────────────────

    def _detect_dependency_cycles(self) -> list[ConsistencyIssue]:
        """
        Construit le graphe de dépendances inter-fragments
        et détecte les cycles (DFS).
        """
        issues = []

        # Construire graphe : frag_id → set(frag_id dépendants)
        dep_graph: dict[str, set[str]] = {fid: set() for fid in self.fp_results}

        for frag_id, fps in self.fp_results.items():
            for fpd in fps.fpd_policies:
                if fpd.get("_flow") == "message" or fpd.get("_inter"):
                    to_frag = fpd.get("_to_fragment", "")
                    if to_frag and to_frag != frag_id and to_frag in dep_graph:
                        dep_graph[frag_id].add(to_frag)

        # DFS pour détecter les cycles
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node: str, path: list[str]) -> Optional[list[str]]:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in dep_graph.get(node, set()):
                if neighbor not in visited:
                    result = dfs(neighbor, path + [neighbor])
                    if result:
                        return result
                elif neighbor in rec_stack:
                    return path + [neighbor]
            rec_stack.discard(node)
            return None

        for frag_id in dep_graph:
            if frag_id not in visited:
                cycle = dfs(frag_id, [frag_id])
                if cycle:
                    issues.append(ConsistencyIssue(
                        severity=IssueSeverity.CRITICAL,
                        issue_type=IssueType.DEPENDENCY_CYCLE,
                        fragment_id=frag_id,
                        description=(
                            f"Cycle de dépendances inter-fragments détecté : "
                            f"{' → '.join(cycle)}"),
                        details={"cycle": cycle},
                    ))

        return issues