"""
policy_auditor.py — Agent 5 : Policy Auditor

Responsabilité unique :
    Valider les policies ODRL JSON-LD générées par l'Agent 4
    selon trois niveaux de vérification, tous déterministes, sans LLM.

═══════════════════════════════════════════════════════════════════════
NIVEAU 1 — SYNTAXE ODRL (structure JSON-LD)
───────────────────────────────────────────
Vérifie que chaque policy respecte le schéma W3C ODRL 2.2 :
  - Champs obligatoires : @context, uid, @type
  - @type valide : Agreement | Offer | Set
  - Au moins une règle (permission | prohibition | obligation)
  - Chaque règle possède : uid, target, action
  - Tous les UIDs sont des URIs bien formées (http:// ou https://)
  - Aucun uid dupliqué au sein d'une même policy
  - Chaque constraint a ses 3 champs : leftOperand, operator, rightOperand
  - Chaque refinement a ses 3 champs : leftOperand, operator, rightOperand

Score syntaxique : 1 - (N_errors / N_checks)   [formule AgentODRL]

═══════════════════════════════════════════════════════════════════════
NIVEAU 2 — COHÉRENCE SÉMANTIQUE FPa  (policy d'activité)
──────────────────────────────────────────────────────────
Source de vérité : la B2P source de chaque FPa (champ _source_b2p).
Vérifie que la FPa encode fidèlement la B2P d'origine :
  - Le type de règle correspond (permission / prohibition / obligation)
  - L'assigner est présent et correspond
  - L'assignee est présent et correspond
  - Le target (asset) correspond
  - Chaque constraint de la B2P est présente dans la FPa
    (comparaison leftOperand + operator + rightOperand)
  - Pas de règles supplémentaires absentes de la B2P

═══════════════════════════════════════════════════════════════════════
NIVEAU 3 — COHÉRENCE SÉMANTIQUE FPd  (policy de dépendance)
─────────────────────────────────────────────────────────────
Source de vérité : l'EnrichedGraph d'Agent 1 + le ValidationReport
d'Agent 3 pour les dépendances implicites.

  FPd XOR     → les activités encodées correspondent aux branches
                réelles du fork XOR dans le graphe ;
                les conditions encodées correspondent aux conditions
                des arêtes sortantes de cette gateway.

  FPd AND     → les activités correspondent aux branches réelles
                du fork AND dans le graphe.

  FPd OR      → idem pour fork OR.

  FPd SEQ     → une ConnectionInfo (sequence, is_inter=False) existe
                entre from_activity et to_activity dans le graphe.

  FPd MESSAGE → une ConnectionInfo (message, is_inter=True) existe
                entre les deux fragments et les deux activités.

  FPd IMPLICIT→ une ConnectionInfo implicite acceptée par Agent 3
                existe avec les mêmes source/target ; le type de
                règle ODRL généré correspond au suggested_odrl_rule
                validé par Agent 3.

═══════════════════════════════════════════════════════════════════════
NIVEAU 4 — COHÉRENCE GLOBALE (intra + inter fragments)
────────────────────────────────────────────────────────
Hérité et étendu du ConsistencyAuditor original :
  - Conflit permission/prohibition sur le même target
  - FPd XOR avec deux conditions identiques
  - FPd XOR sans refinement conditionnel
  - FPd AND/OR ciblant des règles non indexées
  - Activités référencées dans FPd sans FPa locale
  - Message flows vers fragments inexistants
  - Assignee / duty target introuvables dans l'index global
  - Cycles de dépendances inter-fragments

═══════════════════════════════════════════════════════════════════════
Sorties :
    AuditIssue          : une anomalie individuelle
    AuditReport         : rapport complet — is_valid si aucun CRITICAL

Couche multi-agent :
    - receive()  : accepte POLICIES_READY depuis Agent 4
    - send()     : émet SYNTAX_CORRECTION vers Agent 4,
                   ou ODRL_VALID / ODRL_SYNTAX_ERROR vers le pipeline
    - Boucle syntaxique (max MAX_SYNTAX_LOOPS tours) :
        issues CRITICAL SYNTAX → SYNTAX_CORRECTION → Agent 4 → POLICIES_READY
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .structural_analyzer import (
    AgentMessage,
    EnrichedGraph,
    ConnectionInfo,
    MessageType,
    StructuralPattern,
)
from .constraint_validator import ValidationReport
from .policy_projection_agent import FragmentPolicySet


# ══════════════════════════════════════════════════════════════════════
#  Enums
# ══════════════════════════════════════════════════════════════════════

class IssueSeverity(Enum):
    CRITICAL = "critical"   # anomalie bloquante — policy invalide
    WARNING  = "warning"    # anomalie potentielle — policy douteuse
    INFO     = "info"       # observation — pas un problème


class IssueLayer(Enum):
    """Quel niveau a détecté l'anomalie."""
    SYNTAX    = "syntax"       # Niveau 1
    FPA_SEM   = "fpa_semantic" # Niveau 2
    FPD_SEM   = "fpd_semantic" # Niveau 3
    GLOBAL    = "global"       # Niveau 4


class IssueCode(Enum):
    # ── Syntaxe ──────────────────────────────────────────────────────
    MISSING_CONTEXT           = "missing_@context"
    MISSING_UID               = "missing_uid"
    MISSING_TYPE              = "missing_@type"
    INVALID_TYPE              = "invalid_@type"
    MALFORMED_URI             = "malformed_uri"
    NO_RULE                   = "no_rule_found"
    RULE_MISSING_UID          = "rule_missing_uid"
    RULE_MISSING_TARGET       = "rule_missing_target"
    RULE_MISSING_ACTION       = "rule_missing_action"
    DUPLICATE_RULE_UID        = "duplicate_rule_uid"
    CONSTRAINT_INCOMPLETE     = "constraint_missing_field"
    REFINEMENT_INCOMPLETE     = "refinement_missing_field"

    # ── FPa sémantique ───────────────────────────────────────────────
    FPA_WRONG_RULE_TYPE       = "fpa_wrong_rule_type"
    FPA_WRONG_ASSIGNER        = "fpa_wrong_assigner"
    FPA_WRONG_ASSIGNEE        = "fpa_wrong_assignee"
    FPA_WRONG_TARGET          = "fpa_wrong_target"
    FPA_MISSING_CONSTRAINT    = "fpa_missing_constraint"
    FPA_EXTRA_RULE            = "fpa_extra_rule_not_in_b2p"
    FPA_NO_B2P_SOURCE         = "fpa_no_b2p_source"

    # ── FPd sémantique ───────────────────────────────────────────────
    FPD_UNKNOWN_GATEWAY       = "fpd_unknown_gateway"
    FPD_WRONG_XOR_ACTIVITY    = "fpd_wrong_xor_activity"
    FPD_WRONG_XOR_CONDITION   = "fpd_wrong_xor_condition"
    FPD_WRONG_AND_ACTIVITY    = "fpd_wrong_and_activity"
    FPD_WRONG_OR_ACTIVITY     = "fpd_wrong_or_activity"
    FPD_SEQ_NOT_IN_GRAPH      = "fpd_seq_connection_not_in_graph"
    FPD_MSG_NOT_IN_GRAPH      = "fpd_msg_connection_not_in_graph"
    FPD_IMPLICIT_NOT_FOUND    = "fpd_implicit_not_in_validated"
    FPD_IMPLICIT_WRONG_TYPE   = "fpd_implicit_wrong_rule_type"

    # ── Global ───────────────────────────────────────────────────────
    PERM_PROHIB_CONFLICT      = "permission_prohibition_conflict"
    XOR_SAME_CONDITION        = "xor_same_condition"
    XOR_MISSING_CONDITION     = "xor_missing_condition"
    MISSING_FPA               = "missing_fpa_for_fpd_activity"
    DUPLICATE_RULE            = "duplicate_rule_uid_global"
    MESSAGE_NO_RECEIVER       = "message_flow_no_receiver"
    MESSAGE_UNKNOWN_RULE      = "message_flow_unknown_rule"
    MESSAGE_UNKNOWN_TARGET    = "message_flow_unknown_target"
    DEPENDENCY_CYCLE          = "inter_fragment_dependency_cycle"
    ORPHAN_FPD                = "orphan_fpd_no_matching_fpa"


# ══════════════════════════════════════════════════════════════════════
#  Dataclasses résultats
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AuditIssue:
    severity:    IssueSeverity
    layer:       IssueLayer
    code:        IssueCode
    fragment_id: str
    description: str
    policy_uid:  Optional[str] = None
    suggestion:  Optional[str] = None
    details:     Optional[dict] = None

    def __str__(self) -> str:
        icon = {
            IssueSeverity.CRITICAL: "❌",
            IssueSeverity.WARNING:  "⚠️ ",
            IssueSeverity.INFO:     "ℹ️ ",
        }[self.severity]
        return (
            f"{icon} [{self.severity.value.upper():8s}]"
            f"[{self.layer.value:12s}]"
            f"[{self.code.value}] "
            f"({self.fragment_id}) {self.description}"
        )


@dataclass
class AuditReport:
    issues: list[AuditIssue] = field(default_factory=list)

    # ── Filtres par layer ──────────────────────────────────────────
    @property
    def syntax_issues(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.layer == IssueLayer.SYNTAX]

    @property
    def fpa_issues(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.layer == IssueLayer.FPA_SEM]

    @property
    def fpd_issues(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.layer == IssueLayer.FPD_SEM]

    @property
    def global_issues(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.layer == IssueLayer.GLOBAL]

    # ── Filtres par sévérité ───────────────────────────────────────
    @property
    def criticals(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]

    @property
    def warnings(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    @property
    def infos(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.INFO]

    # ── Validité globale ───────────────────────────────────────────
    @property
    def is_valid(self) -> bool:
        """True si aucune issue CRITICAL, tous niveaux confondus."""
        return len(self.criticals) == 0

    @property
    def is_syntactically_valid(self) -> bool:
        return not any(
            i.severity == IssueSeverity.CRITICAL
            for i in self.syntax_issues
        )

    @property
    def is_semantically_valid(self) -> bool:
        return not any(
            i.severity == IssueSeverity.CRITICAL
            for i in (self.fpa_issues + self.fpd_issues)
        )

    # ── Score syntaxique (formule AgentODRL) ──────────────────────
    def syntax_score(self, n_checks: int = 10) -> float:
        n_errors = sum(1 for i in self.syntax_issues
                       if i.severity == IssueSeverity.CRITICAL)
        return max(0.0, 1.0 - n_errors / n_checks)

    def summary(self) -> dict:
        # ── Agrégation basique des compteurs ─────────────────────────
        base = {
            "is_valid":              self.is_valid,
            "is_syntactically_valid": self.is_syntactically_valid,
            "is_semantically_valid":  self.is_semantically_valid,
            "total_issues":          len(self.issues),
            "critical":              len(self.criticals),
            "warning":               len(self.warnings),
            "info":                  len(self.infos),
            "by_layer": {
                "syntax":       len(self.syntax_issues),
                "fpa_semantic": len(self.fpa_issues),
                "fpd_semantic": len(self.fpd_issues),
                "global":       len(self.global_issues),
            },
        }

        # ── Décomposition de la conformité syntaxique selon C1..C9 ──
        #
        # On projette les IssueCode syntaxiques existants sur les
        # critères de Mustafa Daham et al. (C1..C9). Quand un critère
        # n'est pas directement vérifié par nos règles actuelles,
        # on le laisse à None (non évalué).
        syntax_criticals = [
            i for i in self.syntax_issues
            if i.severity == IssueSeverity.CRITICAL
        ]
        syntax_codes = {i.code for i in syntax_criticals}

        # C1 — odrl:uid pour Policy / Rule (approximation via UID + URI)
        c1_ok = not any(
            c in syntax_codes
            for c in (
                IssueCode.MISSING_UID,
                IssueCode.RULE_MISSING_UID,
                IssueCode.MALFORMED_URI,
            )
        )

        # C5 — Relation target présent
        c5_ok = IssueCode.RULE_MISSING_TARGET not in syntax_codes

        # C6 — Action présente
        c6_ok = IssueCode.RULE_MISSING_ACTION not in syntax_codes

        # C7 — Au moins une règle dans la policy
        c7_ok = IssueCode.NO_RULE not in syntax_codes

        # C8 — Triple (leftOperand, operator, rightOperand) complet
        c8_ok = not any(
            c in syntax_codes
            for c in (
                IssueCode.CONSTRAINT_INCOMPLETE,
                IssueCode.REFINEMENT_INCOMPLETE,
            )
        )

        base["syntax_criteria"] = {
            # C1 : ODRL UID (Policy / Rule)
            "C1_uid": c1_ok,
            # C2 : Data type specification (non évalué explicitement)
            "C2_datatype": None,
            # C3 : Meta info (dc:creator/title/description/issued) — non évalué
            "C3_meta_info": None,
            # C4 : Function / Party (assigner/assignee de type Party) — non évalué
            "C4_function_party": None,
            # C5 : Relation (target Asset présent)
            "C5_relation_target": c5_ok,
            # C6 : Action (action de type Action présente)
            "C6_action": c6_ok,
            # C7 : Rule (au moins une règle Permission/Prohibition/Obligation)
            "C7_rule": c7_ok,
            # C8 : Constraint complète (leftOperand/operator/rightOperand)
            "C8_constraint": c8_ok,
            # C9 : ODRL Extension (profil / extension contrôlée) — non évalué
            "C9_odrl_extension": None,
        }

        return base


# ══════════════════════════════════════════════════════════════════════
#  Agent 5 — Policy Auditor
# ══════════════════════════════════════════════════════════════════════

class PolicyAuditor:
    """
    Agent 5 du pipeline multi-agent.

    Fusionne la validation syntaxique ODRL et la vérification sémantique
    déterministe des FPa (via B2P source) et des FPd (via EnrichedGraph),
    ainsi que l'audit de cohérence globale intra/inter-fragment.

    Mode standalone :
        Instancier avec fp_results, enriched_graph, raw_b2p et appeler audit().

    Mode pipeline :
        Instancier avec fp_results={}, enriched_graph=None, raw_b2p=b2p_policies.
        Les données réelles arrivent via receive(POLICIES_READY).
        Enregistrer le callback de sortie avant le démarrage.

    Boucle syntaxique :
        Si audit() détecte des issues CRITICAL de layer SYNTAX,
        Agent 5 émet SYNTAX_CORRECTION vers Agent 4 (max MAX_SYNTAX_LOOPS tours).
        Agent 4 applique les corrections et ré-émet POLICIES_READY.
        Une fois la limite atteinte ou aucune issue syntaxique critique,
        Agent 5 émet le signal final ODRL_VALID ou ODRL_SYNTAX_ERROR.

    Paramètres
    ----------
    fp_results      : dict[fragment_id → FragmentPolicySet]  (Agent 4)
    enriched_graph  : EnrichedGraph                           (Agent 1)
    raw_b2p         : list[dict]  — policies B2P originales
    validation_report : ValidationReport | None               (Agent 3)
                        Nécessaire pour valider les FPd IMPLICIT.
    """

    # Types ODRL acceptés
    _VALID_TYPES = {"Agreement", "Offer", "Set"}

    # URI minimale : doit commencer par http:// ou https://
    _URI_RE = re.compile(r"^https?://")

    # Types de règles ODRL
    _RULE_TYPES = ("permission", "prohibition", "obligation")

    AGENT_NAME       = "agent5"
    MAX_SYNTAX_LOOPS = 2   # max de tours dans la boucle syntaxique Agent5↔Agent4

    def __init__(
        self,
        fp_results:         dict[str, FragmentPolicySet],
        enriched_graph:     Optional[EnrichedGraph],
        raw_b2p:            list[dict],
        validation_report:  Optional[ValidationReport] = None,
    ):
        self.fp_results        = fp_results
        self.enriched_graph    = enriched_graph
        self.raw_b2p           = raw_b2p
        self.validation_report = validation_report

        # ── Couche multi-agent ──────────────────────────────────────
        self._on_send:           Optional[Callable[[AgentMessage], None]] = None
        self._syntax_loop_count: int = 0

        # ── Index B2P : uid → policy dict ──────────────────────────
        self._b2p_index: dict[str, dict] = {
            b.get("uid", ""): b for b in raw_b2p if b.get("uid")
        }

        # ── Index ConnectionInfo par type ──────────────────────────
        self._seq_index:     dict[tuple[str, str], ConnectionInfo] = {}
        self._msg_index:     dict[tuple[str, str], ConnectionInfo] = {}
        self._pattern_index: dict[str, StructuralPattern] = {}
        self._fork_edges:    dict[str, list[ConnectionInfo]] = {}

        # ── Index global rule_uid → (fragment_id, policy_uid, rule_type)
        self._rule_index: dict[str, tuple[str, str, str]] = {}

        # ── Index validated candidates (Agent 3) ───────────────────
        self._validated_index: dict[tuple[str, str], object] = {}

        # Construire les index seulement si on a un graphe (mode standalone)
        if self.enriched_graph is not None:
            self._build_indexes()

    # ═════════════════════════════════════════
    #  COUCHE MULTI-AGENT
    # ═════════════════════════════════════════

    def register_send_callback(self, fn: Callable[[AgentMessage], None]) -> None:
        """Enregistre le callback d'envoi (typiquement collector.receive)."""
        self._on_send = fn

    def send(self, msg: AgentMessage) -> None:
        """Émet un message via le callback enregistré."""
        print(f"[Agent 5] ► SEND {msg}")
        if self._on_send:
            self._on_send(msg)
        else:
            print(f"[Agent 5][WARN] Aucun callback — message {msg.msg_type.value} non transmis.")

    def receive(self, msg: AgentMessage) -> None:
        """
        Point d'entrée des messages entrants.

        Messages acceptés :
          - POLICIES_READY : policies générées depuis Agent 4

        Tout autre type est ignoré avec un warning.
        """
        print(f"[Agent 5] ◄ RECEIVE {msg}")
        if msg.msg_type == MessageType.POLICIES_READY:
            fp_results     = msg.payload["fp_results"]
            enriched_graph = msg.payload["enriched_graph"]

            # Mettre à jour l'état interne et reconstruire les index
            self.fp_results     = fp_results
            self.enriched_graph = enriched_graph
            self._build_indexes()

            self._audit_and_route(loop_turn=msg.loop_turn)
        else:
            print(f"[Agent 5][WARN] Message '{msg.msg_type.value}' non géré — ignoré.")

    # ─────────────────────────────────────────
    #  Routing interne
    # ─────────────────────────────────────────

    def _audit_and_route(self, loop_turn: int) -> None:
        """
        Audite les policies courantes et route vers la destination appropriée.

        Logique :
          1. Appeler audit() → AuditReport
          2. Si au moins une issue CRITICAL ET _syntax_loop_count < MAX_SYNTAX_LOOPS :
               → incrémenter _syntax_loop_count
               → émettre SYNTAX_CORRECTION vers Agent 4
               → RETURN (boucle de correction)
          3. Sinon : émettre le signal final ODRL_VALID ou ODRL_SYNTAX_ERROR

        On utilise _syntax_loop_count (compteur propre à Agent 5), pas loop_turn du message,
        car loop_turn peut être > MAX_SYNTAX_LOOPS après la boucle Agent 2↔3 (reformulation).
        """
        report = self.audit()

        # ── Toute issue CRITICAL déclenche la boucle de correction ───
        any_critical = report.criticals
        critical_syntax_count = sum(1 for i in any_critical if i.layer == IssueLayer.SYNTAX)

        # ── Boucle de correction Agent 5 → Agent 4 ───────────────────
        # Limite basée sur le compteur local, pas sur loop_turn (qui vient de la boucle 2↔3)
        if any_critical and self._syntax_loop_count < self.MAX_SYNTAX_LOOPS:
            self._syntax_loop_count += 1

            errors = [
                {
                    "policy_uid": i.policy_uid,
                    "code":       i.code.value,
                    "layer":      i.layer.value,
                    "path":       (i.details or {}).get("path", ""),
                    "suggestion": i.suggestion,
                }
                for i in any_critical
            ]

            affected_policies = list({
                i.policy_uid
                for i in any_critical
                if i.policy_uid
            })

            print(
                f"[Agent 5] {len(any_critical)} issue(s) CRITICAL détectée(s) "
                f"({critical_syntax_count} syntax) — émission SYNTAX_CORRECTION "
                f"vers Agent 4 (boucle #{self._syntax_loop_count})"
            )

            self.send(AgentMessage(
                sender    = self.AGENT_NAME,
                recipient = "agent4",
                msg_type  = MessageType.SYNTAX_CORRECTION,
                payload   = {
                    "affected_policies": affected_policies,
                    "errors":            errors,
                },
                loop_turn = self._syntax_loop_count,  # tour de la boucle 5↔4
            ))
            return

        # ── Signal final ─────────────────────────────────────────────
        if any_critical and self._syntax_loop_count >= self.MAX_SYNTAX_LOOPS:
            print(
                f"[Agent 5][WARN] MAX_SYNTAX_LOOPS ({self.MAX_SYNTAX_LOOPS}) atteint — "
                "émission du rapport final avec erreurs restantes."
            )

        final_type = (
            MessageType.ODRL_VALID
            if report.is_valid
            else MessageType.ODRL_SYNTAX_ERROR
        )

        print(
            f"[Agent 5] Signal final : {final_type.value} — "
            f"is_valid={report.is_valid}, "
            f"syntax_loops_used={self._syntax_loop_count}"
        )

        self.send(AgentMessage(
            sender    = self.AGENT_NAME,
            recipient = "pipeline",
            msg_type  = final_type,
            payload   = {
                "report":          report,
                "is_valid":        report.is_valid,
                "summary":         report.summary(),
                "syntax_score":    report.syntax_score(),
                "loop_turns_used": self._syntax_loop_count,
            },
            loop_turn = loop_turn,
        ))

    # ═════════════════════════════════════════
    #  LOGIQUE MÉTIER (inchangée)
    # ═════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════
    #  Construction des index
    # ══════════════════════════════════════════════════════════════

    def _build_indexes(self) -> None:
        # Réinitialiser les index avant reconstruction
        # (nécessaire en mode pipeline où les données changent entre les passes)
        self._seq_index      = {}
        self._msg_index      = {}
        self._pattern_index  = {}
        self._fork_edges     = {}
        self._rule_index     = {}
        self._validated_index = {}

        if self.enriched_graph is None:
            return

        # Connexions du graphe
        for conn in self.enriched_graph.connections:
            key = (conn.from_activity, conn.to_activity)
            if conn.connection_type == "message":
                self._msg_index[key] = conn
            elif conn.connection_type == "sequence":
                self._seq_index[key] = conn

        # Patterns : gateway_name → pattern + arêtes sortantes
        for pattern in self.enriched_graph.patterns:
            if pattern.gateway_name:
                self._pattern_index[pattern.gateway_name] = pattern

        # Index global des règles (pour cohérence inter-fragment)
        for frag_id, fps in self.fp_results.items():
            for policy in fps.fpa_policies:
                for rt in self._RULE_TYPES:
                    for rule in policy.get(rt, []):
                        uid = rule.get("uid")
                        if uid:
                            self._rule_index[uid] = (
                                frag_id, policy.get("uid", ""), rt
                            )

        # Index des candidats validés par Agent 3
        if self.validation_report:
            for candidate in self.validation_report.validated_candidates():
                key = (candidate.source_activity, candidate.target_activity)
                self._validated_index[key] = candidate

    def _get_fork_branches(self, gateway_name: str) -> list[ConnectionInfo]:
        """
        Retourne les ConnectionInfo correspondant aux branches sortantes
        d'une gateway identifiée par son nom.
        On cherche dans le graphe BPMN le nœud dont le nom correspond,
        puis on récupère les arêtes sortantes via les ConnectionInfo.

        Les ConnectionInfo portent la condition de la branche quand
        elle est définie dans le BPMN source.
        """
        branches: list[ConnectionInfo] = []
        for conn in self.enriched_graph.connections:
            # Agent 1 encode le fork via ConnectionInfo.gateway_name
            if conn.gateway_name != gateway_name:
                continue
            # On ne conserve que les branches de contrôle (XOR/AND/OR)
            if conn.connection_type not in {"xor", "and", "or"}:
                continue
            branches.append(conn)
        return branches


    # ══════════════════════════════════════════════════════════════
    #  Point d'entrée standalone
    # ══════════════════════════════════════════════════════════════

    def audit(self) -> AuditReport:
        print("[Agent 5] PolicyAuditor — démarrage de l'audit")
        report = AuditReport()

        for frag_id, fps in self.fp_results.items():
            print(f"[Agent 5] Fragment '{frag_id}' "
                  f"({len(fps.fpa_policies)} FPa, {len(fps.fpd_policies)} FPd)")

            # Niveau 1 — syntaxe de toutes les policies
            for policy in fps.all_policies():
                report.issues.extend(
                    self._check_syntax(policy, frag_id)
                )

            # Niveau 2 — sémantique FPa
            for policy in fps.fpa_policies:
                report.issues.extend(
                    self._check_fpa_semantics(policy, frag_id)
                )

            # Niveau 3 — sémantique FPd
            for fpd in fps.fpd_policies:
                report.issues.extend(
                    self._check_fpd_semantics(fpd, frag_id)
                )

            # Niveau 4 intra — cohérence interne du fragment
            report.issues.extend(
                self._check_intra_consistency(frag_id, fps)
            )

        # Niveau 4 inter — cohérence entre fragments
        report.issues.extend(self._check_inter_consistency())

        s = report.summary()
        status = "✅ VALIDE" if s["is_valid"] else "❌ ANOMALIES CRITIQUES"
        print(
            f"[Agent 5] Audit terminé : {status} — "
            f"{s['critical']} critical, {s['warning']} warning, {s['info']} info"
        )
        return report

    # ══════════════════════════════════════════════════════════════
    #  NIVEAU 1 — Syntaxe ODRL
    # ══════════════════════════════════════════════════════════════

    def _check_syntax(self, policy: dict, frag_id: str) -> list[AuditIssue]:
        issues: list[AuditIssue] = []
        uid = policy.get("uid") or policy.get("@id", "?")

        def issue(severity, code, description, suggestion=None, details=None):
            return AuditIssue(
                severity=severity, layer=IssueLayer.SYNTAX, code=code,
                fragment_id=frag_id, policy_uid=uid,
                description=description, suggestion=suggestion, details=details,
            )

        # @context
        if "@context" not in policy:
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.MISSING_CONTEXT,
                "Champ @context manquant.",
                'Ajouter "@context": "http://www.w3.org/ns/odrl.jsonld"',
                {"path": "@context"},
            ))

        # uid
        if not (policy.get("uid") or policy.get("@id")):
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.MISSING_UID,
                "Champ uid manquant.",
                "Ajouter un uid sous forme d'URI.",
                {"path": "uid"},
            ))
        elif not self._URI_RE.match(str(uid)):
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.MALFORMED_URI,
                f"uid '{uid}' n'est pas une URI valide.",
                "L'uid doit commencer par http:// ou https://",
                {"path": "uid"},
            ))

        # @type
        ptype = policy.get("@type")
        if not ptype:
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.MISSING_TYPE,
                "Champ @type manquant.",
                'Ajouter "@type": "Agreement"',
                {"path": "@type"},
            ))
        elif ptype not in self._VALID_TYPES:
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.INVALID_TYPE,
                f"@type '{ptype}' invalide. Attendu : {self._VALID_TYPES}",
                'Corriger en "@type": "Agreement"',
                {"path": "@type"},
            ))

        # Au moins une règle
        has_rules = any(policy.get(rt) for rt in self._RULE_TYPES)
        if not has_rules:
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.NO_RULE,
                "Aucune règle (permission/prohibition/obligation).",
                "Ajouter au moins une règle.",
            ))
            return issues  # inutile d'aller plus loin

        # Vérification des règles
        seen_uids: set[str] = set()
        for rt in self._RULE_TYPES:
            for i, rule in enumerate(policy.get(rt, [])):
                issues.extend(
                    self._check_rule_syntax(rule, f"{rt}[{i}]", frag_id, uid, seen_uids)
                )

        return issues

    def _check_rule_syntax(
        self,
        rule: dict,
        path: str,
        frag_id: str,
        policy_uid: str,
        seen_uids: set[str],
    ) -> list[AuditIssue]:
        issues: list[AuditIssue] = []

        def issue(severity, code, description, suggestion=None, details=None):
            return AuditIssue(
                severity=severity, layer=IssueLayer.SYNTAX, code=code,
                fragment_id=frag_id, policy_uid=policy_uid,
                description=f"{path} : {description}",
                suggestion=suggestion,
                details=details,
            )

        # uid de la règle
        rule_uid = rule.get("uid")
        if not rule_uid:
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.RULE_MISSING_UID,
                "uid manquant.",
                f"Ajouter un uid URI à la règle {path}.",
                {"path": path},
            ))
        else:
            if not self._URI_RE.match(str(rule_uid)):
                issues.append(issue(
                    IssueSeverity.CRITICAL, IssueCode.MALFORMED_URI,
                    f"uid '{rule_uid}' n'est pas une URI valide.",
                    details={"path": path},
                ))
            if rule_uid in seen_uids:
                issues.append(issue(
                    IssueSeverity.WARNING, IssueCode.DUPLICATE_RULE_UID,
                    f"uid '{rule_uid}' dupliqué dans cette policy.",
                    "Générer un uid unique pour chaque règle.",
                    {"path": path},
                ))
            else:
                seen_uids.add(rule_uid)

        # target
        if not rule.get("target"):
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.RULE_MISSING_TARGET,
                "target manquant.",
                "Ajouter un target (URI de l'asset).",
                {"path": path},
            ))

        # action
        if not rule.get("action"):
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.RULE_MISSING_ACTION,
                "action manquante.",
                "Ajouter une action ODRL (ex: odrl:use, odrl:trigger).",
                {"path": path},
            ))
        else:
            actions = rule["action"] if isinstance(rule["action"], list) else [rule["action"]]
            for j, act in enumerate(actions):
                if isinstance(act, dict):
                    for k, ref in enumerate(act.get("refinement", [])):
                        if isinstance(ref, dict):
                            issues.extend(self._check_triple_fields(
                                ref, f"{path}.action[{j}].refinement[{k}]",
                                frag_id, policy_uid, is_refinement=True,
                            ))

        # constraints au niveau règle
        for k, cst in enumerate(rule.get("constraint", [])):
            if isinstance(cst, dict):
                issues.extend(self._check_triple_fields(
                    cst, f"{path}.constraint[{k}]",
                    frag_id, policy_uid, is_refinement=False,
                ))

        return issues

    def _check_triple_fields(
        self,
        obj: dict,
        path: str,
        frag_id: str,
        policy_uid: str,
        is_refinement: bool,
    ) -> list[AuditIssue]:
        issues: list[AuditIssue] = []
        code = (IssueCode.REFINEMENT_INCOMPLETE if is_refinement
                else IssueCode.CONSTRAINT_INCOMPLETE)
        label = "refinement" if is_refinement else "constraint"

        for field_name in ("leftOperand", "operator", "rightOperand"):
            if field_name not in obj:
                issues.append(AuditIssue(
                    severity=IssueSeverity.CRITICAL,
                    layer=IssueLayer.SYNTAX,
                    code=code,
                    fragment_id=frag_id,
                    policy_uid=policy_uid,
                    description=(
                        f"{path} : champ '{field_name}' manquant "
                        f"dans ce {label}."
                    ),
                    suggestion=f"Ajouter '{field_name}' au {label}.",
                    details={"path": path},
                ))
        return issues

    # ══════════════════════════════════════════════════════════════
    #  NIVEAU 2 — Sémantique FPa
    # ══════════════════════════════════════════════════════════════

    def _check_fpa_semantics(self, policy: dict, frag_id: str) -> list[AuditIssue]:
        issues: list[AuditIssue] = []
        uid        = policy.get("uid", "?")
        source_b2p = policy.get("_source_b2p")

        def issue(severity, code, description, suggestion=None):
            return AuditIssue(
                severity=severity, layer=IssueLayer.FPA_SEM, code=code,
                fragment_id=frag_id, policy_uid=uid,
                description=description, suggestion=suggestion,
            )

        if not source_b2p:
            issues.append(issue(
                IssueSeverity.INFO, IssueCode.FPA_NO_B2P_SOURCE,
                "FPa sans _source_b2p — policy minimale générée (pas de B2P source).",
            ))
            return issues

        b2p = self._b2p_index.get(source_b2p)
        if not b2p:
            issues.append(issue(
                IssueSeverity.WARNING, IssueCode.FPA_NO_B2P_SOURCE,
                f"B2P source '{source_b2p}' introuvable dans raw_b2p.",
                "Vérifier que la B2P source est incluse dans raw_b2p.",
            ))
            return issues

        # Vérifier les types de règles présentes dans B2P vs FPa
        b2p_rule_types = {rt for rt in self._RULE_TYPES if b2p.get(rt)}
        fpa_rule_types = {rt for rt in self._RULE_TYPES if policy.get(rt)}

        for rt in b2p_rule_types - fpa_rule_types:
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.FPA_WRONG_RULE_TYPE,
                f"Type de règle '{rt}' présent dans B2P mais absent de la FPa.",
                f"Ajouter les règles de type '{rt}' depuis la B2P source.",
            ))

        for rt in fpa_rule_types - b2p_rule_types:
            issues.append(issue(
                IssueSeverity.WARNING, IssueCode.FPA_EXTRA_RULE,
                f"Type de règle '{rt}' présent dans FPa mais absent de la B2P source.",
            ))

        # Vérifier les constraints de chaque règle
        for rt in b2p_rule_types & fpa_rule_types:
            b2p_rules = b2p.get(rt, [])
            fpa_rules = policy.get(rt, [])

            for b2p_rule in b2p_rules:
                # Trouver la règle correspondante dans la FPa (par uid ou par ordre)
                matching_fpa_rule = next(
                    (r for r in fpa_rules if r.get("uid") == b2p_rule.get("uid")),
                    fpa_rules[0] if fpa_rules else None,
                )
                if not matching_fpa_rule:
                    continue

                # Vérifier les constraints
                b2p_constraints = b2p_rule.get("constraint", [])
                fpa_constraints = matching_fpa_rule.get("constraint", [])

                b2p_cst_keys = {
                    (c.get("leftOperand"), c.get("operator"))
                    for c in b2p_constraints
                    if isinstance(c, dict)
                }
                fpa_cst_keys = {
                    (c.get("leftOperand"), c.get("operator"))
                    for c in fpa_constraints
                    if isinstance(c, dict)
                }

                for missing_key in b2p_cst_keys - fpa_cst_keys:
                    issues.append(issue(
                        IssueSeverity.WARNING, IssueCode.FPA_MISSING_CONSTRAINT,
                        (f"Constraint (leftOperand='{missing_key[0]}', "
                         f"operator='{missing_key[1]}') présente dans B2P "
                         f"mais absente de la FPa."),
                        "Copier la constraint depuis la B2P source.",
                    ))

                # Vérifier assigner / assignee
                for field_name in ("assigner", "assignee"):
                    b2p_val = b2p_rule.get(field_name)
                    fpa_val = matching_fpa_rule.get(field_name)
                    if b2p_val and fpa_val and b2p_val != fpa_val:
                        code_map = {
                            "assigner": IssueCode.FPA_WRONG_ASSIGNER,
                            "assignee": IssueCode.FPA_WRONG_ASSIGNEE,
                        }
                        issues.append(issue(
                            IssueSeverity.WARNING, code_map[field_name],
                            (f"{field_name} FPa '{fpa_val}' ≠ B2P '{b2p_val}'."),
                        ))

        return issues

    # ══════════════════════════════════════════════════════════════
    #  NIVEAU 3 — Sémantique FPd
    # ══════════════════════════════════════════════════════════════

    def _check_fpd_semantics(self, fpd: dict, frag_id: str) -> list[AuditIssue]:
        issues: list[AuditIssue] = []
        fpd_uid   = fpd.get("uid", "?")
        gateway   = fpd.get("_gateway")
        flow      = fpd.get("_flow")
        is_impl   = fpd.get("_implicit", False)
        acts      = fpd.get("_activities", [])
        gw_name   = fpd.get("_gateway_name", "")
        conditions = fpd.get("_conditions", [])

        def issue(severity, code, description, suggestion=None):
            return AuditIssue(
                severity=severity, layer=IssueLayer.FPD_SEM, code=code,
                fragment_id=frag_id, policy_uid=fpd_uid,
                description=description, suggestion=suggestion,
            )

        # ── FPd avec gateway ────────────────────────────────────────
        if gateway in ("XOR", "AND", "OR"):
            pattern = self._pattern_index.get(gw_name)
            if not pattern:
                issues.append(issue(
                    IssueSeverity.WARNING, IssueCode.FPD_UNKNOWN_GATEWAY,
                    f"Gateway '{gw_name}' introuvable dans les patterns du graphe.",
                    "Vérifier que le nom de gateway est correct.",
                ))
                return issues

            # Résoudre les branches réelles depuis le graphe
            real_branches = self._get_fork_branches_names(gw_name)

            if gateway == "XOR":
                for act in acts:
                    if act not in real_branches:
                        issues.append(issue(
                            IssueSeverity.CRITICAL, IssueCode.FPD_WRONG_XOR_ACTIVITY,
                            (f"Activité '{act}' encodée dans FPd XOR '{gw_name}' "
                             f"ne correspond pas aux branches réelles : {real_branches}."),
                        ))

                # Vérifier les conditions
                real_conditions = self._get_fork_conditions(gw_name)
                for cond in conditions:
                    if real_conditions and cond not in real_conditions:
                        issues.append(issue(
                            IssueSeverity.WARNING, IssueCode.FPD_WRONG_XOR_CONDITION,
                            (f"Condition '{cond}' encodée dans FPd XOR "
                             f"ne correspond pas aux conditions réelles : {real_conditions}."),
                        ))

            elif gateway == "AND":
                for act in acts:
                    if act not in real_branches:
                        issues.append(issue(
                            IssueSeverity.CRITICAL, IssueCode.FPD_WRONG_AND_ACTIVITY,
                            (f"Activité '{act}' encodée dans FPd AND '{gw_name}' "
                             f"ne correspond pas aux branches réelles : {real_branches}."),
                        ))

            elif gateway == "OR":
                for act in acts:
                    if act not in real_branches:
                        issues.append(issue(
                            IssueSeverity.CRITICAL, IssueCode.FPD_WRONG_OR_ACTIVITY,
                            (f"Activité '{act}' encodée dans FPd OR '{gw_name}' "
                             f"ne correspond pas aux branches réelles : {real_branches}."),
                        ))

        # ── FPd séquence ────────────────────────────────────────────
        elif flow == "sequence":
            if len(acts) >= 2:
                key = (acts[0], acts[1])
                if key not in self._seq_index:
                    issues.append(issue(
                        IssueSeverity.CRITICAL, IssueCode.FPD_SEQ_NOT_IN_GRAPH,
                        (f"Connexion séquence '{acts[0]}' → '{acts[1]}' "
                         f"introuvable dans le graphe."),
                        "Vérifier que la connexion existe dans l'EnrichedGraph.",
                    ))

        # ── FPd message ─────────────────────────────────────────────
        elif flow == "message":
            if len(acts) >= 2:
                src, dst = acts[0], acts[1]
                key = (src, dst)

                # 1) Si une ConnectionInfo explicite de type "message" existe, c'est OK.
                conn = self._msg_index.get(key)

                # 2) Sinon, on accepte aussi le cas où il existe une connexion
                #    inter-fragment (is_inter=True) entre ces deux activités,
                #    quel que soit son connection_type (sequence/xor/implicit...).
                if conn is None and self.enriched_graph is not None:
                    for c in self.enriched_graph.connections:
                        if (
                            c.from_activity == src
                            and c.to_activity == dst
                            and c.is_inter
                        ):
                            conn = c
                            break

                if conn is None:
                    issues.append(issue(
                        IssueSeverity.CRITICAL, IssueCode.FPD_MSG_NOT_IN_GRAPH,
                        (f"Dépendance inter-fragment '{src}' → '{dst}' "
                         f"introuvable dans le graphe (aucune ConnectionInfo inter-fragment)."),
                        "Vérifier qu'une connexion inter-fragment existe entre ces activités.",
                    ))

        # ── FPd implicite ───────────────────────────────────────────
        elif is_impl:
            if len(acts) >= 2:
                key = (acts[0], acts[1])
                candidate = self._validated_index.get(key)

                if not candidate:
                    issues.append(issue(
                        IssueSeverity.WARNING, IssueCode.FPD_IMPLICIT_NOT_FOUND,
                        (f"FPd implicite '{acts[0]}' → '{acts[1]}' non trouvée "
                         f"dans les candidats validés par Agent 3."),
                    ))
                else:
                    # Vérifier que le type de règle correspond
                    suggested = getattr(candidate, "suggested_odrl_rule", None) or "obligation"
                    dep_type  = fpd.get("_dep_type", "")
                    fpa_has   = {rt for rt in self._RULE_TYPES if fpd.get(rt)}

                    if suggested not in fpa_has:
                        issues.append(issue(
                            IssueSeverity.WARNING, IssueCode.FPD_IMPLICIT_WRONG_TYPE,
                            (f"FPd implicite encodée avec règle type {fpa_has} "
                             f"mais Agent 3 suggérait '{suggested}'."),
                        ))

        return issues

    def _get_fork_branches_names(self, gateway_name: str) -> list[str]:
        """Retourne les noms des activités cibles d'un fork gateway."""
        branches = self._get_fork_branches(gateway_name)
        return [b.to_activity for b in branches]

    def _get_fork_conditions(self, gateway_name: str) -> list[str]:
        """Retourne les conditions des arêtes sortantes d'un fork gateway."""
        branches = self._get_fork_branches(gateway_name)
        return [b.condition for b in branches if b.condition]

    # ══════════════════════════════════════════════════════════════
    #  NIVEAU 4 — Cohérence intra-fragment
    # ══════════════════════════════════════════════════════════════

    def _check_intra_consistency(
        self, frag_id: str, fps: FragmentPolicySet
    ) -> list[AuditIssue]:
        issues: list[AuditIssue] = []

        # Index local : target → liste de types de règles
        target_rules: dict[str, list[str]] = {}
        # Index local : rule_uid → policy_uid
        local_rules: dict[str, str] = {}

        for policy in fps.all_policies():
            for rt in self._RULE_TYPES:
                for rule in policy.get(rt, []):
                    tgt = rule.get("target", "")
                    uid = rule.get("uid", "")

                    if isinstance(tgt, str) and tgt:
                        target_rules.setdefault(tgt, []).append(rt)

                    if uid:
                        if uid in local_rules:
                            issues.append(AuditIssue(
                                severity=IssueSeverity.WARNING,
                                layer=IssueLayer.GLOBAL,
                                code=IssueCode.DUPLICATE_RULE,
                                fragment_id=frag_id,
                                policy_uid=policy.get("uid"),
                                description=(
                                    f"uid de règle '{uid}' dupliqué "
                                    f"dans le fragment '{frag_id}'."
                                ),
                            ))
                        else:
                            local_rules[uid] = policy.get("uid", "")

        # Conflit permission/prohibition sur le même target
        for tgt, rule_types in target_rules.items():
            if "permission" in rule_types and "prohibition" in rule_types:
                issues.append(AuditIssue(
                    severity=IssueSeverity.CRITICAL,
                    layer=IssueLayer.GLOBAL,
                    code=IssueCode.PERM_PROHIB_CONFLICT,
                    fragment_id=frag_id,
                    description=(
                        f"Conflit permission/prohibition sur le target '{tgt}' "
                        f"dans le fragment '{frag_id}'."
                    ),
                ))

        # Vérifications XOR
        for fpd in fps.fpd_policies:
            if fpd.get("_gateway") != "XOR":
                continue

            conditions = fpd.get("_conditions", [])

            # Deux conditions identiques
            if len(conditions) >= 2 and conditions[0] == conditions[1]:
                issues.append(AuditIssue(
                    severity=IssueSeverity.WARNING,
                    layer=IssueLayer.GLOBAL,
                    code=IssueCode.XOR_SAME_CONDITION,
                    fragment_id=frag_id,
                    policy_uid=fpd.get("uid"),
                    description=(
                        f"FPd XOR : les deux conditions sont identiques "
                        f"('{conditions[0]}')."
                    ),
                ))

            # FPd XOR — refinement absent
            for perm in fpd.get("permission", []):
                for act_obj in (perm.get("action", [])
                                if isinstance(perm.get("action"), list)
                                else [perm.get("action", {})]):
                    if isinstance(act_obj, dict) and not act_obj.get("refinement"):
                        issues.append(AuditIssue(
                            severity=IssueSeverity.WARNING,
                            layer=IssueLayer.GLOBAL,
                            code=IssueCode.XOR_MISSING_CONDITION,
                            fragment_id=frag_id,
                            policy_uid=fpd.get("uid"),
                            description=(
                                f"FPd XOR : permission '{perm.get('uid', '?')}' "
                                f"sans refinement conditionnel."
                            ),
                        ))

            # FPd AND/OR — cibles inconnues localement
            if fpd.get("_gateway") in ("AND", "OR"):
                gw = fpd.get("_gateway")
                for rule in fpd.get("obligation", []):
                    tgt = rule.get("target")
                    if isinstance(tgt, str) and tgt not in local_rules:
                        issues.append(AuditIssue(
                            severity=IssueSeverity.INFO,
                            layer=IssueLayer.GLOBAL,
                            code=IssueCode.MISSING_FPA,
                            fragment_id=frag_id,
                            policy_uid=fpd.get("uid"),
                            description=(
                                f"FPd {gw} : obligation cible '{tgt}' "
                                f"non indexée localement."
                            ),
                            details={"unknown_rule": tgt},
                        ))

        return issues

    # ══════════════════════════════════════════════════════════════
    #  NIVEAU 4 — Cohérence globale inter-fragments
    # ══════════════════════════════════════════════════════════════

    def _check_inter_consistency(self) -> list[AuditIssue]:
        issues: list[AuditIssue] = []

        for frag_id, fps in self.fp_results.items():
            for fpd in fps.fpd_policies:
                if fpd.get("_flow") != "message":
                    continue

                to_frag  = fpd.get("_to_fragment", "")
                acts     = fpd.get("_activities", [])
                from_act = acts[0] if acts else "?"
                to_act   = acts[1] if len(acts) > 1 else "?"
                fpd_uid  = fpd.get("uid", "?")

                # Fragment cible inexistant
                if to_frag and to_frag not in self.fp_results:
                    issues.append(AuditIssue(
                        severity=IssueSeverity.CRITICAL,
                        layer=IssueLayer.GLOBAL,
                        code=IssueCode.MESSAGE_NO_RECEIVER,
                        fragment_id=frag_id,
                        policy_uid=fpd_uid,
                        description=(
                            f"Message flow '{from_act} → {to_act}' : "
                            f"fragment cible '{to_frag}' inexistant."
                        ),
                        details={"to_fragment": to_frag},
                    ))

                for perm in fpd.get("permission", []):
                    # Assignee (ruleix) introuvable
                    assignee = perm.get("assignee", "")
                    if assignee and assignee not in self._rule_index:
                        issues.append(AuditIssue(
                            severity=IssueSeverity.WARNING,
                            layer=IssueLayer.GLOBAL,
                            code=IssueCode.MESSAGE_UNKNOWN_RULE,
                            fragment_id=frag_id,
                            policy_uid=fpd_uid,
                            description=(
                                f"Message flow '{from_act} → {to_act}' : "
                                f"assignee '{assignee}' introuvable dans les FPa."
                            ),
                            details={"assignee": assignee, "to_fragment": to_frag},
                        ))

                    # Duty target introuvable
                    for duty in perm.get("duty", []):
                        tgt = duty.get("target", "")
                        if tgt and tgt not in self._rule_index:
                            issues.append(AuditIssue(
                                severity=IssueSeverity.WARNING,
                                layer=IssueLayer.GLOBAL,
                                code=IssueCode.MESSAGE_UNKNOWN_TARGET,
                                fragment_id=frag_id,
                                policy_uid=fpd_uid,
                                description=(
                                    f"Message flow '{from_act} → {to_act}' : "
                                    f"duty target '{tgt}' introuvable dans les FPa."
                                ),
                                details={"duty_target": tgt, "to_fragment": to_frag},
                            ))

        issues.extend(self._detect_dependency_cycles())
        return issues

    def _detect_dependency_cycles(self) -> list[AuditIssue]:
        """DFS pour détecter les cycles de dépendances inter-fragments."""
        issues: list[AuditIssue] = []

        dep_graph: dict[str, set[str]] = {fid: set() for fid in self.fp_results}
        for frag_id, fps in self.fp_results.items():
            for fpd in fps.fpd_policies:
                if fpd.get("_flow") == "message" or fpd.get("_inter"):
                    to_frag = fpd.get("_to_fragment", "")
                    if to_frag and to_frag != frag_id and to_frag in dep_graph:
                        dep_graph[frag_id].add(to_frag)

        visited:   set[str] = set()
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
                    issues.append(AuditIssue(
                        severity=IssueSeverity.CRITICAL,
                        layer=IssueLayer.GLOBAL,
                        code=IssueCode.DEPENDENCY_CYCLE,
                        fragment_id=frag_id,
                        description=(
                            f"Cycle de dépendances inter-fragments : "
                            f"{' → '.join(cycle)}"
                        ),
                        details={"cycle": cycle},
                    ))

        return issues

    # ══════════════════════════════════════════════════════════════
    #  Rapport console
    # ══════════════════════════════════════════════════════════════

    def print_report(self, report: AuditReport) -> None:
        s = report.summary()
        print("\n" + "═" * 65)
        print("  AGENT 5 — Policy Audit Report")
        print("═" * 65)
        print(f"  Status      : {'✅ VALIDE' if s['is_valid'] else '❌ ANOMALIES CRITIQUES'}")
        print(f"  Critical    : {s['critical']}")
        print(f"  Warning     : {s['warning']}")
        print(f"  Info        : {s['info']}")
        print(f"  Par niveau  :")
        for layer, count in s["by_layer"].items():
            print(f"    {layer:14s} : {count}")
        print()
        for issue in report.issues:
            print(f"  {issue}")
            if issue.suggestion:
                print(f"    ↳ Suggestion : {issue.suggestion}")
        print("═" * 65)