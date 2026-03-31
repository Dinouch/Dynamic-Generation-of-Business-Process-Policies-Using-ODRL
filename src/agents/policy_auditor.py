"""
policy_auditor.py — Agent 5 : Policy Auditor

Responsabilité unique :
    Valider les policies ODRL JSON-LD générées par l'Agent 4
    selon deux niveaux de vérification, tous déterministes, sans LLM.

═══════════════════════════════════════════════════════════════════════
NIVEAU 1 — SYNTAXE ODRL (structure JSON-LD)
───────────────────────────────────────────
Validation via SHACL (pyshacl + pyld) quand disponible,
repli sur vérifications manuelles sinon.

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

Note sur uid / @id :
  Dans le contexte officiel ODRL JSON-LD, "uid" est un alias de "@id".
  pyld expand() ne produit donc pas de triple odrl:uid distinct.
  _dedupe_uid_atid_collision() supprime "uid" redondant avant pyld
  pour éviter l'erreur "colliding keywords".
  odrl_shapes.ttl utilise sh:nodeKind sh:IRI sur les NodeShapes
  (et non sh:path odrl:uid) pour s'aligner sur ce comportement.

═══════════════════════════════════════════════════════════════════════
NIVEAU 4 — COHÉRENCE GLOBALE (intra + inter fragments)
────────────────────────────────────────────────────────
  - Conflit permission/prohibition sur le même target
  - FPd XOR avec deux conditions identiques
  - FPd XOR sans refinement conditionnel
  - FPd AND/OR ciblant des règles non indexées
  - Activités référencées dans FPd sans FPa locale
  - Message flows vers fragments inexistants
  - Assignee / duty target introuvables dans l'index global
  - Cycles de dépendances inter-fragments

═══════════════════════════════════════════════════════════════════════
Niveaux 2 et 3 (FPa/FPd sémantiques) : délégués à l'Agent 3.

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

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
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
    CRITICAL = "critical"
    WARNING  = "warning"
    INFO     = "info"


class IssueLayer(Enum):
    SYNTAX = "syntax"   # Niveau 1 (SHACL ou repli manuel)
    GLOBAL = "global"   # Niveau 4 — cohérence inter/intra fragment


class IssueCode(Enum):
    # ── Syntaxe ──────────────────────────────────────────────────────
    MISSING_CONTEXT       = "missing_@context"
    MISSING_UID           = "missing_uid"
    MISSING_TYPE          = "missing_@type"
    INVALID_TYPE          = "invalid_@type"
    MALFORMED_URI         = "malformed_uri"
    NO_RULE               = "no_rule_found"
    RULE_MISSING_UID      = "rule_missing_uid"
    RULE_MISSING_TARGET   = "rule_missing_target"
    RULE_MISSING_ACTION   = "rule_missing_action"
    DUPLICATE_RULE_UID    = "duplicate_rule_uid"
    CONSTRAINT_INCOMPLETE = "constraint_missing_field"
    REFINEMENT_INCOMPLETE = "refinement_missing_field"
    SHACL_VIOLATION       = "shacl_violation"

    # ── Global ───────────────────────────────────────────────────────
    PERM_PROHIB_CONFLICT  = "permission_prohibition_conflict"
    XOR_SAME_CONDITION    = "xor_same_condition"
    XOR_MISSING_CONDITION = "xor_missing_condition"
    MISSING_FPA           = "missing_fpa_for_fpd_activity"
    DUPLICATE_RULE        = "duplicate_rule_uid_global"
    MESSAGE_NO_RECEIVER   = "message_flow_no_receiver"
    MESSAGE_UNKNOWN_RULE  = "message_flow_unknown_rule"
    MESSAGE_UNKNOWN_TARGET = "message_flow_unknown_target"
    DEPENDENCY_CYCLE      = "inter_fragment_dependency_cycle"
    ORPHAN_FPD            = "orphan_fpd_no_matching_fpa"


# ══════════════════════════════════════════════════════════════════════
#  Helpers JSON-LD / pyld
# ══════════════════════════════════════════════════════════════════════

_RULE_LIST_KEYS = frozenset(("permission", "prohibition", "obligation"))


def _strip_uid_if_redundant_with_atid(d: dict) -> None:
    """
    Remove ``uid`` when it is identical to ``@id``.

    In the official ODRL JSON-LD context, ``uid`` is an alias for ``@id``.
    Having both with the same value causes pyld to raise
    "colliding keywords detected".
    """
    if "@id" in d and "uid" in d and str(d["@id"]) == str(d["uid"]):
        del d["uid"]


def _dedupe_rule_list_value(val: object) -> object:
    """Recursively deduplicate uid/@id in a rule list or single rule dict."""
    if isinstance(val, list):
        return [_dedupe_uid_atid_collision(x) for x in val]
    if isinstance(val, dict):
        return _dedupe_uid_atid_collision(val)
    return val


def _strip_uid_in_nested_rule_lists(obj: object) -> None:
    """
    After the recursive build, strip redundant ``uid`` from every dict
    directly under permission / prohibition / obligation lists, including
    nested ones — remaining source of pyld "colliding keywords" on rules.
    """
    if isinstance(obj, dict):
        for rk in _RULE_LIST_KEYS:
            if rk not in obj:
                continue
            seq = obj[rk]
            if isinstance(seq, list):
                for item in seq:
                    if isinstance(item, dict):
                        _strip_uid_if_redundant_with_atid(item)
            elif isinstance(seq, dict):
                _strip_uid_if_redundant_with_atid(seq)
        for v in obj.values():
            _strip_uid_in_nested_rule_lists(v)
    elif isinstance(obj, list):
        for x in obj:
            _strip_uid_in_nested_rule_lists(x)


def _dedupe_uid_atid_collision(obj: object) -> object:
    """
    Remove redundant ``uid`` keys before passing to pyld expand().

    The official ODRL context maps ``uid`` → ``@id``.  Having both with the
    same value triggers pyld "colliding keywords".  This function strips the
    redundant ``uid`` on a deep copy of the object, covering:
      - the top-level policy dict
      - every rule dict in permission / prohibition / obligation lists
      - all nested structures via a final recursive pass
    """
    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            if k in _RULE_LIST_KEYS:
                out[k] = _dedupe_rule_list_value(v)
            else:
                out[k] = _dedupe_uid_atid_collision(v)
        _strip_uid_if_redundant_with_atid(out)
        _strip_uid_in_nested_rule_lists(out)
        return out
    if isinstance(obj, list):
        return [_dedupe_uid_atid_collision(x) for x in obj]
    return obj


def _odrl_jsonld_to_rdflib_graph(odrl: dict):
    """
    Convert an ODRL JSON-LD dict (already deduplicated) to an rdflib Graph
    via pyld expand() + to_rdf() in N-Quads format.

    Using pyld instead of rdflib's built-in JSON-LD parser gives correct
    handling of remote contexts (e.g. ``@type: @id`` terms like ``uid``).

    The caller is responsible for stripping internal ``_`` keys and calling
    _dedupe_uid_atid_collision() before passing ``odrl`` here.

    Parameters
    ----------
    odrl : dict
        Clean, deduplicated ODRL policy dict (no ``_`` internal keys,
        no duplicate ``@id``/``uid``).

    Returns
    -------
    rdflib.Graph
    """
    from pyld import jsonld
    from rdflib import Graph

    # Deep copy to avoid mutating caller's dict
    doc = json.loads(json.dumps(odrl))
    expanded = jsonld.expand(doc)
    nquads = jsonld.to_rdf(expanded, {"format": "application/n-quads"})
    g = Graph()
    if nquads and str(nquads).strip():
        g.parse(data=nquads, format="nquads")
    return g


def _shacl_report_to_issue_code(report_text: str) -> IssueCode:
    """
    Map a pyshacl report text to an IssueCode that Agent 4 can handle
    via its SYNTAX_CORRECTION handler.

    SHACL_VIOLATION must not be emitted toward Agent 4 — only mapped codes.
    """
    msg = (report_text or "").lower()
    if "context" in msg or "@context" in msg:
        return IssueCode.MISSING_CONTEXT
    if "missing_@type" in msg or ("type" in msg and "agreement" in msg and "invalid" in msg):
        return IssueCode.MISSING_TYPE
    if "uid" in msg and ("missing" in msg or "absent" in msg or "less than" in msg or "mincount" in msg):
        return IssueCode.MISSING_UID
    if "uri" in msg or "iri" in msg or "malformed" in msg or "nodekind" in msg:
        return IssueCode.MALFORMED_URI
    if "action" in msg and ("mincount" in msg or "less than" in msg):
        return IssueCode.RULE_MISSING_ACTION
    if "target" in msg and ("mincount" in msg or "less than" in msg):
        return IssueCode.RULE_MISSING_TARGET
    if "rule" in msg and "uid" in msg:
        return IssueCode.RULE_MISSING_UID
    if "leftoperand" in msg or "rightoperand" in msg or "operator" in msg:
        return IssueCode.CONSTRAINT_INCOMPLETE
    return IssueCode.MISSING_CONTEXT


def _syntax_correction_wire_code(code: IssueCode) -> str:
    """
    Return the string code expected by Agent 4's SYNTAX_CORRECTION handler.

    Agent 4 uses string matching on ``error["code"]``, not IssueCode enum values.
    """
    mapping = {
        IssueCode.MISSING_CONTEXT:       "MISSING_CONTEXT",
        IssueCode.MISSING_UID:           "MISSING_UID",
        IssueCode.MISSING_TYPE:          "MISSING_TYPE",
        IssueCode.INVALID_TYPE:          "MISSING_TYPE",
        IssueCode.MALFORMED_URI:         "MALFORMED_URI",
        IssueCode.NO_RULE:               "NO_RULE",
        IssueCode.RULE_MISSING_UID:      "RULE_MISSING_UID",
        IssueCode.RULE_MISSING_TARGET:   "RULE_MISSING_TARGET",
        IssueCode.RULE_MISSING_ACTION:   "RULE_MISSING_ACTION",
        IssueCode.DUPLICATE_RULE_UID:    "DUPLICATE_RULE_UID",
        IssueCode.CONSTRAINT_INCOMPLETE: "CONSTRAINT_INCOMPLETE",
        IssueCode.REFINEMENT_INCOMPLETE: "REFINEMENT_INCOMPLETE",
    }
    return mapping.get(code, code.name)


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

    @property
    def syntax_issues(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.layer == IssueLayer.SYNTAX]

    @property
    def fpa_issues(self) -> list[AuditIssue]:
        """Deprecated — FPa semantics validated in Agent 3."""
        return []

    @property
    def fpd_issues(self) -> list[AuditIssue]:
        """Deprecated — FPd semantics validated in Agent 3."""
        return []

    @property
    def global_issues(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.layer == IssueLayer.GLOBAL]

    @property
    def criticals(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]

    @property
    def warnings(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    @property
    def infos(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.INFO]

    @property
    def is_valid(self) -> bool:
        return len(self.criticals) == 0

    @property
    def is_syntactically_valid(self) -> bool:
        return not any(i.severity == IssueSeverity.CRITICAL for i in self.syntax_issues)

    @property
    def is_semantically_valid(self) -> bool:
        """Semantic validation delegated to Agent 3 — always True here if no CRITICAL global."""
        return True

    def syntax_score(self, n_checks: int = 10) -> float:
        n_errors = sum(1 for i in self.syntax_issues if i.severity == IssueSeverity.CRITICAL)
        return max(0.0, 1.0 - n_errors / n_checks)

    def summary(self) -> dict:
        base = {
            "is_valid":               self.is_valid,
            "is_syntactically_valid": self.is_syntactically_valid,
            "is_semantically_valid":  self.is_semantically_valid,
            "total_issues":           len(self.issues),
            "critical":               len(self.criticals),
            "warning":                len(self.warnings),
            "info":                   len(self.infos),
            "by_layer": {
                "syntax":       len(self.syntax_issues),
                "fpa_semantic": len(self.fpa_issues),
                "fpd_semantic": len(self.fpd_issues),
                "global":       len(self.global_issues),
            },
        }

        syntax_criticals = [i for i in self.syntax_issues if i.severity == IssueSeverity.CRITICAL]
        syntax_codes = {i.code for i in syntax_criticals}

        c1_ok = not any(c in syntax_codes for c in (
            IssueCode.MISSING_UID, IssueCode.RULE_MISSING_UID, IssueCode.MALFORMED_URI,
        ))
        c5_ok = IssueCode.RULE_MISSING_TARGET not in syntax_codes
        c6_ok = IssueCode.RULE_MISSING_ACTION not in syntax_codes
        c7_ok = IssueCode.NO_RULE not in syntax_codes
        c8_ok = not any(c in syntax_codes for c in (
            IssueCode.CONSTRAINT_INCOMPLETE, IssueCode.REFINEMENT_INCOMPLETE,
        ))

        base["syntax_criteria"] = {
            "C1_uid":             c1_ok,
            "C2_datatype":        None,
            "C3_meta_info":       None,
            "C4_function_party":  None,
            "C5_relation_target": c5_ok,
            "C6_action":          c6_ok,
            "C7_rule":            c7_ok,
            "C8_constraint":      c8_ok,
            "C9_odrl_extension":  None,
        }
        return base


# ══════════════════════════════════════════════════════════════════════
#  Agent 5 — Policy Auditor
# ══════════════════════════════════════════════════════════════════════

class PolicyAuditor:
    """
    Agent 5 of the multi-agent pipeline.

    Validates ODRL JSON-LD policies generated by Agent 4 using two
    deterministic levels — no LLM calls.

    Level 1 — ODRL syntax via SHACL (pyshacl + pyld) with manual fallback.
    Level 4 — Global coherence (intra + inter fragment).

    Levels 2 and 3 (FPa / FPd semantics) are handled by Agent 3.

    Pipeline mode:
        Instantiate with fp_results={}, enriched_graph=None, raw_b2p=b2p_policies.
        Data arrives via receive(POLICIES_READY).
        Register the output callback before starting.

    Syntax loop:
        CRITICAL SYNTAX issues → SYNTAX_CORRECTION → Agent 4 → POLICIES_READY
        Max MAX_SYNTAX_LOOPS rounds; then emit final ODRL_VALID or ODRL_SYNTAX_ERROR.
    """

    _VALID_TYPES  = {"Agreement", "Offer", "Set"}
    _URI_RE       = re.compile(r"^https?://")
    _RULE_TYPES   = ("permission", "prohibition", "obligation")

    AGENT_NAME       = "agent5"
    MAX_SYNTAX_LOOPS = 2

    def __init__(
        self,
        fp_results:        dict[str, FragmentPolicySet],
        enriched_graph:    Optional[EnrichedGraph],
        raw_b2p:           list[dict],
        validation_report: Optional[ValidationReport] = None,
    ):
        self.fp_results        = fp_results
        self.enriched_graph    = enriched_graph
        self.raw_b2p           = raw_b2p
        self.validation_report = validation_report

        self._on_send:           Optional[Callable[[AgentMessage], None]] = None
        self._syntax_loop_count: int = 0

        self._b2p_index:      dict[str, dict]                  = {
            b.get("uid", ""): b for b in raw_b2p if b.get("uid")
        }
        self._seq_index:      dict[tuple[str, str], ConnectionInfo] = {}
        self._msg_index:      dict[tuple[str, str], ConnectionInfo] = {}
        self._pattern_index:  dict[str, StructuralPattern]          = {}
        self._fork_edges:     dict[str, list[ConnectionInfo]]        = {}
        self._rule_index:     dict[str, tuple[str, str, str]]        = {}
        self._validated_index: dict[tuple[str, str], object]         = {}

        if self.enriched_graph is not None:
            self._build_indexes()

    # ═════════════════════════════════════════
    #  Multi-agent layer
    # ═════════════════════════════════════════

    def register_send_callback(self, fn: Callable[[AgentMessage], None]) -> None:
        self._on_send = fn

    def send(self, msg: AgentMessage) -> None:
        print(f"[Agent 5] ► SEND {msg}")
        if self._on_send:
            self._on_send(msg)
        else:
            print(f"[Agent 5][WARN] No callback — message {msg.msg_type.value} not transmitted.")

    def receive(self, msg: AgentMessage) -> None:
        print(f"[Agent 5] ◄ RECEIVE {msg}")
        if msg.msg_type == MessageType.POLICIES_READY:
            self.fp_results     = msg.payload["fp_results"]
            self.enriched_graph = msg.payload["enriched_graph"]
            # Optional workflow metadata (ACL status / merge manifest)
            self._workflow_status = msg.payload.get("status")
            self._merge_manifest  = msg.payload.get("manifest")
            self._build_indexes()
            self._audit_and_route(loop_turn=msg.loop_turn)
        else:
            print(f"[Agent 5][WARN] Message '{msg.msg_type.value}' not handled — ignored.")

    def _audit_and_route(self, loop_turn: int) -> None:
        report = self.audit()
        any_critical = report.criticals
        critical_syntax_count = sum(1 for i in any_critical if i.layer == IssueLayer.SYNTAX)

        if any_critical and self._syntax_loop_count < self.MAX_SYNTAX_LOOPS:
            self._syntax_loop_count += 1
            errors = [
                {
                    "policy_uid": i.policy_uid,
                    "code":       _syntax_correction_wire_code(i.code),
                    "layer":      i.layer.value,
                    "path":       (i.details or {}).get("path", ""),
                    "suggestion": i.suggestion,
                }
                for i in any_critical
            ]
            affected_policies = list({i.policy_uid for i in any_critical if i.policy_uid})
            print(
                f"[Agent 5] {len(any_critical)} issue(s) CRITICAL "
                f"({critical_syntax_count} syntax) — SYNTAX_CORRECTION "
                f"to Agent 4 (loop #{self._syntax_loop_count})"
            )
            self.send(AgentMessage(
                sender    = self.AGENT_NAME,
                recipient = "agent4",
                msg_type  = MessageType.SYNTAX_CORRECTION,
                payload   = {"affected_policies": affected_policies, "errors": errors},
                loop_turn = self._syntax_loop_count,
            ))
            return

        if any_critical and self._syntax_loop_count >= self.MAX_SYNTAX_LOOPS:
            print(
                f"[Agent 5][WARN] MAX_SYNTAX_LOOPS ({self.MAX_SYNTAX_LOOPS}) reached — "
                "emitting final report with remaining errors."
            )

        final_type = MessageType.ODRL_VALID if report.is_valid else MessageType.ODRL_SYNTAX_ERROR
        print(
            f"[Agent 5] Final signal: {final_type.value} — "
            f"is_valid={report.is_valid}, syntax_loops_used={self._syntax_loop_count}"
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
                "status":          getattr(self, "_workflow_status", None),
                "manifest":        getattr(self, "_merge_manifest", None),
            },
            loop_turn = loop_turn,
        ))

    # ═════════════════════════════════════════
    #  Index construction
    # ═════════════════════════════════════════

    def _build_indexes(self) -> None:
        self._seq_index       = {}
        self._msg_index       = {}
        self._pattern_index   = {}
        self._fork_edges      = {}
        self._rule_index      = {}
        self._validated_index = {}

        if self.enriched_graph is None:
            return

        for conn in self.enriched_graph.connections:
            key = (conn.from_activity, conn.to_activity)
            if conn.connection_type == "message":
                self._msg_index[key] = conn
            elif conn.connection_type == "sequence":
                self._seq_index[key] = conn

        for pattern in self.enriched_graph.patterns:
            if pattern.gateway_name:
                self._pattern_index[pattern.gateway_name] = pattern

        for frag_id, fps in self.fp_results.items():
            for policy in fps.fpa_policies:
                for rt in self._RULE_TYPES:
                    for rule in policy.get(rt, []):
                        uid = rule.get("uid")
                        if uid:
                            self._rule_index[uid] = (frag_id, policy.get("uid", ""), rt)

        if self.validation_report:
            for p in getattr(self.validation_report, "accepted_unsupported_proposals", []) or []:
                key = (getattr(p, "gateway_name", ""), getattr(p, "fragment_id", ""))
                self._validated_index[key] = p

    def _get_fork_branches(self, gateway_name: str) -> list[ConnectionInfo]:
        branches: list[ConnectionInfo] = []
        for conn in self.enriched_graph.connections:
            if conn.gateway_name != gateway_name:
                continue
            if conn.connection_type not in {"xor", "and", "or"}:
                continue
            branches.append(conn)
        return branches

    # ═════════════════════════════════════════
    #  Main audit entry point
    # ═════════════════════════════════════════

    def audit(self) -> AuditReport:
        print("[Agent 5] PolicyAuditor — starting audit")
        report = AuditReport()

        for frag_id, fps in self.fp_results.items():
            print(f"[Agent 5] Fragment '{frag_id}' "
                  f"({len(fps.fpa_policies)} FPa, {len(fps.fpd_policies)} FPd)")
            for policy in fps.all_policies():
                report.issues.extend(self._check_syntax(policy, frag_id))
            report.issues.extend(self._check_intra_consistency(frag_id, fps))

        report.issues.extend(self._check_inter_consistency())

        s = report.summary()
        status = "✅ VALID" if s["is_valid"] else "❌ CRITICAL ISSUES"
        print(f"[Agent 5] Audit done: {status} — "
              f"{s['critical']} critical, {s['warning']} warning, {s['info']} info")
        return report

    # ══════════════════════════════════════════════════════════════
    #  Level 1 — ODRL syntax (SHACL + manual fallback)
    # ══════════════════════════════════════════════════════════════

    def _check_syntax(self, policy: dict, frag_id: str) -> list[AuditIssue]:
        """
        Validate one ODRL policy dict.

        Steps:
        1. Strip internal ``_`` keys.
        2. Deep-copy and deduplicate ``uid``/``@id`` collisions via
           _dedupe_uid_atid_collision() — required before passing to pyld.
        3. Try SHACL validation (passes ``deduped`` to _check_syntax_shacl).
        4. On any exception, fall back to manual field checks on ``deduped``.

        Both SHACL and manual fallback receive the same deduplicated dict,
        so manual checks never see spurious errors caused by the @id/uid
        collision that pyld would reject.
        """
        odrl   = {k: v for k, v in policy.items() if not str(k).startswith("_")}
        deduped = _dedupe_uid_atid_collision(json.loads(json.dumps(odrl)))
        try:
            return self._check_syntax_shacl(deduped, frag_id)
        except ImportError:
            print(
                "[Agent 5][WARN] pyshacl / rdflib / pyld not available — "
                "SHACL inactive; using manual ODRL field checks."
            )
            return self._check_syntax_manual(deduped, frag_id)
        except Exception as e:
            print(f"[Agent 5][WARN] SHACL validation error: {e} — manual fallback.")
            return self._check_syntax_manual(deduped, frag_id)

    def _check_syntax_shacl(self, policy: dict, frag_id: str) -> list[AuditIssue]:
        """
        Run pyshacl against agents/odrl_shapes.ttl for one policy dict.

        Parameters
        ----------
        policy
            Already deduplicated ODRL dict — no internal ``_`` keys,
            no duplicate ``@id``/``uid``.  Produced by _check_syntax().
        frag_id
            Fragment identifier for reporting.

        Notes
        -----
        _odrl_jsonld_to_rdflib_graph() receives ``policy`` as-is (already
        deduplicated) — it must NOT re-deduplicate internally.

        odrl_shapes.ttl uses ``sh:nodeKind sh:IRI`` on NodeShapes rather
        than ``sh:path odrl:uid`` because ``uid`` in the ODRL context is
        an alias for ``@id`` and never produces an ``odrl:uid`` RDF triple.
        """
        import pyshacl
        from rdflib import Graph

        # policy is already clean and deduplicated — pass directly
        data = _odrl_jsonld_to_rdflib_graph(policy)

        shapes = Graph()
        shapes_path = Path(__file__).resolve().parent / "odrl_shapes.ttl"
        shapes.parse(shapes_path)

        conforms, _rg, report_text = pyshacl.validate(
            data_graph=data,
            shacl_graph=shapes,
            inference="rdfs",
            abort_on_first=False,
        )

        # uid for reporting — after deduplication, @id holds the IRI
        uid = policy.get("@id") or policy.get("uid", "?")

        if conforms:
            
            return []

        sev = IssueSeverity.CRITICAL
        if "Warning" in (report_text or "") and "Violation" not in (report_text or ""):
            sev = IssueSeverity.WARNING

        mapped   = _shacl_report_to_issue_code(report_text or "")
        path_hint = {
            IssueCode.MISSING_UID:     "uid",
            IssueCode.MISSING_CONTEXT: "@context",
            IssueCode.MISSING_TYPE:    "@type",
            IssueCode.MALFORMED_URI:   "uid",
        }.get(mapped, "")

        details: dict = {"shacl_report": report_text}
        if path_hint:
            details["path"] = path_hint

        # Provide a parseable suggestion so Agent 4's MALFORMED_URI handler
        # can extract the correct URI via regex rather than using free text.
        suggestion = None
        if mapped == IssueCode.MALFORMED_URI:
            suggestion = f"set uid to {uid}"
        elif mapped == IssueCode.MISSING_UID:
            suggestion = f"add uid as URI for policy {uid}"

        return [
            AuditIssue(
                severity    = sev,
                layer       = IssueLayer.SYNTAX,
                code        = mapped,
                fragment_id = frag_id,
                policy_uid  = str(uid),
                description = (report_text or "SHACL validation failed.")[:4000],
                suggestion  = suggestion,
                details     = details,
            )
        ]

    def _check_syntax_manual(self, policy: dict, frag_id: str) -> list[AuditIssue]:
        """
        Deterministic field-by-field ODRL syntax check.

        Receives the already-deduplicated dict from _check_syntax(), so
        ``uid`` may have been removed if ``@id`` was present — check both.
        """
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
                "Missing @context field.",
                'Add "@context": "http://www.w3.org/ns/odrl.jsonld"',
                {"path": "@context"},
            ))

        # uid / @id
        if not (policy.get("uid") or policy.get("@id")):
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.MISSING_UID,
                "Missing uid field.",
                "Add uid as a URI.",
                {"path": "uid"},
            ))
        elif not self._URI_RE.match(str(uid)):
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.MALFORMED_URI,
                f"uid '{uid}' is not a valid URI.",
                f"set uid to {uid}",
                {"path": "uid"},
            ))

        # @type
        ptype = policy.get("@type")
        if not ptype:
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.MISSING_TYPE,
                "Missing @type field.",
                'Add "@type": "Agreement"',
                {"path": "@type"},
            ))
        elif ptype not in self._VALID_TYPES:
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.INVALID_TYPE,
                f"@type '{ptype}' invalid. Expected: {self._VALID_TYPES}",
                'Fix to "@type": "Agreement"',
                {"path": "@type"},
            ))

        # At least one rule
        has_rules = any(policy.get(rt) for rt in self._RULE_TYPES)
        if not has_rules:
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.NO_RULE,
                "No rule found (permission/prohibition/obligation).",
                "Add at least one rule.",
            ))
            return issues

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
        """
        Validate a single ODRL rule dict (permission / prohibition / obligation).

        FIX — duty objects are NOT top-level rules:
            A ``duty`` dict nested inside a permission has ``"action"`` but
            typically no ``"uid"`` or ``"target"`` at the rule level.
            Validating duties as rules produces false RULE_MISSING_UID /
            RULE_MISSING_TARGET / RULE_MISSING_ACTION errors (e.g. on FPd SEQ
            where duty = {"action": "nextPolicy", "uid": <policy_uri>}).

            Guard: if the dict has no ``"target"`` AND no ``"uid"``/``"@id"``
            at this level, it is a duty-like nested object — skip it.
            This avoids false positives without masking real rule errors.
        """
        issues: list[AuditIssue] = []

        # ── Duty guard ────────────────────────────────────────────────
        # A duty nested inside a permission is not a top-level ODRL rule.
        # It will have "action" but no "target" and often no "uid".
        # Skip validation to avoid false RULE_MISSING_* errors.
        has_uid    = bool(rule.get("uid") or rule.get("@id"))
        has_target = bool(rule.get("target"))
        has_action = bool(rule.get("action"))
        if not has_uid and not has_target and has_action:
            # Looks like a duty object — skip
            
            return []

        def issue(severity, code, description, suggestion=None, details=None):
            return AuditIssue(
                severity=severity, layer=IssueLayer.SYNTAX, code=code,
                fragment_id=frag_id, policy_uid=policy_uid,
                description=f"{path} : {description}",
                suggestion=suggestion,
                details=details,
            )

        # uid
        rule_uid = rule.get("uid") or rule.get("@id")
        if not rule_uid:
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.RULE_MISSING_UID,
                "uid missing.",
                f"Add a URI uid to rule {path}.",
                {"path": path},
            ))
        else:
            rule_uid_str = str(rule_uid)
            if not self._URI_RE.match(rule_uid_str):
                issues.append(issue(
                    IssueSeverity.CRITICAL, IssueCode.MALFORMED_URI,
                    f"uid '{rule_uid_str}' is not a valid URI.",
                    details={"path": path},
                ))
            if rule_uid_str in seen_uids:
                issues.append(issue(
                    IssueSeverity.WARNING, IssueCode.DUPLICATE_RULE_UID,
                    f"uid '{rule_uid_str}' duplicated in this policy.",
                    "Generate a unique uid for each rule.",
                    {"path": path},
                ))
            else:
                seen_uids.add(rule_uid_str)

        # target
        if not rule.get("target"):
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.RULE_MISSING_TARGET,
                "target missing.",
                "Add a target (asset URI).",
                {"path": path},
            ))

        # action
        if not rule.get("action"):
            issues.append(issue(
                IssueSeverity.CRITICAL, IssueCode.RULE_MISSING_ACTION,
                "action missing.",
                "Add an ODRL action (e.g. odrl:use, odrl:trigger).",
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

        # constraints
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
        code  = IssueCode.REFINEMENT_INCOMPLETE if is_refinement else IssueCode.CONSTRAINT_INCOMPLETE
        label = "refinement" if is_refinement else "constraint"

        for field_name in ("leftOperand", "operator", "rightOperand"):
            if field_name not in obj:
                issues.append(AuditIssue(
                    severity    = IssueSeverity.CRITICAL,
                    layer       = IssueLayer.SYNTAX,
                    code        = code,
                    fragment_id = frag_id,
                    policy_uid  = policy_uid,
                    description = f"{path}: field '{field_name}' missing in {label}.",
                    suggestion  = f"Add '{field_name}' to the {label}.",
                    details     = {"path": path},
                ))
        return issues

    def _get_fork_branches_names(self, gateway_name: str) -> list[str]:
        return [b.to_activity for b in self._get_fork_branches(gateway_name)]

    def _get_fork_conditions(self, gateway_name: str) -> list[str]:
        return [b.condition for b in self._get_fork_branches(gateway_name) if b.condition]

    # ══════════════════════════════════════════════════════════════
    #  Level 4 — Intra-fragment coherence
    # ══════════════════════════════════════════════════════════════

    def _check_intra_consistency(
        self, frag_id: str, fps: FragmentPolicySet
    ) -> list[AuditIssue]:
        issues: list[AuditIssue] = []
        target_rules: dict[str, list[str]] = {}
        local_rules:  dict[str, str]       = {}

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
                                severity    = IssueSeverity.WARNING,
                                layer       = IssueLayer.GLOBAL,
                                code        = IssueCode.DUPLICATE_RULE,
                                fragment_id = frag_id,
                                policy_uid  = policy.get("uid"),
                                description = f"Rule uid '{uid}' duplicated in fragment '{frag_id}'.",
                            ))
                        else:
                            local_rules[uid] = policy.get("uid", "")

        for tgt, rule_types in target_rules.items():
            if "permission" in rule_types and "prohibition" in rule_types:
                issues.append(AuditIssue(
                    severity    = IssueSeverity.CRITICAL,
                    layer       = IssueLayer.GLOBAL,
                    code        = IssueCode.PERM_PROHIB_CONFLICT,
                    fragment_id = frag_id,
                    description = f"Permission/prohibition conflict on target '{tgt}' in '{frag_id}'.",
                ))

        for fpd in fps.fpd_policies:
            if fpd.get("_gateway") != "XOR":
                continue

            conditions = fpd.get("_conditions", [])
            if len(conditions) >= 2 and conditions[0] == conditions[1]:
                issues.append(AuditIssue(
                    severity    = IssueSeverity.WARNING,
                    layer       = IssueLayer.GLOBAL,
                    code        = IssueCode.XOR_SAME_CONDITION,
                    fragment_id = frag_id,
                    policy_uid  = fpd.get("uid"),
                    description = f"FPd XOR: both conditions are identical ('{conditions[0]}').",
                ))

            for perm in fpd.get("permission", []):
                actions = perm.get("action", [])
                if not isinstance(actions, list):
                    actions = [actions]
                for act_obj in actions:
                    if isinstance(act_obj, dict) and not act_obj.get("refinement"):
                        issues.append(AuditIssue(
                            severity    = IssueSeverity.WARNING,
                            layer       = IssueLayer.GLOBAL,
                            code        = IssueCode.XOR_MISSING_CONDITION,
                            fragment_id = frag_id,
                            policy_uid  = fpd.get("uid"),
                            description = (
                                f"FPd XOR: permission '{perm.get('uid', '?')}' "
                                f"has no conditional refinement."
                            ),
                        ))

            if fpd.get("_gateway") in ("AND", "OR"):
                gw = fpd.get("_gateway")
                for rule in fpd.get("obligation", []):
                    tgt = rule.get("target")
                    if isinstance(tgt, str) and tgt not in local_rules:
                        issues.append(AuditIssue(
                            severity    = IssueSeverity.INFO,
                            layer       = IssueLayer.GLOBAL,
                            code        = IssueCode.MISSING_FPA,
                            fragment_id = frag_id,
                            policy_uid  = fpd.get("uid"),
                            description = f"FPd {gw}: obligation target '{tgt}' not in local index.",
                            details     = {"unknown_rule": tgt},
                        ))

        return issues

    # ══════════════════════════════════════════════════════════════
    #  Level 4 — Inter-fragment coherence
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

                if to_frag and to_frag not in self.fp_results:
                    issues.append(AuditIssue(
                        severity    = IssueSeverity.CRITICAL,
                        layer       = IssueLayer.GLOBAL,
                        code        = IssueCode.MESSAGE_NO_RECEIVER,
                        fragment_id = frag_id,
                        policy_uid  = fpd_uid,
                        description = (
                            f"Message flow '{from_act} → {to_act}': "
                            f"target fragment '{to_frag}' does not exist."
                        ),
                        details = {"to_fragment": to_frag},
                    ))

                for perm in fpd.get("permission", []):
                    assignee = perm.get("assignee", "")
                    if assignee and assignee not in self._rule_index:
                        issues.append(AuditIssue(
                            severity    = IssueSeverity.WARNING,
                            layer       = IssueLayer.GLOBAL,
                            code        = IssueCode.MESSAGE_UNKNOWN_RULE,
                            fragment_id = frag_id,
                            policy_uid  = fpd_uid,
                            description = (
                                f"Message flow '{from_act} → {to_act}': "
                                f"assignee '{assignee}' not found in FPa index."
                            ),
                            details = {"assignee": assignee, "to_fragment": to_frag},
                        ))

                    for duty in perm.get("duty", []):
                        tgt = duty.get("target", "")
                        if tgt and tgt not in self._rule_index:
                            issues.append(AuditIssue(
                                severity    = IssueSeverity.WARNING,
                                layer       = IssueLayer.GLOBAL,
                                code        = IssueCode.MESSAGE_UNKNOWN_TARGET,
                                fragment_id = frag_id,
                                policy_uid  = fpd_uid,
                                description = (
                                    f"Message flow '{from_act} → {to_act}': "
                                    f"duty target '{tgt}' not found in FPa index."
                                ),
                                details = {"duty_target": tgt, "to_fragment": to_frag},
                            ))

        issues.extend(self._detect_dependency_cycles())
        return issues

    def _detect_dependency_cycles(self) -> list[AuditIssue]:
        """DFS to detect inter-fragment dependency cycles."""
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
                        severity    = IssueSeverity.CRITICAL,
                        layer       = IssueLayer.GLOBAL,
                        code        = IssueCode.DEPENDENCY_CYCLE,
                        fragment_id = frag_id,
                        description = f"Inter-fragment dependency cycle: {' → '.join(cycle)}",
                        details     = {"cycle": cycle},
                    ))

        return issues

    # ══════════════════════════════════════════════════════════════
    #  Console report
    # ══════════════════════════════════════════════════════════════

    def print_report(self, report: AuditReport) -> None:
        s = report.summary()
        print("\n" + "═" * 65)
        print("  AGENT 5 — Policy Audit Report")
        print("═" * 65)
        print(f"  Status   : {'✅ VALID' if s['is_valid'] else '❌ CRITICAL ISSUES'}")
        print(f"  Critical : {s['critical']}")
        print(f"  Warning  : {s['warning']}")
        print(f"  Info     : {s['info']}")
        print(f"  By layer :")
        for layer, count in s["by_layer"].items():
            print(f"    {layer:14s} : {count}")
        print()
        for issue in report.issues:
            print(f"  {issue}")
            if issue.suggestion:
                print(f"    ↳ Suggestion: {issue.suggestion}")
        print("═" * 65)