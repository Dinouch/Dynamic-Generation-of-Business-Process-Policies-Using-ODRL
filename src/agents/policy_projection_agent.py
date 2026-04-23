"""
policy_projection_agent.py — Agent 4 : Policy Projection Agent

Responsabilité unique :
    Générer les Fragment Policies (FPa et FPd) en JSON-LD ODRL
    conformément à la section 4.2 du rapport technique.

Corrections v2 :
    - FPd XOR/AND/OR générés depuis les PATTERNS du graphe formel
      (les connexions relient gateway→activité, pas activité→activité directement)
    - Méthode export() pour sérialiser en JSON-LD propre (sans clés _xxx)
      et écrire un fichier .jsonld par policy

Couche multi-agent :
    - receive()  : accepte VALIDATION_DONE, SEMANTIC_*, ODRL_SYNTAX_FAILURE, ODRL_VALID
    - Flux      : SYNTAX_AUDIT_REQUEST (A5) → ODRL_VALID → POLICIES_READY (A3, sémantique)
                  → SEMANTIC_VALIDATED → SYNTAX_AUDIT_REQUEST (A5, passe finale)
    - Boucle syntaxique (MAX_SYNTAX_LOOPS côté A5) : ODRL_SYNTAX_FAILURE → SYNTAX_CORRECTION
      (A4→A5) → AGREE (A5) → corrections → POLICIES_READY vers A5
"""

import copy
import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import AzureOpenAI, OpenAI

from .structural_analyzer import (
    AgentMessage,
    EnrichedGraph,
    ConnectionInfo,
    MessageType,
)
from .Agent_3.constraint_validator import ValidationReport
from .exception_handling_agent import UnsupportedCaseProposal


# ─────────────────────────────────────────────
#  Template keys for hint-based ODRL patching (extend without changing core logic)
# ─────────────────────────────────────────────

TEMPLATE_KEY_MAP: dict[str, str] = {
    "constraint_operator": "operator",
    "constraint_left_operand": "leftOperand",
    "constraint_right_operand": "rightOperand",
    "action_id": "action",
    "rule_type": "rule_type",
}


# ─────────────────────────────────────────────
#  Helpers — génération d'URIs
# ─────────────────────────────────────────────

BASE_URI = "http://example.com"

def uri_policy(uid: str) -> str:
    return f"{BASE_URI}/policy:{uid}"

def uri_rule(name: str) -> str:
    return f"{BASE_URI}/rules/{name.replace(' ','_').replace('-','_')}"

def uri_asset(name: str) -> str:
    return f"{BASE_URI}/asset/{name.replace(' ','_').replace('-','_')}"


def uri_collection(name: str) -> str:
    return f"{BASE_URI}/assets/{name.replace(' ','_').replace('-','_')}"

def uri_message(fi: str, fj: str) -> str:
    return f"{BASE_URI}/messages/msg_{fi}_{fj}"

def new_uid() -> str:
    return str(uuid.uuid4())[:8]


# Chaîne "enable" seule ne matérialise pas odrl:action sur le nœud règle après expand→RDF (pyld).
ODRL_ACTION_ENABLE: dict[str, str] = {"@id": "odrl:enable"}


def _coerce_odrl_action_from_hint(action: object) -> object:
    """
    Normalise le hint ``action`` de l'exception handling agent vers une valeur compatible pyld/SHACL.

    Les sorties LLM du type ``odrl:trigger (re-trigger...)`` ne sont pas des IRIs ;
    on retombe sur ``odrl:enable``.
    """
    if action is None:
        return ODRL_ACTION_ENABLE
    if isinstance(action, dict):
        return action if action.get("@id") else ODRL_ACTION_ENABLE
    if isinstance(action, str):
        s = action.strip()
        if not s or s.lower() == "enable":
            return ODRL_ACTION_ENABLE
        if " " in s or "(" in s or "\n" in s or len(s) > 72:
            return ODRL_ACTION_ENABLE
        if s.startswith("odrl:"):
            return {"@id": s}
        if s.startswith("http://") or s.startswith("https://"):
            return {"@id": s}
        return ODRL_ACTION_ENABLE
    return ODRL_ACTION_ENABLE


# ─────────────────────────────────────────────
#  Structures de sortie
# ─────────────────────────────────────────────

@dataclass
class FragmentPolicySet:
    """
    Ensemble complet des policies d'un fragment fi.
    FPij(aij) = < idfi, FPa(aij), [FPd(aij, aik)] >
    """
    fragment_id:  str
    fpa_policies: list[dict] = field(default_factory=list)
    fpd_policies: list[dict] = field(default_factory=list)

    def all_policies(self) -> list[dict]:
        return self.fpa_policies + self.fpd_policies

    def to_odrl(self) -> list[dict]:
        """Retourne les policies en JSON-LD ODRL pur — sans clés internes _xxx."""
        return [{k: v for k, v in p.items() if not k.startswith("_")}
                for p in self.all_policies()]

    def summary(self) -> dict:
        return {
            "fragment_id": self.fragment_id,
            "fpa_count":   len(self.fpa_policies),
            "fpd_count":   len(self.fpd_policies),
            "total":       len(self.all_policies()),
        }


# ─────────────────────────────────────────────
#  Agent 4 — Policy Projection Agent
# ─────────────────────────────────────────────

class PolicyProjectionAgent:
    """
    Agent 4 du pipeline multi-agent.

    Mode standalone :
        Instancier avec enriched_graph + validation_report et appeler generate().

    Mode pipeline :
        Instancier avec enriched_graph=None, validation_report=None.
        Les données arrivent via receive(VALIDATION_DONE).
        Enregistrer le callback bus (A3 / A5 / pipeline) avant le démarrage.

    Enchaînement :
        Après projection : audit syntaxe (A5), puis sémantique (A3), puis audit syntaxe final (A5).

    Boucle syntaxique :
        Agent 5 émet ODRL_SYNTAX_FAILURE ; Agent 4 envoie SYNTAX_CORRECTION, applique des
        patches déterministes, puis ré-émet POLICIES_READY vers A5. MAX_SYNTAX_LOOPS est géré par A5.
    """

    AGENT_NAME = "agent4"
    MODEL = "gpt-4o"
    TEMPERATURE = 0.2

    _UNSUPPORTED_SYSTEM = (
        "You are an ODRL 2.2 expert. You output only valid JSON (no markdown fences). "
        "The JSON must include keys problem_interpretation, business_intent, odrl_policy. "
        "Use strict ODRL vocabulary from https://www.w3.org/TR/odrl-vocab/ only. "
        "Do not use profile extensions."
    )

    def __init__(
        self,
        enriched_graph:    Optional[EnrichedGraph] = None,
        validation_report: Optional[ValidationReport] = None,
        api_key: Optional[str] = None,
        *,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
    ):
        self.enriched_graph    = enriched_graph
        self.validation_report = validation_report
        self._activity_rule_index:   dict[str, str] = {}
        self._activity_policy_index: dict[str, str] = {}

        self._unsupported_fpd_llm_cache: dict[str, dict] = {}
        self._unsupported_cache_lock = threading.Lock()
        self._use_azure = False
        self._deployment: Optional[str] = None
        self.client: Optional[Any] = None

        key_azure = api_key or os.environ.get("AZURE_OPENAI_KEY") or os.environ.get(
            "AZURE_OPENAI_API_KEY"
        )
        endpoint = (azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")).rstrip("/")
        if key_azure and endpoint:
            self._use_azure = True
            self._deployment = (
                azure_deployment
                or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
                or os.environ.get("AZURE_OPENAI_MODEL")
                or "gpt-4o"
            )
            api_ver = azure_api_version or os.environ.get(
                "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
            )
            self.client = AzureOpenAI(
                api_key=key_azure,
                api_version=api_ver,
                azure_endpoint=endpoint,
            )
        else:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if key:
                self.client = OpenAI(api_key=key)

        # ── Couche multi-agent ──────────────────────────────────────
        self._on_send:             Optional[Callable[[AgentMessage], None]] = None
        self._last_fp_results:     Optional[dict] = None
        self._last_enriched_graph: Optional[EnrichedGraph] = None
        # ACL FAILURE (semantic-audit) must reference this AGREE envelope id (see ``acl_failure_reply_target``).
        self._semantic_agree_anchor: Optional[str] = None
        # After ODRL FAILURE from Agent 5: payload until Agent 5 AGREE on our SYNTAX_CORRECTION request.
        self._pending_syntax_correction: Optional[dict[str, Any]] = None
        # "pre_semantic" | "post_semantic" — which syntax pass ODRL_VALID completes (None = terminal forward).
        self._odrl_syntax_stage: Optional[str] = None
        # Optional pipeline status (e.g. partial_template_only) threaded into A5 payloads.
        self._pipeline_content_status: Optional[str] = None

    # ═════════════════════════════════════════
    #  COUCHE MULTI-AGENT
    # ═════════════════════════════════════════

    def register_send_callback(self, fn: Callable[[AgentMessage], None]) -> None:
        """Enregistre le callback d'envoi (typiquement agent5.receive)."""
        self._on_send = fn

    def send(self, msg: AgentMessage) -> None:
        """Émet un message via le callback enregistré."""
        print(f"[Agent 4] ► SEND {msg}")
        if self._on_send:
            self._on_send(msg)
        else:
            print(f"[Agent 4][WARN] Aucun callback — message {msg.msg_type.value} non transmis.")

    def receive(self, msg: AgentMessage) -> None:
        """
        Point d'entrée des messages entrants.

        Messages acceptés :
          - VALIDATION_DONE     : rapport Agent 3 → projection → SYNTAX_AUDIT_REQUEST vers Agent 5
          - SEMANTIC_CORRECTION : correctifs sémantiques depuis Agent 3
          - SEMANTIC_VALIDATED  : après sémantique → SYNTAX_AUDIT_REQUEST vers Agent 5 (passe finale)
          - ODRL_VALID / ODRL_SYNTAX_ERROR : depuis Agent 5 — routage sémantique ou fin pipeline
          - SYNTAX_CORRECTION   : corrections depuis Agent 5 (syntaxe)

        Tout autre type est ignoré avec un warning.
        """
        print(f"[Agent 4] ◄ RECEIVE {msg}")
        if (
            msg.sender == "agent5"
            and msg.msg_type == MessageType.GRAPH_READY
            and (msg.payload or {}).get("_acl_ontology") == "odrl-syntax-audit"
        ):
            # Si l'adaptateur ACL a encore mappé un FIPA « agree » en graph_ready, on rattrape.
            print(
                "[Agent 4][INFO] Rattrapage : graph_ready + odrl-syntax-audit depuis Agent 5 → DELEGATION_AGREE."
            )
            self.receive(
                AgentMessage(
                    sender=msg.sender,
                    recipient=msg.recipient,
                    msg_type=MessageType.DELEGATION_AGREE,
                    payload=dict(msg.payload or {}),
                    loop_turn=msg.loop_turn,
                )
            )
            return
        if msg.msg_type == MessageType.DELEGATION_AGREE:
            if (
                msg.sender == "agent5"
                and (msg.payload or {}).get("_acl_ontology") == "odrl-syntax-audit"
                and self._pending_syntax_correction is not None
            ):
                pending = self._pending_syntax_correction
                self._pending_syntax_correction = None
                self._run_syntax_repair_after_a5_agree(pending)
                return
            print("[Agent 4] Received delegation AGREE (informational).")
            return

        if msg.msg_type == MessageType.VALIDATION_DONE:
            self._agree_semantic_delegation(
                msg,
                "projection_request_id",
                "I will project fragment policies from the validation report for semantic audit.",
            )
            validation_report = msg.payload["validation_report"]
            enriched_graph = msg.payload["enriched_graph"]
            self._last_enriched_graph = enriched_graph
            self._pipeline_content_status = (msg.payload or {}).get("status")
            self._project_and_send(enriched_graph, validation_report, loop_turn=msg.loop_turn)

        elif msg.msg_type == MessageType.SEMANTIC_CORRECTION:
            self._agree_semantic_delegation(
                msg,
                "semantic_correction_request_id",
                "I will apply the semantic correction hints and resend projected policies.",
            )
            self._handle_semantic_correction(msg)

        elif msg.msg_type == MessageType.SEMANTIC_VALIDATED:
            self._handle_semantic_validated(msg)

        elif msg.msg_type == MessageType.SYNTAX_CORRECTION:
            # Legacy: Agent 5 used to push SYNTAX_CORRECTION; flow is now FAILURE → A4 REQUEST → A5 AGREE.
            if msg.sender == "agent5":
                print("[Agent 4][WARN] SYNTAX_CORRECTION from Agent 5 (legacy) — applying without AGREE handshake.")
                self._handle_syntax_correction(msg)
            else:
                print(f"[Agent 4][WARN] SYNTAX_CORRECTION from {msg.sender} — ignoré.")

        elif msg.msg_type == MessageType.ODRL_SYNTAX_FAILURE:
            print(
                f"[Agent 4] ODRL syntax FAILURE from Agent 5: "
                f"{(msg.payload or {}).get('reason', '')}"
            )
            self._pending_syntax_correction = {
                "affected_policies": list((msg.payload or {}).get("affected_policies") or []),
                "errors": list((msg.payload or {}).get("errors") or []),
                "loop_turn": int(msg.loop_turn or 0),
            }
            self.send(
                AgentMessage(
                    sender=self.AGENT_NAME,
                    recipient="agent5",
                    msg_type=MessageType.SYNTAX_CORRECTION,
                    payload={
                        "affected_policies": self._pending_syntax_correction["affected_policies"],
                        "errors": self._pending_syntax_correction["errors"],
                        "utterance": (
                            "Requesting syntax repair round: apply deterministic fixes, "
                            "then resubmit policies for audit."
                        ),
                    },
                    loop_turn=msg.loop_turn,
                )
            )

        elif msg.msg_type in (MessageType.ODRL_VALID, MessageType.ODRL_SYNTAX_ERROR):
            if self._odrl_syntax_stage == "pre_semantic":
                if msg.msg_type == MessageType.ODRL_SYNTAX_ERROR:
                    self._odrl_syntax_stage = None
                    self._forward_final_audit_to_pipeline(msg)
                    return
                self._odrl_syntax_stage = None
                self._send_policies_ready_for_semantic(loop_turn=msg.loop_turn)
                return
            if self._odrl_syntax_stage == "post_semantic":
                self._odrl_syntax_stage = None
            self._forward_final_audit_to_pipeline(msg)

        elif msg.msg_type == MessageType.SEMANTIC_VALIDATION_FAILURE:
            print(
                "[Agent 4] Semantic FAILURE from Agent 3: "
                f"{(msg.payload or {}).get('reason', '')}"
            )

        else:
            print(f"[Agent 4][WARN] Message '{msg.msg_type.value}' non géré — ignoré.")

    def _agree_semantic_delegation(self, msg: AgentMessage, id_key: str, utterance: str) -> None:
        rid = msg.payload.get(id_key)
        if not rid:
            return
        agree_rw = uuid.uuid4().hex
        self._semantic_agree_anchor = agree_rw
        self.send(
            AgentMessage(
                sender=self.AGENT_NAME,
                recipient="agent3",
                msg_type=MessageType.DELEGATION_AGREE,
                payload={
                    "utterance": utterance,
                    "_acl_ontology": "semantic-audit",
                    "acl_in_reply_to": rid,
                    "_acl_reply_with": agree_rw,
                },
                loop_turn=msg.loop_turn,
            )
        )

    def _forward_final_audit_to_pipeline(self, msg: AgentMessage) -> None:
        """Agent 5 completes ODRL syntax audit; Agent 4 forwards the terminal result to the orchestrator."""
        self.send(
            AgentMessage(
                sender=self.AGENT_NAME,
                recipient="pipeline",
                msg_type=msg.msg_type,
                payload=dict(msg.payload or {}),
                loop_turn=msg.loop_turn,
            )
        )

    # ─────────────────────────────────────────
    #  Handlers internes
    # ─────────────────────────────────────────

    def _project_and_send(
        self,
        enriched_graph:    EnrichedGraph,
        validation_report: ValidationReport,
        loop_turn:         int,
    ) -> None:
        """
        Génère les Fragment Policies puis lance l'audit syntaxique (Agent 5) avant la sémantique.
        """
        fp_results = self.project(enriched_graph, validation_report)
        self._last_fp_results = fp_results
        self._odrl_syntax_stage = "pre_semantic"
        self._emit_syntax_audit_request(loop_turn)

    def _emit_syntax_audit_request(self, loop_turn: int) -> None:
        """Demande d'audit ODRL (syntaxe + cohérence globale) vers Agent 5."""
        pl: dict[str, Any] = {
            "fp_results": self._last_fp_results,
            "enriched_graph": self._last_enriched_graph,
            "utterance": (
                "Validate ODRL syntax and global coherence for these projected fragment policies."
            ),
        }
        if self._pipeline_content_status:
            pl["status"] = self._pipeline_content_status
        self.send(
            AgentMessage(
                sender=self.AGENT_NAME,
                recipient="agent5",
                msg_type=MessageType.SYNTAX_AUDIT_REQUEST,
                payload=pl,
                loop_turn=loop_turn,
            )
        )

    def _send_policies_ready_for_semantic(self, loop_turn: int) -> None:
        """Après une première passe syntaxique OK — validation sémantique (Agent 3)."""
        pl: dict[str, Any] = {
            "fp_results": self._last_fp_results,
            "enriched_graph": self._last_enriched_graph,
        }
        if self._semantic_agree_anchor:
            pl["acl_failure_reply_target"] = self._semantic_agree_anchor
        if self._pipeline_content_status:
            pl["status"] = self._pipeline_content_status
        self.send(
            AgentMessage(
                sender=self.AGENT_NAME,
                recipient="agent3",
                msg_type=MessageType.POLICIES_READY,
                payload=pl,
                loop_turn=loop_turn,
            )
        )

    def _run_syntax_repair_after_a5_agree(self, pending: dict[str, Any]) -> None:
        """After Agent 5 AGREE on our SYNTAX_CORRECTION request, apply fixes and resubmit."""
        affected_policies = list(pending.get("affected_policies") or [])
        errors = list(pending.get("errors") or [])
        loop_turn = int(pending.get("loop_turn", 0) or 0)
        print(
            f"[Agent 4] SYNTAX repair (post A5 AGREE) — "
            f"{len(affected_policies)} policy(ies), {len(errors)} erreur(s)"
        )
        self._apply_syntax_corrections(affected_policies, errors)
        self.send(
            AgentMessage(
                sender=self.AGENT_NAME,
                recipient="agent5",
                msg_type=MessageType.POLICIES_READY,
                payload={
                    "fp_results": self._last_fp_results,
                    "enriched_graph": self._last_enriched_graph,
                },
                loop_turn=loop_turn + 1,
            )
        )

    def _handle_syntax_correction(self, msg: AgentMessage) -> None:
        """
        Legacy path: correction payload from Agent 5 (deprecated — prefer FAILURE + A4 REQUEST + A5 AGREE).
        """
        affected_policies = msg.payload.get("affected_policies", [])
        errors = msg.payload.get("errors", [])
        print(
            f"[Agent 4] SYNTAX_CORRECTION (legacy) — "
            f"{len(affected_policies)} policy(ies), {len(errors)} erreur(s)"
        )
        self._apply_syntax_corrections(affected_policies, errors)
        self.send(
            AgentMessage(
                sender=self.AGENT_NAME,
                recipient="agent5",
                msg_type=MessageType.POLICIES_READY,
                payload={
                    "fp_results": self._last_fp_results,
                    "enriched_graph": self._last_enriched_graph,
                },
                loop_turn=msg.loop_turn + 1,
            )
        )

    def _apply_syntax_corrections(
        self,
        affected_policies: list[str],
        errors:            list[dict],
    ) -> None:
        """
        Applique des corrections déterministes JSON-LD sur les policies affectées.
        Pas de LLM — uniquement des patches structurels sur les champs ODRL.

        Codes d'erreur supportés :
          MISSING_CONTEXT  → ajouter "@context": "http://www.w3.org/ns/odrl.jsonld"
          MISSING_TYPE     → ajouter "@type": "Agreement"
          MALFORMED_URI    → extraire une IRI http(s) depuis ``suggestion`` (regex) ;
                             sinon préfixer ``uid`` avec BASE_URI si relatif
          RULE_MISSING_UID → générer un uid via new_uid() pour la règle concernée
          MISSING_UID      → générer un uid de policy via new_uid()

        Les corrections ne touchent qu'aux champs structurels JSON-LD —
        jamais à la logique métier de la policy (règles, actions, contraintes).
        """
        if self._last_fp_results is None:
            print("[Agent 4][WARN] _last_fp_results est None — aucune correction possible.")
            return

        affected_set = set(affected_policies)

        for fragment_id, fps in self._last_fp_results.items():
            for policy in fps.all_policies():
                policy_uid = policy.get("uid", "")
                if policy_uid not in affected_set:
                    continue

                for error in errors:
                    if error.get("policy_uid") != policy_uid:
                        continue

                    code = error.get("code", "")
                    path = error.get("path", "")

                    if code == "MISSING_CONTEXT":
                        if "@context" not in policy:
                            policy["@context"] = "http://www.w3.org/ns/odrl.jsonld"
                            print(f"[Agent 4] MISSING_CONTEXT corrigé : {policy_uid}")

                    elif code == "MISSING_TYPE":
                        if "@type" not in policy:
                            policy["@type"] = "Agreement"
                            print(f"[Agent 4] MISSING_TYPE corrigé : {policy_uid}")

                    elif code == "MALFORMED_URI":
                        import re

                        suggestion = error.get("suggestion") or ""
                        match = re.search(r'https?://[^\s"\'\.>]+', suggestion)
                        if match:
                            policy["uid"] = match.group(0)
                            print(
                                f"[Agent 4] MALFORMED_URI corrigé : {suggestion[:60]} → {policy['uid']}"
                            )
                        else:
                            uid = policy.get("uid", "")
                            if uid and not uid.startswith("http"):
                                policy["uid"] = f"{BASE_URI}/{uid}"
                                print(
                                    f"[Agent 4] MALFORMED_URI corrigé (fallback) : {uid} → {policy['uid']}"
                                )

                    elif code == "MISSING_UID":
                        if not policy.get("uid"):
                            policy["uid"] = uri_policy(f"FP_{new_uid()}")
                            print(f"[Agent 4] MISSING_UID corrigé : {policy['uid']}")

                    elif code == "RULE_MISSING_UID":
                        # path indique le type de règle ("permission", "obligation", etc.)
                        rule_type = path or "permission"
                        rules = policy.get(rule_type, [])
                        for rule in rules:
                            if isinstance(rule, dict) and not rule.get("uid"):
                                ru = uri_rule(f"rule_{new_uid()}")
                                rule["uid"] = ru
                                print(
                                    f"[Agent 4] RULE_MISSING_UID corrigé dans "
                                    f"'{rule_type}' de {policy_uid}"
                                )

                    else:
                        print(
                            f"[Agent 4][WARN] Code d'erreur inconnu '{code}' "
                            f"pour {policy_uid} — ignoré."
                        )

    def _handle_semantic_correction(self, msg: AgentMessage) -> None:
        """
        Apply semantic hints from Agent 3 to policies in ``_last_fp_results``,
        then re-emit ``POLICIES_READY`` to Agent 3.

        Parameters
        ----------
        msg
            Payload includes ``semantic_hints`` (list of dicts) and ``fp_results``.
        """
        hints = msg.payload.get("semantic_hints") or []
        fp_results = msg.payload.get("fp_results") or self._last_fp_results
        if fp_results is not None:
            self._last_fp_results = fp_results
        eg = msg.payload.get("enriched_graph")
        if eg is not None:
            self._last_enriched_graph = eg

        print(f"[Agent 4] SEMANTIC_CORRECTION — {len(hints)} hint(s)")

        for h in hints:
            uid = h.get("policy_uid")
            if not uid or not self._last_fp_results:
                continue
            policy = self._find_policy_by_uid(uid)
            if policy is None:
                print(f"[Agent 4][WARN] Policy uid {uid} not found for semantic patch.")
                continue
            path = h.get("field_path", "")
            if not path:
                print(f"[Agent 4][WARN] Empty field_path for policy {uid}.")
                continue
            self._set_field_path(policy, path, h.get("suggested_fix", ""))

        pl2: dict[str, Any] = {
            "fp_results": self._last_fp_results,
            "enriched_graph": self._last_enriched_graph,
        }
        if self._semantic_agree_anchor:
            pl2["acl_failure_reply_target"] = self._semantic_agree_anchor
        self.send(
            AgentMessage(
                sender=self.AGENT_NAME,
                recipient="agent3",
                msg_type=MessageType.POLICIES_READY,
                payload=pl2,
                loop_turn=msg.loop_turn + 1,
            )
        )

    def _handle_semantic_validated(self, msg: AgentMessage) -> None:
        """
        Après validation sémantique — repasse syntaxe / cohérence (Agent 5) pour prendre en
        compte d'éventuelles modifications structurelles des correctifs sémantiques.
        """
        fp_results = msg.payload.get("fp_results")
        if fp_results is not None:
            self._last_fp_results = fp_results
        eg = msg.payload.get("enriched_graph")
        if eg is not None:
            self._last_enriched_graph = eg
        status = msg.payload.get("status")
        if status:
            self._pipeline_content_status = status

        self._odrl_syntax_stage = "post_semantic"
        self._emit_syntax_audit_request(msg.loop_turn)

    def _find_policy_by_uid(self, policy_uid: str) -> Optional[dict]:
        """Locate a policy dict by its ``uid`` across all fragment results."""
        if not self._last_fp_results:
            return None
        for fps in self._last_fp_results.values():
            for pol in fps.all_policies():
                if pol.get("uid") == policy_uid:
                    return pol
        return None

    def _set_field_path(self, policy: dict, path: str, fix: str) -> None:
        """
        Navigate ``path`` (e.g. permission[0].constraint[0].operator) and assign
        a value derived from ``suggested_fix`` text (best-effort).
        """
        parts = path.split(".")
        cur: Any = policy
        for seg in parts[:-1]:
            if "[" in seg:
                name, rest = seg.split("[", 1)
                idx = int(rest.split("]")[0])
                cur = cur[name][idx]
            else:
                cur = cur[seg]
        last = parts[-1]
        last_name = last.split("[", 1)[0] if "[" in last else last
        val = self._coerce_semantic_fix(fix, field_name=last_name)
        if "[" in last:
            name, rest = last.split("[", 1)
            idx = int(rest.split("]")[0])
            cur[name][idx] = val
        else:
            cur[last] = val

    def _coerce_semantic_fix(self, fix: str, field_name: str = "") -> str:
        """
        Map common English fix phrases to ODRL operator tokens.
        For ``uid`` / targets, extract an http(s) IRI from LLM prose (\"set uid to ...\").
        """
        fix = fix.strip().strip('"')
        if field_name == "uid" or (field_name and "uid" in field_name.lower()):
            m = re.search(r"https?://[^\s\"'<>\\\]]+", fix)
            if m:
                return m.group(0).rstrip(".,;)")
        m = re.search(r"https?://[^\s\"'<>\\\]]+", fix)
        if m and field_name in ("uid", "target", "assignee", "assigner"):
            return m.group(0).rstrip(".,;)")
        low = fix.lower()
        for token in ("gteq", "lteq", "neq", "eq", "gt", "lt"):
            if token in low:
                return token
        return fix


    def _parse_llm_json_object(self, raw: str) -> dict:
        """Extrait un objet JSON depuis une réponse LLM (éventuellement entourée de fences)."""
        text = raw.strip()
        if "```" in text:
            for chunk in text.split("```"):
                chunk = chunk.strip()
                if chunk.lower().startswith("json"):
                    chunk = chunk[4:].strip()
                if chunk.startswith("{") and chunk.endswith("}"):
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
        return json.loads(text)

    def _unsupported_structural_context(self, proposal: UnsupportedCaseProposal) -> Optional[dict]:
        """Contexte Agent 1 (UnsupportedPattern) pour enrichir le prompt — même flux pour tous les types."""
        eg = self.enriched_graph
        if not eg:
            return None
        patterns = getattr(eg, "unsupported_patterns", None) or []
        for up in patterns:
            if up.pattern_type != proposal.pattern_type:
                continue
            if proposal.fragment_id and up.fragment_id != proposal.fragment_id:
                continue
            if proposal.gateway_name and up.gateway_name != proposal.gateway_name:
                continue
            return {
                "structural_description": up.description,
                "bpmn_semantic": up.bpmn_semantic,
                "involved_fragment_ids": list(up.involved_fragment_ids or []),
                "involved_activity_names": list(up.involved_activity_names or []),
            }
        return None

    def _call_llm(self, system: str, user: str) -> str:
        if not self.client:
            raise RuntimeError("LLM client not configured for Agent 4")
        model = self._deployment if self._use_azure else self.MODEL
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": self.TEMPERATURE,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if not self._use_azure:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Empty LLM response (Agent 4 unsupported)")
        return content

    def _normalize_odrl_actions_in_policy(self, policy: dict) -> None:
        """Assure des actions ODRL expansibles (pyld/SHACL) sur permission/prohibition/obligation."""
        for key in ("permission", "prohibition", "obligation"):
            rules = policy.get(key)
            if not isinstance(rules, list):
                continue
            for rule in rules:
                if isinstance(rule, dict) and "action" in rule:
                    rule["action"] = _coerce_odrl_action_from_hint(rule.get("action"))

    def _llm_synthesize_unsupported_policy(self, proposal: UnsupportedCaseProposal) -> Optional[dict]:
        """
        Interprète le cas, déduit l'intention métier et produit un document ODRL JSON-LD
        (même chemin pour tous les pattern_type, sans branche spéciale).
        """
        structural = self._unsupported_structural_context(proposal)
        payload = {
            "proposal": proposal.to_dict(),
            "bpmn_structural_context": structural,
            "base_uri": BASE_URI,
            "requested_rule_type": proposal.odrl_rule_type,
        }
        user_prompt = f"""You assist in BPMN-to-ODRL translation for a validated unsupported pattern.

Work through these steps internally, then output JSON only:
1) INTERPRET the structural problem (cycles, gateways, activities, fragments, hints from the exception handling agent).
2) DEDUCE a clear business intent (one or two sentences) that a sensible ODRL policy should express.
3) GENERATE one ODRL 2.2 JSON-LD policy object in field "odrl_policy".

Input (JSON):
{json.dumps(payload, ensure_ascii=False, indent=2)}

Requirements for "odrl_policy":
- "@context": "http://www.w3.org/ns/odrl.jsonld"
- "@type": "Set" or "Agreement"
- "uid": a full IRI under {BASE_URI}/policy:...
- Include exactly one of permission, prohibition, obligation matching requested_rule_type "{proposal.odrl_rule_type}"
- Each rule: "uid" (IRI), "target" (IRI asset), "action" must be a strict ODRL vocabulary action (no custom action, no profile extension)
- Prefer concrete assets derived from activity/gateway names; avoid nonsense policies — if the pattern is incoherent, still emit a minimal prohibition or obligation that documents "no valid operation" rather than fake permissions.
- Never set "profile" in output policies.

Respond with JSON only:
{{
  "problem_interpretation": "string",
  "business_intent": "string",
  "odrl_policy": {{ ... }}
}}
"""
        raw = self._call_llm(self._UNSUPPORTED_SYSTEM, user_prompt)
        data = self._parse_llm_json_object(raw)
        interp = data.get("problem_interpretation", "")
        intent = data.get("business_intent", "")
        if interp or intent:
            print(
                f"[Agent 4] Unsupported LLM — interprétation: {interp[:160]}{'…' if len(interp) > 160 else ''}"
            )
            print(
                f"[Agent 4] Unsupported LLM — intention métier: {intent[:160]}{'…' if len(intent) > 160 else ''}"
            )
        odrl = data.get("odrl_policy")
        if not isinstance(odrl, dict):
            print("[Agent 4][WARN] LLM response missing odrl_policy object")
            return None
        if not odrl.get("uid"):
            odrl["uid"] = uri_policy(f"FPd_UNH_{proposal.pattern_type}_{new_uid()}")
        if "@context" not in odrl:
            odrl["@context"] = "http://www.w3.org/ns/odrl.jsonld"
        self._normalize_odrl_actions_in_policy(odrl)
        return odrl

    def _fallback_minimal_unsupported_fpd(
        self,
        proposal: UnsupportedCaseProposal,
        target_fragment_id: Optional[str],
    ) -> dict:
        """Repli sans LLM : policy minimale cohérente, sans logique spécifique à un pattern."""
        frag = target_fragment_id or proposal.fragment_id or "fragment"
        uid = uri_policy(f"FPd_UNH_{proposal.pattern_type}_{frag.replace('-', '_')}_{new_uid()}")
        rule_uid = uri_rule(f"rule_unh_{new_uid()}")
        rt = proposal.odrl_rule_type if proposal.odrl_rule_type in (
            "permission",
            "prohibition",
            "obligation",
        ) else "permission"
        tgt = uri_asset(proposal.gateway_name or frag or "asset")
        rule_body: dict = {
            "uid": rule_uid,
            "target": tgt,
            "action": ODRL_ACTION_ENABLE,
        }
        out: dict = {
            "@context": "http://www.w3.org/ns/odrl.jsonld",
            "uid": uid,
            "@type": "Set",
            "_fragment_id": frag,
            "_type": "FPd",
            "_unsupported_pattern": proposal.pattern_type,
            "_gateway_name": proposal.gateway_name,
            "_hint_text": proposal.hint_text,
        }
        if proposal.hint_text:
            out["dct:description"] = proposal.hint_text
        out[rt] = [rule_body]
        return out

    def _generate_fpd_from_unsupported_proposal(
        self,
        proposal: UnsupportedCaseProposal,
        target_fragment_id: Optional[str] = None,
    ) -> Optional[dict]:
        """
        FPd depuis une proposition unsupported validée : synthèse LLM (interprétation → intention → ODRL),
        avec cache par proposition pour les fragments multiples.
        """
        cache_key = json.dumps(proposal.to_dict(), sort_keys=True)
        with self._unsupported_cache_lock:
            if cache_key not in self._unsupported_fpd_llm_cache:
                body: Optional[dict] = None
                if self.client:
                    try:
                        body = self._llm_synthesize_unsupported_policy(proposal)
                    except Exception as e:
                        print(f"[Agent 4][WARN] LLM unsupported synthesis failed: {e}")
                if body is None:
                    print(
                        "[Agent 4] Repli déterministe minimal pour unsupported "
                        f"({proposal.pattern_type}, fragment cible "
                        f"{target_fragment_id or proposal.fragment_id})"
                    )
                    body = self._fallback_minimal_unsupported_fpd(proposal, target_fragment_id)
                self._unsupported_fpd_llm_cache[cache_key] = body

            base = self._unsupported_fpd_llm_cache[cache_key]
        out = copy.deepcopy(base)
        frag = target_fragment_id or proposal.fragment_id or "fragment"
        out["_fragment_id"] = frag
        out["_type"] = "FPd"
        out["_unsupported_pattern"] = proposal.pattern_type
        out["_gateway_name"] = proposal.gateway_name
        out["_hint_text"] = proposal.hint_text
        return out

    # ═════════════════════════════════════════
    #  LOGIQUE MÉTIER (inchangée)
    # ═════════════════════════════════════════

    def project(
        self,
        enriched_graph:    Optional[EnrichedGraph] = None,
        validation_report: Optional[ValidationReport] = None,
    ) -> dict[str, "FragmentPolicySet"]:
        """
        Point d'entrée unifié : génère les Fragment Policies pour tous les fragments.

        Peut être appelé :
          - En mode standalone : enriched_graph et validation_report passés en argument
            ou stockés dans self depuis le constructeur.
          - Depuis _project_and_send() : arguments explicites fournis par le message.
        """
        graph  = enriched_graph    or self.enriched_graph
        report = validation_report or self.validation_report

        if graph is None:
            raise ValueError("[Agent 4] Aucun enriched_graph disponible pour la projection.")

        # Mettre à jour les références internes pour les méthodes métier
        self.enriched_graph    = graph
        self.validation_report = report

        return self.generate()

    def generate_unsupported_only(
        self,
        enriched_graph: EnrichedGraph,
        validation_report: ValidationReport,
    ) -> dict[str, "FragmentPolicySet"]:
        """
        Generate ONLY FPd policies derived from accepted unsupported proposals.

        This supports the additive merge strategy:
        - keep already-generated template policies (FPa + FPd templates) untouched
        - append the FPd unsupported policies produced here.
        """
        self.enriched_graph = enriched_graph
        self.validation_report = validation_report

        # Pre-index activities to reuse stable rule/policy identifiers for cross-links.
        self._preindex_all_activities()

        results: dict[str, FragmentPolicySet] = {
            frag_id: FragmentPolicySet(fragment_id=frag_id)
            for frag_id in self.enriched_graph.fragment_contexts.keys()
        }

        if not self.validation_report:
            return results

        accepted = getattr(self.validation_report, "accepted_unsupported_proposals", []) or []
        for raw in accepted:
            prop = UnsupportedCaseProposal.from_dict(raw) if isinstance(raw, dict) else raw
            targets = list(prop.involved_fragment_ids or [])
            if not targets:
                targets = [prop.fragment_id] if prop.fragment_id else []
            for frag_id in targets:
                if frag_id not in results:
                    continue
                fpd = self._generate_fpd_from_unsupported_proposal(prop, target_fragment_id=frag_id)
                if fpd:
                    results[frag_id].fpd_policies.append(fpd)

        return results

    # ─────────────────────────────────────────
    #  Point d'entrée — génération (séquentielle)
    # ─────────────────────────────────────────

    def generate(self) -> dict[str, "FragmentPolicySet"]:
        """Génération séquentielle (comportement historique)."""
        print("[Agent 4] Policy Projection Agent — démarrage de la génération")

        self._unsupported_fpd_llm_cache = {}

        # Pré-indexer TOUTES les activités avant de générer les FPd inter-fragments
        self._preindex_all_activities()

        results: dict[str, FragmentPolicySet] = {}
        for fragment_id in self.enriched_graph.fragment_contexts.keys():
            fps = self._generate_fragment(fragment_id)
            results[fragment_id] = fps
        return results

    # ─────────────────────────────────────────
    #  Point d'entrée — génération (parallèle)
    # ─────────────────────────────────────────

    def generate_parallel(self, max_workers: Optional[int] = None) -> dict[str, "FragmentPolicySet"]:
        """
        Génère les policies "par fragment" en parallèle (ThreadPool).

        Notes :
        - On pré-indexe d'abord toutes les activités (global) pour que les FPd
          inter-fragments (message) puissent résoudre ruleix/rulejy.
        - Ensuite chaque fragment est généré indépendamment.
        """
        print("[Agent 4] Policy Projection Agent — génération parallèle")
        self._unsupported_fpd_llm_cache = {}
        self._preindex_all_activities()

        frag_ids = list(self.enriched_graph.fragment_contexts.keys())
        results: dict[str, FragmentPolicySet] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self._generate_fragment, frag_id): frag_id for frag_id in frag_ids}
            for fut in as_completed(futures):
                frag_id = futures[fut]
                fps = fut.result()
                results[frag_id] = fps

        return {fid: results[fid] for fid in frag_ids if fid in results}

    # ─────────────────────────────────────────
    #  Génération par fragment (thread-safe)
    # ─────────────────────────────────────────

    def _generate_fragment(self, fragment_id: str) -> FragmentPolicySet:
        context = self.enriched_graph.fragment_contexts[fragment_id]
        print(f"[Agent 4] Génération des policies pour '{fragment_id}'...")

        fps = FragmentPolicySet(fragment_id=fragment_id)

        # Étape 1 — FPa pour chaque activité (pas d'écriture dans index global)
        for activity_name in context.activities:
            fpa = self._generate_fpa(activity_name, fragment_id)
            if fpa:
                fps.fpa_policies.append(fpa)

        # Étape 2a — FPd gateway XOR/AND/OR depuis les PATTERNS du graphe
        for pattern in self.enriched_graph.patterns:
            if pattern.fragment_id != fragment_id:
                continue
            if pattern.pattern_type not in ("fork_xor", "fork_and", "fork_or"):
                continue
            fps.fpd_policies.extend(self._fpd_from_pattern(pattern, fragment_id))

        # Étape 2b — FPd séquences internes
        for conn in context.connections:
            if conn.connection_type.lower() == "sequence" and not conn.is_inter:
                fpd = self._fpd_flow_sequence(conn, fragment_id)
                if fpd:
                    fps.fpd_policies.append(fpd)

        # Étape 2c — FPd message inter-fragment (downstream)
        for conn in context.downstream_deps:
            fpd = self._generate_fpd_message(conn, fragment_id)
            if fpd:
                fps.fpd_policies.append(fpd)

        # Étape 3 — FPd depuis propositions « unsupported » validées (Agent 3)
        if self.validation_report:
            for prop in getattr(
                self.validation_report, "accepted_unsupported_proposals", []
            ) or []:
                if isinstance(prop, dict):
                    prop = UnsupportedCaseProposal.from_dict(prop)
                targets = list(prop.involved_fragment_ids or [])
                if not targets:
                    targets = [prop.fragment_id] if prop.fragment_id else []
                if fragment_id not in targets:
                    continue
                fpd = self._generate_fpd_from_unsupported_proposal(
                    prop, target_fragment_id=fragment_id
                )
                if fpd:
                    fps.fpd_policies.append(fpd)

        s = fps.summary()
        print(f"[Agent 4] '{fragment_id}' : "
              f"{s['fpa_count']} FPa + {s['fpd_count']} FPd = {s['total']} policies")

        return fps

    # ─────────────────────────────────────────
    #  Export JSON-LD ODRL
    # ─────────────────────────────────────────

    def export(
        self,
        fp_results: dict[str, FragmentPolicySet],
        output_dir: str = "./odrl_policies",
    ) -> dict[str, list[str]]:
        """
        Sérialise chaque policy en JSON-LD ODRL valide.
        Un fichier .jsonld par policy sous ``output_dir/{fragment_id}/`` (sans sous-dossiers FPa/FPd).
        Le préfixe du nom de fichier (FPa_, FPd_message_, etc.) permet de distinguer les familles.
        Retourne dict[fragment_id → liste des chemins fichiers].
        """
        os.makedirs(output_dir, exist_ok=True)
        exported: dict[str, list[str]] = {}

        for fragment_id, fps in fp_results.items():
            frag_dir = os.path.join(output_dir, fragment_id)
            os.makedirs(frag_dir, exist_ok=True)
            exported[fragment_id] = []

            for i, policy in enumerate(fps.all_policies()):
                odrl = {k: v for k, v in policy.items() if not k.startswith("_")}

                ptype = policy.get("_type", "FP")
                subtype = (
                    policy.get("_gateway")
                    or policy.get("_flow")
                    or policy.get("_dep_type")
                    or policy.get("_unsupported_pattern")
                    or str(i)
                )
                activity = (
                    policy.get("_activity")
                    or "_".join(policy.get("_activities", [])[:2])  # message flow: from+to
                    or policy.get("_gateway_name", "policy")
                )
                uid = str(odrl.get("uid") or odrl.get("@id") or f"idx_{i}")
                uid_short = hashlib.sha1(uid.encode("utf-8")).hexdigest()[:8]
                filename = (
                    f"{ptype}_{subtype}_{activity.replace('-', '_')}_{uid_short}.jsonld"
                )
                filepath = os.path.join(frag_dir, filename)
                # Garder une sécurité en cas de collision cryptographique (très improbable).
                counter = 1
                base_path = filepath
                while os.path.exists(filepath):
                    filepath = base_path.replace(".jsonld", f"__{counter}.jsonld")
                    counter += 1

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(odrl, f, indent=2, ensure_ascii=False)
                exported[fragment_id].append(filepath)

        total = sum(len(v) for v in exported.values())
        print(f"[Agent 4] Export terminé : {total} fichiers .jsonld → '{output_dir}'")
        return exported

    # ─────────────────────────────────────────
    #  Pré-indexation globale
    # ─────────────────────────────────────────

    def _preindex_all_activities(self) -> None:
        """
        Pré-indexe rule_id et policy_uid pour TOUTES les activités.
        Nécessaire pour les FPd inter-fragments où l'activité cible
        est dans un autre fragment que la source.
        """
        for mapping in self.enriched_graph.b2p_mappings.values():
            act = mapping.activity_name
            if act in self._activity_rule_index:
                continue

            b2p = None
            if mapping.b2p_policy_ids:
                b2p = self._pick_best_b2p_for_activity(act, mapping.b2p_policy_ids)

            if b2p:
                rid = self._pick_main_rule_id_for_activity(b2p, act)
                self._activity_rule_index[act] = rid
                self._activity_policy_index[act] = b2p.get("uid", "")
            else:
                self._activity_rule_index[act]   = uri_rule(f"rule_{act.replace('-','_')}")
                self._activity_policy_index[act] = uri_policy(f"FPa_{act.replace('-','_')}")

    # ─────────────────────────────────────────
    #  FPa — Listing 4
    # ─────────────────────────────────────────

    def _generate_fpa(self, activity_name: str, fragment_id: str) -> Optional[dict]:
        mapping = next(
            (m for m in self.enriched_graph.b2p_mappings.values()
             if m.activity_name == activity_name), None
        )
        policy_uid = uri_policy(f"FPa_{activity_name.replace('-','_')}_{new_uid()}")

        if mapping and mapping.b2p_policy_ids:
            b2p = self._pick_best_b2p_for_activity(activity_name, mapping.b2p_policy_ids)
            if b2p:
                fpa = {
                    "@context":     "http://www.w3.org/ns/odrl.jsonld",
                    "uid":          policy_uid,
                    "@type":        b2p.get("@type", "Set"),
                    "_fragment_id": fragment_id,
                    "_activity":    activity_name,
                    "_type":        "FPa",
                    "_source_b2p":  b2p.get("uid"),
                }
                for rule_type in ("permission", "prohibition", "obligation"):
                    if rule_type in b2p:
                        fpa[rule_type] = b2p[rule_type]
                return fpa

        rule_uid = uri_rule(f"rule_{activity_name.replace('-','_')}")
        return {
            "@context":     "http://www.w3.org/ns/odrl.jsonld",
            "uid":          policy_uid,
            "@type":        "Set",
            "_fragment_id": fragment_id,
            "_activity":    activity_name,
            "_type":        "FPa",
            "_source_b2p":  None,
            "permission": [{
                "uid":    rule_uid,
                "target": uri_asset(activity_name),
                "action": "trigger",
                "constraint": [{
                    "leftOperand":  "dateTime",
                    "operator":     "gteq",
                    "rightOperand": {"@value": "2024-01-01", "@type": "xsd:date"}
                }]
            }]
        }

    def _extract_main_rule_id(self, fpa: dict, activity_name: str) -> str:
        for rule_type in ("permission", "prohibition", "obligation"):
            rules = fpa.get(rule_type, [])
            if rules and isinstance(rules, list) and rules[0].get("uid"):
                return rules[0]["uid"]
        return uri_rule(f"rule_{activity_name.replace('-','_')}")

    def _pick_best_b2p_for_activity(
        self,
        activity_name: str,
        candidate_policy_uids: list[str],
    ) -> Optional[dict]:
        """
        Choisit la policy B2P la plus pertinente pour une activité.
        Priorité:
        1) une règle dont target contient explicitement l'activité (slug)
        2) fallback: première policy candidate
        """
        if not candidate_policy_uids:
            return None

        candidates = [
            p for p in self.enriched_graph.raw_b2p
            if p.get("uid") in candidate_policy_uids
        ]
        if not candidates:
            return None

        act_slug = activity_name.replace("_", "-").lower()
        for p in candidates:
            if self._policy_targets_activity(p, act_slug):
                return p
        return candidates[0]

    def _policy_targets_activity(self, b2p_policy: dict, act_slug: str) -> bool:
        for rule_type in ("permission", "prohibition", "obligation"):
            rules = b2p_policy.get(rule_type, [])
            seq = rules if isinstance(rules, list) else [rules]
            for r in seq:
                if not isinstance(r, dict):
                    continue
                tgt = r.get("target")
                if isinstance(tgt, str) and act_slug in tgt.lower().replace("_", "-"):
                    return True
                if isinstance(tgt, dict):
                    tid = tgt.get("@id")
                    if isinstance(tid, str) and act_slug in tid.lower().replace("_", "-"):
                        return True
        return False

    def _pick_main_rule_id_for_activity(self, b2p_policy: dict, activity_name: str) -> str:
        act_slug = activity_name.replace("_", "-").lower()
        for rule_type in ("permission", "prohibition", "obligation"):
            rules = b2p_policy.get(rule_type, [])
            seq = rules if isinstance(rules, list) else [rules]
            # D'abord règle ciblant explicitement l'activité
            for r in seq:
                if not isinstance(r, dict):
                    continue
                tgt = r.get("target")
                uid = r.get("uid")
                if not uid:
                    continue
                tgt_s = ""
                if isinstance(tgt, str):
                    tgt_s = tgt
                elif isinstance(tgt, dict):
                    tgt_s = str(tgt.get("@id") or "")
                if act_slug and act_slug in tgt_s.lower().replace("_", "-"):
                    return uid
            # Fallback sur la première uid disponible
            for r in seq:
                if isinstance(r, dict) and r.get("uid"):
                    return r["uid"]
        return uri_rule(f"rule_{activity_name.replace('-','_')}")

    # ─────────────────────────────────────────
    #  FIX XOR : FPd depuis les patterns du graphe
    # ─────────────────────────────────────────

    def _fpd_from_pattern(self, pattern, fragment_id: str) -> list[dict]:
        """
        Génère les FPd XOR/AND/OR depuis les patterns détectés par l'Agent 1.
        """
        graph   = self.enriched_graph.graph
        gw_node = graph.get_node(pattern.gateway_id)
        if not gw_node:
            return []

        out_edges = graph.out_edges(pattern.gateway_id)
        if len(out_edges) < 2:
            return []

        branches = []
        for edge in out_edges:
            target = graph.get_node(edge.target)
            if target:
                branches.append({
                    "activity":  target.name,
                    "condition": edge.condition or f"condition_{target.name}",
                })

        gw_type  = pattern.pattern_type
        policies = []

        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                b_i, b_j = branches[i], branches[j]
                if gw_type == "fork_xor":
                    policies.append(self._fpd_xor_pair(b_i, b_j, gw_node.name, fragment_id))
                elif gw_type == "fork_and":
                    policies.append(self._fpd_and_pair(b_i, b_j, gw_node.name, fragment_id))
                elif gw_type == "fork_or":
                    policies.append(self._fpd_or_pair(b_i, b_j, gw_node.name, fragment_id))

        return policies

    # ─────────────────────────────────────────
    #  gateway:XOR → Listing 7/8
    # ─────────────────────────────────────────

    def _fpd_xor_pair(self, b_i: dict, b_j: dict, gw_name: str, fragment_id: str) -> dict:
        act_i  = b_i["activity"]
        act_j  = b_j["activity"]
        ruleij = self._activity_rule_index.get(act_i, uri_rule(f"rule_{act_i.replace('-','_')}"))
        ruleik = self._activity_rule_index.get(act_j, uri_rule(f"rule_{act_j.replace('-','_')}"))

        policy_uid = uri_policy(f"FPd_XOR_{new_uid()}")
        r_ij = uri_rule(f"XOR_ij_{act_i}_{new_uid()}")
        r_ik = uri_rule(f"XOR_ik_{act_j}_{new_uid()}")
        return {
            "@context":      "http://www.w3.org/ns/odrl.jsonld",
            "uid":           policy_uid,
            "@type":         "Set",
            "_fragment_id":  fragment_id,
            "_type":         "FPd",
            "_gateway":      "XOR",
            "_gateway_name": gw_name,
            "_activities":   [act_i, act_j],
            "_conditions":   [b_i["condition"], b_j["condition"]],
            "permission": [
                {
                    "uid":    r_ij,
                    "target": ruleij,
                    "action": [{
                        "rdf:value":  {"@id": "odrl:enable"},
                        "refinement": [{"leftOperand": "product",
                                        "operator":    "eq",
                                        "rightOperand": b_i["condition"]}]
                    }]
                },
                {
                    "uid":    r_ik,
                    "target": ruleik,
                    "action": [{
                        "rdf:value":  {"@id": "odrl:enable"},
                        "refinement": [{"leftOperand": "product",
                                        "operator":    "eq",
                                        "rightOperand": b_j["condition"]}]
                    }]
                }
            ]
        }

    # ─────────────────────────────────────────
    #  gateway:AND → Listing 5
    # ─────────────────────────────────────────

    def _fpd_and_pair(self, b_i: dict, b_j: dict, gw_name: str, fragment_id: str) -> dict:
        act_i  = b_i["activity"]
        act_j  = b_j["activity"]
        ruleij = self._activity_rule_index.get(act_i, uri_rule(f"rule_{act_i.replace('-','_')}"))
        ruleik = self._activity_rule_index.get(act_j, uri_rule(f"rule_{act_j.replace('-','_')}"))
        coll   = uri_collection(f"{act_i}_{act_j}_AND")

        policy_uid = uri_policy(f"FPd_AND_{new_uid()}")
        rule_uid = uri_rule(f"AND_{act_i}_{act_j}_{new_uid()}")
        return {
            "@context":      "http://www.w3.org/ns/odrl.jsonld",
            "uid":           policy_uid,
            "@type":         "Set",
            "_fragment_id":  fragment_id,
            "_type":         "FPd",
            "_gateway":      "AND",
            "_gateway_name": gw_name,
            "_activities":   [act_i, act_j],
            "obligation": [{
                "uid":    rule_uid,
                "target": {"@type": "AssetCollection", "uid": coll},
                "action": ODRL_ACTION_ENABLE,
            }],
            "_asset_collection": [
                {"@type": "dc:Document", "@id": ruleij,
                 "dc:title": "concurrent rules", "odrl:partOf": coll},
                {"@type": "dc:Document", "@id": ruleik,
                 "dc:title": "concurrent rules", "odrl:partOf": coll},
            ]
        }

    # ─────────────────────────────────────────
    #  gateway:OR → Listing 6
    # ─────────────────────────────────────────

    def _fpd_or_pair(self, b_i: dict, b_j: dict, gw_name: str, fragment_id: str) -> dict:
        act_i  = b_i["activity"]
        act_j  = b_j["activity"]
        ruleij = self._activity_rule_index.get(act_i, uri_rule(f"rule_{act_i.replace('-','_')}"))
        ruleik = self._activity_rule_index.get(act_j, uri_rule(f"rule_{act_j.replace('-','_')}"))

        policy_uid = uri_policy(f"FPd_OR_{new_uid()}")
        r_ij = uri_rule(f"OR_ij_{act_i}_{new_uid()}")
        r_ik = uri_rule(f"OR_ik_{act_j}_{new_uid()}")
        return {
            "@context":      "http://www.w3.org/ns/odrl.jsonld",
            "uid":           policy_uid,
            "@type":         "Set",
            "_fragment_id":  fragment_id,
            "_type":         "FPd",
            "_gateway":      "OR",
            "_gateway_name": gw_name,
            "_activities":   [act_i, act_j],
            "obligation": [
                {
                    "uid":    r_ij,
                    "target": ruleij,
                    "action": ODRL_ACTION_ENABLE,
                    "consequence": [{"target": ruleik, "action": ODRL_ACTION_ENABLE}],
                },
                {
                    "uid":    r_ik,
                    "target": ruleik,
                    "action": ODRL_ACTION_ENABLE,
                    "consequence": [{"target": ruleij, "action": ODRL_ACTION_ENABLE}],
                }
            ]
        }

    # ─────────────────────────────────────────
    #  flow:sequence → Listing 9
    # ─────────────────────────────────────────

    def _fpd_flow_sequence(self, conn: ConnectionInfo, fragment_id: str) -> dict:
        ruleij    = self._activity_rule_index.get(
            conn.from_activity, uri_rule(f"rule_{conn.from_activity.replace('-','_')}"))
        policy_ik = self._activity_policy_index.get(
            conn.to_activity, uri_policy(f"FPa_{conn.to_activity.replace('-','_')}"))

        policy_uid = uri_policy(f"FPd_SEQ_{new_uid()}")
        rule_uid = uri_rule(f"SEQ_{conn.from_activity}_{conn.to_activity}_{new_uid()}")
        return {
            "@context":     "http://www.w3.org/ns/odrl.jsonld",
            "uid":          policy_uid,
            "@type":        "Set",
            "_fragment_id": fragment_id,
            "_type":        "FPd",
            "_flow":        "sequence",
            "_activities":  [conn.from_activity, conn.to_activity],
            "permission": [{
                "uid":    rule_uid,
                "target": ruleij,
                "action": ODRL_ACTION_ENABLE,
                "duty":   [{"action": "nextPolicy", "uid": policy_ik}],
                "constraint": [{
                    "leftOperand":  "event",
                    "operator":     "gt",
                    "rightOperand": {"@id": "odrl:policyUsage"}
                }]
            }]
        }

    # ─────────────────────────────────────────
    #  flow:message → Listing 10
    # ─────────────────────────────────────────

    def _generate_fpd_message(self, conn, fragment_id: str) -> dict:
        ruleix    = self._activity_rule_index.get(
            conn.from_activity, uri_rule(f"rule_{conn.from_activity.replace('-','_')}"))
        rulejy    = self._activity_rule_index.get(
            conn.to_activity, uri_rule(f"rule_{conn.to_activity.replace('-','_')}"))
        from_frag = getattr(conn, "from_fragment", fragment_id)
        to_frag   = getattr(conn, "to_fragment",   "unknown")

        policy_uid = uri_policy(f"FPd_MSG_{new_uid()}")
        rule_uid = uri_rule(f"MSG_{conn.from_activity}_{conn.to_activity}_{new_uid()}")
        return {
            "@context":       "http://www.w3.org/ns/odrl.jsonld",
            "uid":            policy_uid,
            "@type":          "Set",
            "_fragment_id":   fragment_id,
            "_type":          "FPd",
            "_flow":          "message",
            "_from_fragment": from_frag,
            "_to_fragment":   to_frag,
            "_activities":    [conn.from_activity, conn.to_activity],
            "permission": [{
                "uid":      rule_uid,
                "target":   uri_message(from_frag, to_frag),
                "assignee": ruleix,
                "action": [{
                    "rdf:value":  {"@id": "odrl:transfer"},
                    "refinement": [{
                        "leftOperand":  "recipient",
                        "operator":     "eq",
                        "rightOperand": rulejy
                    }]
                }],
                "duty": [{
                    "target": rulejy,
                    "action": ODRL_ACTION_ENABLE,
                    "constraint": [{
                        "leftOperand":  "event",
                        "operator":     "gt",
                        "rightOperand": {"@id": "odrl:policyUsage"}
                    }]
                }]
            }]
        }
