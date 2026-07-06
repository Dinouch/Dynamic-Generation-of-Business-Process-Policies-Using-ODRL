"""
policy_projection_agent.py — Agent 4: Policy Projection Agent

Single responsibility:
    Generate Fragment Policies (FPa and FPd) as ODRL JSON-LD
    in accordance with section 4.2 of the technical report.

v2 fixes:
    - FPd XOR/AND/OR generated from formal graph PATTERNS
      (connections link gateway→activity, not activity→activity directly)
    - export() method to serialize clean JSON-LD (no _xxx keys)
      and write one .jsonld file per policy

v3 fixes:
    - Refactored LLM prompts: validation_notes before odrl_policy (reasoning
      forced before generation), explicit strict ODRL rules, prohibition of
      recycling BPMN terms in ODRL fields
    - _llm_synthesize_unmapped_policy: LLM receives a targeted snapshot of
      enriched_graph (activities, connections, patterns, gateway types) + hints
      as informational context only — it reasons freely to derive
      the policy without copy-pasting BPMN names into ODRL constraints
    - _build_enriched_graph_snapshot: targeted extraction from the real enriched_graph

Multi-agent layer:
    - receive()  : accepts VALIDATION_DONE, SEMANTIC_*, ODRL_SYNTAX_FAILURE, ODRL_VALID
    - Flow       : SYNTAX_AUDIT_REQUEST (A5) → ODRL_VALID → POLICIES_READY (A3, semantic)
                  → SEMANTIC_VALIDATED → SYNTAX_AUDIT_REQUEST (A5, final pass)
    - Syntax loop (MAX_SYNTAX_LOOPS on A5 side): ODRL_SYNTAX_FAILURE → SYNTAX_CORRECTION
      (A4→A5) → AGREE (A5) → corrections → POLICIES_READY to A5
"""

import copy
import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import AzureOpenAI, OpenAI

from ..structural_analyzer import (
    AgentMessage,
    EnrichedGraph,
    ConnectionInfo,
    MessageType,
)
from ..Agent_3.constraint_validator import ValidationReport
from ..exception_handling_agent import UnmappedCaseProposal
from ..bpmn_odrl_reasoning_prompts import (
    GENERATOR_INTERNAL_REASONING,
    RULE_AND_TARGET_BINDING,
    SURGICAL_CORRECTION_FOR_TEMPLATES,
    generator_user_prompt_body,
    semantic_repair_user_prompt_body,
)

from .odrl_deterministic_templates import (
    FragmentPolicySet,
    BASE_URI,
    coerce_odrl_action_from_hint,
    compact_jsonld_fpdm_inline_ids,
    is_deterministic_template_fpd,
    new_uid,
    sanitize_unmapped_odrl_constraints,
    template_fpa_default_without_b2p,
    template_fpa_from_b2p,
    template_fpd_and_pair,
    template_fpd_flow_sequence,
    template_fpd_message,
    template_fpd_or_pair,
    template_fpd_xor_pair,
    template_fallback_unmapped_fpd,
    uri_asset,
    uri_collection,
    uri_policy,
    uri_rule,
)


# ─────────────────────────────────────────────
#  Agent 4 — Policy Projection Agent
# ─────────────────────────────────────────────

class PolicyProjectionAgent:
    """
    Agent 4 of the multi-agent pipeline.

    Standalone mode:
        Instantiate with enriched_graph + validation_report and call generate().

    Pipeline mode:
        Instantiate with enriched_graph=None, validation_report=None.
        Data arrives via receive(VALIDATION_DONE).
        Register the bus callback (A3 / A5 / pipeline) before startup.

    Sequence:
        After projection: syntax audit (A5), then semantic (A3), then final syntax audit (A5).

    Syntax loop:
        Agent 5 emits ODRL_SYNTAX_FAILURE; Agent 4 sends SYNTAX_CORRECTION, applies
        deterministic patches, then re-emits POLICIES_READY to A5. MAX_SYNTAX_LOOPS is managed by A5.
    """

    AGENT_NAME = "agent4"
    MODEL = "gpt-5.2"
    TEMPERATURE = 0.2

    # ── Prompts LLM (unmapped pattern synthesis) ──────────────────────────
    # Latent reasoning (see bpmn_odrl_reasoning_prompts); output = odrl_policy only.
    _UNMAPPED_SYSTEM = f"""\
You are an ODRL 2.2 structural compiler: BPMN control-flow → conditional rule activation → ODRL.

{GENERATOR_INTERNAL_REASONING}

{RULE_AND_TARGET_BINDING}

Your visible output is ONLY a JSON object containing the key "odrl_policy".
Never output reasoning, compilation traces, or validation notes.

Hard rules for odrl_policy fields:
• Never use the same leftOperand twice with operator eq in AND-combined constraints.
• constraint rightOperand = operational business condition / event — never BPMN routing vocabulary.
• action = strict ODRL 2.2 vocabulary only (use, transfer, read, modify, delete, execute, …).
• target = existing branch activity asset OR existing B2P rule uid — never invent IRIs.
• Policies govern rule enablement; they do not "describe" gateways as targets.
• Each rule must include at least one constraint (or action refinement).
• Hints from the exception agent are context only — not field values to copy.

If compilation yields no coherent rule, emit a minimal prohibition (action use) on a valid
branch activity asset from valid_asset_targets — not on the gateway name.
"""

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

        self._unmapped_fpd_llm_cache: dict[str, dict] = {}
        self._unmapped_cache_lock = threading.Lock()

        self._increment_profile: Optional[str] = None
        self._increment_stop_before_semantic: bool = False
        self._use_azure = False
        self._deployment: Optional[str] = None
        self._openai_model = (
            (os.environ.get("OPENAI_MODEL") or "").strip() or PolicyProjectionAgent.MODEL
        )
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
                or PolicyProjectionAgent.MODEL
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

        self._on_send:             Optional[Callable[[AgentMessage], None]] = None
        self._last_fp_results:     Optional[dict] = None
        self._last_enriched_graph: Optional[EnrichedGraph] = None
        self._semantic_agree_anchor: Optional[str] = None
        self._pending_syntax_correction: Optional[dict[str, Any]] = None
        self._odrl_syntax_stage: Optional[str] = None
        self._pipeline_content_status: Optional[str] = None

    # ═════════════════════════════════════════
    #  MULTI-AGENT LAYER
    # ═════════════════════════════════════════

    def register_send_callback(self, fn: Callable[[AgentMessage], None]) -> None:
        self._on_send = fn

    def send(self, msg: AgentMessage) -> None:
        print(f"[Agent 4] ► SEND {msg}")
        if self._on_send:
            self._on_send(msg)
        else:
            print(f"[Agent 4][WARN] No callback — message {msg.msg_type.value} not sent.")

    def receive(self, msg: AgentMessage) -> None:
        print(f"[Agent 4] ◄ RECEIVE {msg}")
        if (
            msg.sender == "agent5"
            and msg.msg_type == MessageType.GRAPH_READY
            and (msg.payload or {}).get("_acl_ontology") == "odrl-syntax-audit"
        ):
            print(
                "[Agent 4][INFO] Catch-up: graph_ready + odrl-syntax-audit from Agent 5 → DELEGATION_AGREE."
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
            if msg.sender == "agent5" and self._pending_syntax_correction is not None:
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

        elif msg.msg_type == MessageType.FP_BUNDLE_READY:
            fp_results = msg.payload.get("fp_results")
            enriched_graph = msg.payload.get("enriched_graph")
            if fp_results is not None:
                self._last_fp_results = fp_results
            if enriched_graph is not None:
                self._last_enriched_graph = enriched_graph
            self._pipeline_content_status = (msg.payload or {}).get("status")
            print(
                "[Agent 4] FP_BUNDLE_READY — merged batch received; "
                "syntax then semantic audit (no intermediate export)."
            )
            self._odrl_syntax_stage = "pre_semantic"
            self._emit_syntax_audit_request(msg.loop_turn)

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
            if msg.sender == "agent5":
                print("[Agent 4][WARN] SYNTAX_CORRECTION from Agent 5 (legacy) — applying without AGREE handshake.")
                self._handle_syntax_correction(msg)
            else:
                print(f"[Agent 4][WARN] SYNTAX_CORRECTION from {msg.sender} — ignored.")

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
                if self._increment_stop_before_semantic:
                    self._forward_final_audit_to_pipeline(msg)
                    return
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
            print(f"[Agent 4][WARN] Message '{msg.msg_type.value}' not handled — ignored.")

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
        pl = dict(msg.payload or {})
        if self._last_fp_results is not None:
            pl["fp_results"] = self._last_fp_results
        if self._last_enriched_graph is not None:
            pl["enriched_graph"] = self._last_enriched_graph
        self.send(
            AgentMessage(
                sender=self.AGENT_NAME,
                recipient="pipeline",
                msg_type=msg.msg_type,
                payload=pl,
                loop_turn=msg.loop_turn,
            )
        )

    # ─────────────────────────────────────────
    #  Internal handlers
    # ─────────────────────────────────────────

    def _project_and_send(
        self,
        enriched_graph:    EnrichedGraph,
        validation_report: ValidationReport,
        loop_turn:         int,
    ) -> None:
        fp_results = self.project(enriched_graph, validation_report)
        self._last_fp_results = fp_results
        self._odrl_syntax_stage = "pre_semantic"
        self._emit_syntax_audit_request(loop_turn)

    def _emit_syntax_audit_request(self, loop_turn: int) -> None:
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
        affected_policies = list(pending.get("affected_policies") or [])
        errors = list(pending.get("errors") or [])
        loop_turn = int(pending.get("loop_turn", 0) or 0)
        print(
            f"[Agent 4] SYNTAX repair (post A5 AGREE) — "
            f"{len(affected_policies)} policy(ies), {len(errors)} error(s)"
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
        affected_policies = msg.payload.get("affected_policies", [])
        errors = msg.payload.get("errors", [])
        print(
            f"[Agent 4] SYNTAX_CORRECTION (legacy) — "
            f"{len(affected_policies)} policy(ies), {len(errors)} error(s)"
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
        Apply deterministic JSON-LD corrections on affected policies.
        No LLM — structural patches on ODRL fields only.
        """
        if self._last_fp_results is None:
            print("[Agent 4][WARN] _last_fp_results is None — no correction possible.")
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
                            print(f"[Agent 4] MISSING_CONTEXT fixed: {policy_uid}")

                    elif code == "MISSING_TYPE":
                        if "@type" not in policy:
                            policy["@type"] = "Agreement"
                            print(f"[Agent 4] MISSING_TYPE fixed: {policy_uid}")

                    elif code == "MALFORMED_URI":
                        import re

                        suggestion = error.get("suggestion") or ""
                        match = re.search(r'https?://[^\s"\'\.>]+', suggestion)
                        if match:
                            policy["uid"] = match.group(0)
                            print(
                                f"[Agent 4] MALFORMED_URI fixed: {suggestion[:60]} → {policy['uid']}"
                            )
                        else:
                            uid = policy.get("uid", "")
                            if uid and not uid.startswith("http"):
                                policy["uid"] = f"{BASE_URI}/{uid}"
                                print(
                                    f"[Agent 4] MALFORMED_URI fixed (fallback): {uid} → {policy['uid']}"
                                )

                    elif code == "MISSING_UID":
                        if not policy.get("uid"):
                            policy["uid"] = uri_policy(f"FP_{new_uid()}")
                            print(f"[Agent 4] MISSING_UID fixed: {policy['uid']}")

                    elif code == "RULE_MISSING_UID":
                        rule_type = path or "permission"
                        rules = policy.get(rule_type, [])
                        for rule in rules:
                            if isinstance(rule, dict) and not rule.get("uid"):
                                ru = uri_rule(f"rule_{new_uid()}")
                                rule["uid"] = ru
                                print(
                                    f"[Agent 4] RULE_MISSING_UID fixed in "
                                    f"'{rule_type}' of {policy_uid}"
                                )

                    else:
                        print(
                            f"[Agent 4][WARN] Unknown error code '{code}' "
                            f"for {policy_uid} — ignored."
                        )

    def _merge_public_odrl_into_policy(self, original: dict, odrl_public: dict) -> dict:
        out = {k: v for k, v in odrl_public.items() if not str(k).startswith("_")}
        for k, v in original.items():
            if str(k).startswith("_"):
                out[k] = v
        return out

    def _llm_apply_semantic_repairs(
        self,
        hints: list[dict[str, Any]],
        summary: str,
        warnings: list[str],
        validation_reports: Optional[list[dict[str, Any]]] = None,
        *,
        template_mode: bool = False,
    ) -> bool:
        if not self.client or not self._last_fp_results or not hints:
            return False
        if not summary:
            summary = "\n".join(
                f"- {h.get('policy_uid')}: {h.get('issue')} (path={h.get('field_path')})"
                for h in hints
            )
        affected = {str(h.get("policy_uid") or "") for h in hints if h.get("policy_uid")}
        affected.discard("")
        bundles: list[dict[str, Any]] = []
        for uid in affected:
            pol = self._find_policy_by_uid(uid)
            if not pol:
                continue
            if template_mode and not is_deterministic_template_fpd(pol):
                continue
            if not template_mode and is_deterministic_template_fpd(pol):
                continue
            pub = {k: v for k, v in pol.items() if not str(k).startswith("_")}
            meta = {
                k: v
                for k, v in pol.items()
                if str(k).startswith("_")
            }
            bundles.append({"policy_uid": uid, "policy": pub, "metadata": meta})
        if not bundles:
            return False
        if template_mode:
            system = (
                "You are an ODRL 2.2 constraint editor for deterministic fragment-policy templates. "
                "Apply SURGICAL CORRECTION ONLY — never rewrite policy structure. "
                + SURGICAL_CORRECTION_FOR_TEMPLATES
                + " Output one JSON object at the end, no markdown."
            )
        else:
            system = (
                "You are an ODRL 2.2 policy editor for unmapped BPMN-compiled fragment policies. "
                "Apply minimal targeted fixes aligned with validation hints. "
                "Do not collapse unrelated policies into identical shapes. "
                "Output one JSON object at the end, no markdown."
            )
        user = semantic_repair_user_prompt_body(
            summary=summary,
            validation_reports_json=json.dumps(validation_reports or [], ensure_ascii=False, indent=2),
            warnings_json=json.dumps(warnings or [], ensure_ascii=False, indent=2),
            hints_json=json.dumps(hints, ensure_ascii=False, indent=2),
            bundles_json=json.dumps(bundles, ensure_ascii=False, indent=2),
            template_mode=template_mode,
        )
        try:
            raw = self._call_llm(system, user)
            data = self._parse_llm_json_object(raw)
            corrected = data.get("corrected")
            if not isinstance(corrected, list):
                return False
            updated = 0
            for item in corrected:
                if not isinstance(item, dict):
                    continue
                uid = item.get("policy_uid")
                odrl = item.get("odrl")
                if not uid or not isinstance(odrl, dict):
                    continue
                orig = self._find_policy_by_uid(str(uid))
                if orig is None:
                    continue
                merged = self._merge_public_odrl_into_policy(orig, odrl)
                self._normalize_odrl_actions_in_policy(merged)
                sanitize_unmapped_odrl_constraints(merged)
                orig.clear()
                orig.update(merged)
                updated += 1
            if updated:
                print(f"[Agent 4] LLM semantic repair applied to {updated} policy/policies.")
            return updated > 0
        except Exception as e:
            print(f"[Agent 4][WARN] LLM semantic repair failed: {e}")
            return False

    def _handle_semantic_correction(self, msg: AgentMessage) -> None:
        hints = msg.payload.get("semantic_hints") or []
        fp_results = msg.payload.get("fp_results") or self._last_fp_results
        if fp_results is not None:
            self._last_fp_results = fp_results
        eg = msg.payload.get("enriched_graph")
        if eg is not None:
            self._last_enriched_graph = eg

        summary = (msg.payload.get("semantic_correction_summary") or "").strip()
        warnings_list = list(msg.payload.get("semantic_warnings") or [])
        validation_reports = list(msg.payload.get("semantic_validation_reports") or [])

        print(
            f"[Agent 4] SEMANTIC_CORRECTION — {len(hints)} hint(s)"
            + (f" ; Agent 3 summary: {len(summary)} characters" if summary else "")
            + (
                f" ; {len(validation_reports)} business report(s)"
                if validation_reports
                else ""
            )
        )

        template_hints: list[dict[str, Any]] = []
        other_hints: list[dict[str, Any]] = []
        for h in hints:
            uid = str(h.get("policy_uid") or "")
            pol = self._find_policy_by_uid(uid) if uid else None
            if pol is not None and is_deterministic_template_fpd(pol):
                template_hints.append(h)
            else:
                other_hints.append(h)

        patched = 0
        for h in template_hints:
            uid = h.get("policy_uid")
            if not uid:
                continue
            policy = self._find_policy_by_uid(uid)
            if policy is None:
                continue
            path = h.get("field_path", "")
            fix = h.get("suggested_fix", "")
            if path and fix:
                self._set_field_path(policy, path, fix)
                patched += 1
            else:
                print(
                    f"[Agent 4] Template FPd '{uid}': hint without field_path/fix — "
                    "structure preserved (surgical correction impossible)."
                )
        if patched:
            print(f"[Agent 4] Deterministic patches applied to {patched} template field(s).")

        repaired_llm = False
        if other_hints:
            repaired_llm = self._llm_apply_semantic_repairs(
                other_hints, summary, warnings_list, validation_reports=validation_reports,
                template_mode=False,
            )
        if not repaired_llm and other_hints:
            print("[Agent 4] Fallback: deterministic application of hints (field_path + suggested_fix).")
            for h in other_hints:
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
        elif not repaired_llm and template_hints and not patched:
            print("[Agent 4] Deterministic templates: no LLM rewrite (structure preserved).")

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
        if not self._last_fp_results:
            return None
        for fps in self._last_fp_results.values():
            for pol in fps.all_policies():
                if pol.get("uid") == policy_uid:
                    return pol
        return None

    def _set_field_path(self, policy: dict, path: str, fix: str) -> None:
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

    # ─────────────────────────────────────────
    #  Enriched graph snapshot
    # ─────────────────────────────────────────

    def _build_enriched_graph_snapshot(
        self,
        proposal: UnmappedCaseProposal,
    ) -> dict[str, Any]:
        """
        Extract a targeted snapshot of enriched_graph for fragments involved
        in the unmapped proposal.

        The snapshot contains only what the LLM needs to reason:
        activities, connections (type + source/target), and detected patterns.
        Internal BPMN names (gateway IDs, edge IDs) are excluded — only
        business names (activity names, connection labels) are exposed.

        This snapshot is passed to the LLM as reasoning context, not as a
        template to copy.
        """
        eg = self.enriched_graph
        if not eg:
            return {}

        target_fragments: list[str] = list(proposal.involved_fragment_ids or [])
        if not target_fragments and proposal.fragment_id:
            target_fragments = [proposal.fragment_id]

        snapshot: dict[str, Any] = {
            "involved_fragments": [],
            "detected_patterns": [],
        }

        for frag_id in target_fragments:
            ctx = eg.fragment_contexts.get(frag_id)
            if not ctx:
                continue

            frag_entry: dict[str, Any] = {
                "fragment_id": frag_id,
                "activities": list(ctx.activities or []),
                "connections": [],
                "downstream_dependencies": [],
            }

            for conn in (ctx.connections or []):
                conn_entry: dict[str, Any] = {
                    "from": getattr(conn, "from_activity", None),
                    "to": getattr(conn, "to_activity", None),
                    "type": getattr(conn, "connection_type", None),
                    "condition": getattr(conn, "condition", None),
                }
                frag_entry["connections"].append(
                    {k: v for k, v in conn_entry.items() if v is not None}
                )

            for dep in (ctx.downstream_deps or []):
                dep_entry: dict[str, Any] = {
                    "from": getattr(dep, "from_activity", None),
                    "to": getattr(dep, "to_activity", None),
                    "from_fragment": getattr(dep, "from_fragment", None),
                    "to_fragment": getattr(dep, "to_fragment", None),
                }
                frag_entry["downstream_dependencies"].append(
                    {k: v for k, v in dep_entry.items() if v is not None}
                )

            snapshot["involved_fragments"].append(frag_entry)

        all_acts: set[str] = set()
        for fe in snapshot["involved_fragments"]:
            all_acts.update(fe.get("activities") or [])
        snapshot["valid_asset_targets"] = sorted(uri_asset(a) for a in all_acts if a)

        # Structural patterns detected by Agent 1 for these fragments
        for pattern in (getattr(eg, "patterns", None) or []):
            if pattern.fragment_id not in target_fragments:
                continue
            snapshot["detected_patterns"].append({
                "fragment_id": pattern.fragment_id,
                "pattern_type": pattern.pattern_type,
                "gateway_name": getattr(pattern, "gateway_name", None),
                # Branches: activity names only (not internal IDs)
                "branches": [
                    {
                        "activity": eg.graph.get_node(e.target).name
                        if eg.graph.get_node(e.target) else e.target,
                        "condition": e.condition,
                    }
                    for e in (eg.graph.out_edges(pattern.gateway_id) or [])
                    if eg.graph.get_node(e.target)
                ] if hasattr(pattern, "gateway_id") else [],
            })

        return snapshot

    # ─────────────────────────────────────────
    #  LLM synthesis (unmapped patterns)
    # ─────────────────────────────────────────

    def _llm_synthesize_unmapped_policy(self, proposal: UnmappedCaseProposal) -> Optional[dict]:
        """
        Generate an ODRL policy for an unsupported BPMN pattern.

        The LLM receives:
        - A targeted enriched_graph snapshot (activities, connections, detected patterns)
          as the primary reasoning source.
        - Exception handling agent hints as informational context only
          (WHY the pattern is unmapped) — not as a template to copy.

        Internal reasoning required; JSON (odrl_policy only) as the final prompt step.
        """
        eg_snapshot = self._build_enriched_graph_snapshot(proposal)

        payload = {
            "base_uri": BASE_URI,
            "requested_rule_type": proposal.odrl_rule_type,
            "pattern_type": proposal.pattern_type,
            "exception_handling_hints": {
                "pattern_type": proposal.pattern_type,
                "gateway_name": proposal.gateway_name,
                "hint_text": proposal.hint_text,
                "note": (
                    "Context only — understand WHY the pattern is unmapped; "
                    "do not copy hint phrases into ODRL constraint values. "
                    "Compile: event/condition → branch → rule target (activity asset), "
                    "not gateway-as-target."
                ),
            },
            "compilation_expectation": (
                "One governable rule per branch activity unless BPMN converges; "
                "gateway semantics = conditional enablement, not a policy target."
            ),
            "enriched_graph_snapshot": eg_snapshot,
        }

        user_prompt = generator_user_prompt_body(
            payload_json=json.dumps(payload, ensure_ascii=False, indent=2),
            requested_rule_type=proposal.odrl_rule_type,
            base_uri=BASE_URI,
            pattern_type=proposal.pattern_type,
        )
        raw = self._call_llm(self._UNMAPPED_SYSTEM, user_prompt)
        data = self._parse_llm_json_object(raw)

        odrl = data.get("odrl_policy")
        if not isinstance(odrl, dict) and isinstance(data.get("@context"), str):
            odrl = data
        if not isinstance(odrl, dict):
            print("[Agent 4][WARN] LLM response missing odrl_policy object")
            return None

        if not odrl.get("uid"):
            odrl["uid"] = uri_policy(f"FPd_UNH_{proposal.pattern_type}_{new_uid()}")
        if "@context" not in odrl:
            odrl["@context"] = "http://www.w3.org/ns/odrl.jsonld"

        self._normalize_odrl_actions_in_policy(odrl)
        sanitize_unmapped_odrl_constraints(odrl)
        if proposal.hint_text:
            odrl["_business_intent"] = proposal.hint_text.strip()
        return odrl

    def _fallback_minimal_unmapped_fpd(
        self,
        proposal: UnmappedCaseProposal,
        target_fragment_id: Optional[str],
    ) -> dict:
        frag = target_fragment_id or proposal.fragment_id or "fragment"
        uid = uri_policy(f"FPd_UNH_{proposal.pattern_type}_{frag.replace('-', '_')}_{new_uid()}")
        rule_uid = uri_rule(f"rule_unh_{new_uid()}")
        rt = proposal.odrl_rule_type if proposal.odrl_rule_type in (
            "permission",
            "prohibition",
            "obligation",
        ) else "permission"
        tgt = uri_asset(proposal.gateway_name or frag or "asset")
        return template_fallback_unmapped_fpd(
            policy_uid=uid,
            rule_uid=rule_uid,
            fragment_id=frag,
            pattern_type=proposal.pattern_type,
            gateway_name=proposal.gateway_name,
            hint_text=proposal.hint_text,
            rule_kind=rt,
            target_asset_uri=tgt,
        )

    def _generate_fpd_from_unmapped_proposal(
        self,
        proposal: UnmappedCaseProposal,
        target_fragment_id: Optional[str] = None,
    ) -> Optional[dict]:
        cache_key = json.dumps(proposal.to_dict(), sort_keys=True)
        with self._unmapped_cache_lock:
            if cache_key not in self._unmapped_fpd_llm_cache:
                body: Optional[dict] = None
                if self.client:
                    try:
                        body = self._llm_synthesize_unmapped_policy(proposal)
                    except Exception as e:
                        print(f"[Agent 4][WARN] LLM unmapped synthesis failed: {e}")
                if body is None:
                    print(
                        "[Agent 4] Minimal deterministic fallback for unmapped "
                        f"({proposal.pattern_type}, target fragment "
                        f"{target_fragment_id or proposal.fragment_id})"
                    )
                    body = self._fallback_minimal_unmapped_fpd(proposal, target_fragment_id)
                self._unmapped_fpd_llm_cache[cache_key] = body

            base = self._unmapped_fpd_llm_cache[cache_key]
        out = copy.deepcopy(base)
        frag = target_fragment_id or proposal.fragment_id or "fragment"
        out["_fragment_id"] = frag
        out["_type"] = "FPd"
        out["_unmapped_pattern"] = proposal.pattern_type
        out["_gateway_name"] = proposal.gateway_name
        out["_hint_text"] = proposal.hint_text
        if out.get("_business_intent"):
            pass
        elif proposal.hint_text:
            out["_business_intent"] = proposal.hint_text
        return out

    # ─────────────────────────────────────────
    #  LLM utilities
    # ─────────────────────────────────────────

    def _call_llm(self, system: str, user: str) -> str:
        if not self.client:
            raise RuntimeError("LLM client not configured for Agent 4")
        model = self._deployment if self._use_azure else self._openai_model
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
            raise RuntimeError("Empty LLM response (Agent 4 unmapped)")
        return content

    def _normalize_odrl_actions_in_policy(self, policy: dict) -> None:
        for key in ("permission", "prohibition", "obligation"):
            rules = policy.get(key)
            if not isinstance(rules, list):
                continue
            for rule in rules:
                if isinstance(rule, dict) and "action" in rule:
                    rule["action"] = coerce_odrl_action_from_hint(rule.get("action"))

    # ═════════════════════════════════════════
    #  BUSINESS LOGIC (unchanged)
    # ═════════════════════════════════════════

    def project_i0_llm_baseline(self, **kwargs: Any) -> dict[str, "FragmentPolicySet"]:
        from .i0_llm_baseline import generate_i0_baseline_for_scenario

        print(
            "[Agent 4][WARN] project_i0_llm_baseline is deprecated — "
            "I0 uses i0_llm_baseline (single prompt, no templates)."
        )
        allowed = frozenset({
            "scenario_id",
            "fragments",
            "fragments_enhanced",
            "b2p_policies",
            "bp_model",
            "api_key",
            "azure_endpoint",
            "azure_api_version",
            "azure_deployment",
        })
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        if "api_key" not in filtered and self.client:
            filtered.setdefault("api_key", os.environ.get("OPENAI_API_KEY"))
        return generate_i0_baseline_for_scenario(**filtered)

    def project(
        self,
        enriched_graph:    Optional[EnrichedGraph] = None,
        validation_report: Optional[ValidationReport] = None,
    ) -> dict[str, "FragmentPolicySet"]:
        graph  = enriched_graph    or self.enriched_graph
        report = validation_report or self.validation_report

        if graph is None:
            raise ValueError("[Agent 4] No enriched_graph available for projection.")

        self.enriched_graph    = graph
        self.validation_report = report

        return self.generate()

    def generate_unmapped_only(
        self,
        enriched_graph: EnrichedGraph,
        validation_report: ValidationReport,
    ) -> dict[str, "FragmentPolicySet"]:
        self.enriched_graph = enriched_graph
        self.validation_report = validation_report

        self._preindex_all_activities()

        results: dict[str, FragmentPolicySet] = {
            frag_id: FragmentPolicySet(fragment_id=frag_id)
            for frag_id in self.enriched_graph.fragment_contexts.keys()
        }

        if not self.validation_report:
            return results

        accepted = getattr(self.validation_report, "accepted_unmapped_proposals", []) or []
        for raw in accepted:
            prop = UnmappedCaseProposal.from_dict(raw) if isinstance(raw, dict) else raw
            targets = list(prop.involved_fragment_ids or [])
            if not targets:
                targets = [prop.fragment_id] if prop.fragment_id else []
            for frag_id in targets:
                if frag_id not in results:
                    continue
                fpd = self._generate_fpd_from_unmapped_proposal(prop, target_fragment_id=frag_id)
                if fpd:
                    results[frag_id].fpd_policies.append(fpd)

        return results

    # ─────────────────────────────────────────
    #  Entry point — generation (sequential)
    # ─────────────────────────────────────────

    def generate(self) -> dict[str, "FragmentPolicySet"]:
        print("[Agent 4] Policy Projection Agent — starting generation")

        self._unmapped_fpd_llm_cache = {}
        self._preindex_all_activities()

        results: dict[str, FragmentPolicySet] = {}
        for fragment_id in self.enriched_graph.fragment_contexts.keys():
            fps = self._generate_fragment(fragment_id)
            results[fragment_id] = fps
        return results

    # ─────────────────────────────────────────
    #  Entry point — generation (parallel)
    # ─────────────────────────────────────────

    def generate_parallel(self, max_workers: Optional[int] = None) -> dict[str, "FragmentPolicySet"]:
        print("[Agent 4] Policy Projection Agent — parallel generation")
        self._unmapped_fpd_llm_cache = {}
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
    #  Per-fragment generation (thread-safe)
    # ─────────────────────────────────────────

    def _generate_fragment(self, fragment_id: str) -> FragmentPolicySet:
        context = self.enriched_graph.fragment_contexts[fragment_id]
        print(f"[Agent 4] Generating policies for '{fragment_id}'...")

        fps = FragmentPolicySet(fragment_id=fragment_id)

        for activity_name in context.activities:
            fpa = self._generate_fpa(activity_name, fragment_id)
            if fpa:
                fps.fpa_policies.append(fpa)

        for pattern in self.enriched_graph.patterns:
            if pattern.fragment_id != fragment_id:
                continue
            if pattern.pattern_type not in ("fork_xor", "fork_and", "fork_or"):
                continue
            fps.fpd_policies.extend(self._fpd_from_pattern(pattern, fragment_id))

        for conn in context.connections:
            if conn.connection_type.lower() == "sequence" and not conn.is_inter:
                fpd = self._fpd_flow_sequence(conn, fragment_id)
                if fpd:
                    fps.fpd_policies.append(fpd)

        for conn in context.downstream_deps:
            fpd = self._generate_fpd_message(conn, fragment_id)
            if fpd:
                fps.fpd_policies.append(fpd)

        if self.validation_report:
            for prop in getattr(
                self.validation_report, "accepted_unmapped_proposals", []
            ) or []:
                if isinstance(prop, dict):
                    prop = UnmappedCaseProposal.from_dict(prop)
                targets = list(prop.involved_fragment_ids or [])
                if not targets:
                    targets = [prop.fragment_id] if prop.fragment_id else []
                if fragment_id not in targets:
                    continue
                fpd = self._generate_fpd_from_unmapped_proposal(
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
        *,
        replace_existing: bool = True,
    ) -> dict[str, list[str]]:
        """
        Write policies as .jsonld files.

        ``replace_existing=True``: removes previous .jsonld files in the output folder
        (avoids accumulating versions rejected by the semantic validator).
        """
        os.makedirs(output_dir, exist_ok=True)
        exported: dict[str, list[str]] = {}

        for fragment_id, fps in fp_results.items():
            frag_dir = os.path.join(output_dir, fragment_id)
            if replace_existing and os.path.isdir(frag_dir):
                shutil.rmtree(frag_dir, ignore_errors=True)
            os.makedirs(frag_dir, exist_ok=True)
            exported[fragment_id] = []

            for i, policy in enumerate(fps.all_policies()):
                odrl = {k: v for k, v in policy.items() if not k.startswith("_")}

                ptype = policy.get("_type", "FP")
                subtype = (
                    policy.get("_gateway")
                    or policy.get("_flow")
                    or policy.get("_dep_type")
                    or policy.get("_unmapped_pattern")
                    or str(i)
                )
                activity = (
                    policy.get("_activity")
                    or "_".join(policy.get("_activities", [])[:2])
                    or policy.get("_gateway_name", "policy")
                )
                uid = str(odrl.get("uid") or odrl.get("@id") or f"idx_{i}")
                uid_short = hashlib.sha1(uid.encode("utf-8")).hexdigest()[:8]
                filename = (
                    f"{ptype}_{subtype}_{activity.replace('-', '_')}_{uid_short}.jsonld"
                )
                filepath = os.path.join(frag_dir, filename)
                counter = 1
                base_path = filepath
                while os.path.exists(filepath):
                    filepath = base_path.replace(".jsonld", f"__{counter}.jsonld")
                    counter += 1

                payload = json.dumps(odrl, indent=2, ensure_ascii=False)
                if policy.get("_type") == "FPd" and policy.get("_flow") == "message":
                    payload = compact_jsonld_fpdm_inline_ids(payload)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(payload)
                exported[fragment_id].append(filepath)

        total = sum(len(v) for v in exported.values())
        print(f"[Agent 4] Export complete: {total} .jsonld files -> '{output_dir}'")
        return exported

    # ─────────────────────────────────────────
    #  Global pre-indexing
    # ─────────────────────────────────────────

    def _preindex_all_activities(self) -> None:
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
                return template_fpa_from_b2p(
                    policy_uid=policy_uid,
                    fragment_id=fragment_id,
                    activity_name=activity_name,
                    source_b2p_uid=b2p.get("uid"),
                    b2p=b2p,
                )

        rule_uid = uri_rule(f"rule_{activity_name.replace('-','_')}")
        return template_fpa_default_without_b2p(
            policy_uid=policy_uid,
            fragment_id=fragment_id,
            activity_name=activity_name,
            rule_uid=rule_uid,
        )

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
            for r in seq:
                if isinstance(r, dict) and r.get("uid"):
                    return r["uid"]
        return uri_rule(f"rule_{activity_name.replace('-','_')}")

    # ─────────────────────────────────────────
    #  FPd from graph patterns
    # ─────────────────────────────────────────

    def _fpd_from_pattern(self, pattern, fragment_id: str) -> list[dict]:
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

    def _fpd_xor_pair(self, b_i: dict, b_j: dict, gw_name: str, fragment_id: str) -> dict:
        act_i  = b_i["activity"]
        act_j  = b_j["activity"]
        ruleij = self._activity_rule_index.get(act_i, uri_rule(f"rule_{act_i.replace('-','_')}"))
        ruleik = self._activity_rule_index.get(act_j, uri_rule(f"rule_{act_j.replace('-','_')}"))

        policy_uid = uri_policy(f"FPd_XOR_{new_uid()}")
        r_ij = uri_rule(f"XOR_ij_{act_i}_{new_uid()}")
        r_ik = uri_rule(f"XOR_ik_{act_j}_{new_uid()}")
        return template_fpd_xor_pair(
            policy_uid=policy_uid,
            fragment_id=fragment_id,
            gw_name=gw_name,
            act_i=act_i,
            act_j=act_j,
            ruleij_uri=ruleij,
            ruleik_uri=ruleik,
            perm_rule_ij_uid=r_ij,
            perm_rule_ik_uid=r_ik,
            condition_i=b_i["condition"],
            condition_j=b_j["condition"],
        )

    def _fpd_and_pair(self, b_i: dict, b_j: dict, gw_name: str, fragment_id: str) -> dict:
        act_i  = b_i["activity"]
        act_j  = b_j["activity"]
        ruleij = self._activity_rule_index.get(act_i, uri_rule(f"rule_{act_i.replace('-','_')}"))
        ruleik = self._activity_rule_index.get(act_j, uri_rule(f"rule_{act_j.replace('-','_')}"))
        coll   = uri_collection(f"{act_i}_{act_j}_AND")

        policy_uid = uri_policy(f"FPd_AND_{new_uid()}")
        rule_uid = uri_rule(f"AND_{act_i}_{act_j}_{new_uid()}")
        return template_fpd_and_pair(
            policy_uid=policy_uid,
            fragment_id=fragment_id,
            gw_name=gw_name,
            act_i=act_i,
            act_j=act_j,
            ruleij_uri=ruleij,
            ruleik_uri=ruleik,
            collection_uid=coll,
            obligation_rule_uid=rule_uid,
        )

    def _fpd_or_pair(self, b_i: dict, b_j: dict, gw_name: str, fragment_id: str) -> dict:
        act_i  = b_i["activity"]
        act_j  = b_j["activity"]
        ruleij = self._activity_rule_index.get(act_i, uri_rule(f"rule_{act_i.replace('-','_')}"))
        ruleik = self._activity_rule_index.get(act_j, uri_rule(f"rule_{act_j.replace('-','_')}"))

        policy_uid = uri_policy(f"FPd_OR_{new_uid()}")
        r_ij = uri_rule(f"OR_ij_{act_i}_{new_uid()}")
        r_ik = uri_rule(f"OR_ik_{act_j}_{new_uid()}")
        return template_fpd_or_pair(
            policy_uid=policy_uid,
            fragment_id=fragment_id,
            gw_name=gw_name,
            act_i=act_i,
            act_j=act_j,
            ruleij_uri=ruleij,
            ruleik_uri=ruleik,
            obligation_ij_uid=r_ij,
            obligation_ik_uid=r_ik,
        )

    def _fpd_flow_sequence(self, conn: ConnectionInfo, fragment_id: str) -> dict:
        ruleij    = self._activity_rule_index.get(
            conn.from_activity, uri_rule(f"rule_{conn.from_activity.replace('-','_')}"))
        policy_ik = self._activity_policy_index.get(
            conn.to_activity, uri_policy(f"FPa_{conn.to_activity.replace('-','_')}"))

        policy_uid = uri_policy(f"FPd_SEQ_{new_uid()}")
        rule_uid = uri_rule(f"SEQ_{conn.from_activity}_{conn.to_activity}_{new_uid()}")
        return template_fpd_flow_sequence(
            policy_uid=policy_uid,
            fragment_id=fragment_id,
            from_activity=conn.from_activity,
            to_activity=conn.to_activity,
            rule_source_uri=ruleij,
            downstream_fpa_uid=policy_ik,
            permission_rule_uid=rule_uid,
        )

    def _generate_fpd_message(self, conn, fragment_id: str) -> dict:
        ruleix    = self._activity_rule_index.get(
            conn.from_activity, uri_rule(f"rule_{conn.from_activity.replace('-','_')}"))
        rulejy    = self._activity_rule_index.get(
            conn.to_activity, uri_rule(f"rule_{conn.to_activity.replace('-','_')}"))
        from_frag = getattr(conn, "from_fragment", fragment_id)
        to_frag   = getattr(conn, "to_fragment",   "unknown")

        policy_uid = uri_policy(f"FPd_MSG_{new_uid()}")
        rule_uid = uri_rule(
            f"MSG_{conn.from_activity}_{conn.to_activity}_{new_uid()}"
        )
        return template_fpd_message(
            policy_uid=policy_uid,
            fragment_id=fragment_id,
            from_frag=from_frag,
            to_frag=to_frag,
            from_activity=conn.from_activity,
            to_activity=conn.to_activity,
            rule_source_uri=ruleix,
            rule_target_uri=rulejy,
            permission_rule_uid=rule_uid,
        )