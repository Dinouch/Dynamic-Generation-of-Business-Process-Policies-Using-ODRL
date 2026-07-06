"""
constraint_validator.py — Agent 3: Constraint Validator

Validates unmapped-case proposals from the exception handling agent, resolves B2P mapping ambiguity (LLM),
runs semantic validation on policies from Agent 4, and routes messages in the pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from openai import AzureOpenAI, OpenAI

from ..structural_analyzer import (
    AgentMessage,
    EnrichedGraph,
    MessageType,
    SemanticHint,
)
from ..exception_handling_agent import UnmappedCaseFormulator, UnmappedCaseProposal
from .semantic_deterministic_validation import (
    merge_semantic_hints,
    run_deterministic_semantic_validation,
)
from .semantic_llm_validation import run_business_semantic_llm_validation


def _build_semantic_correction_payload(
    hints: list[SemanticHint],
    warnings: list[str],
    validation_reports: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Structured report for Agent 4: business errors, ODRL paths, per-policy reports.
    """
    lines = [
        "=== Agent 3 report — business semantic validation (excluding JSON-LD syntax) ===",
        "Goal: business intent ↔ BPMN ↔ ODRL alignment; no syntax correction.",
        "",
    ]

    reports = validation_reports or []
    if reports:
        lines.append("=== Per-policy verdicts (business LLM validator) ===")
        for rep in reports:
            uid = rep.get("policy_uid", "?")
            verdict = rep.get("verdict", "?")
            lines.append(f"--- {uid} → {verdict} ---")
            if rep.get("business_intent"):
                lines.append(f"    business_intent: {rep['business_intent']}")
            for err in rep.get("errors") or []:
                lines.append(f"    [ERROR] {err}")
            for w in rep.get("warnings") or []:
                lines.append(f"    [WARN] {w}")
            lines.append("")

    lines.append("=== Structured hints (Agent 4 repair) ===")
    for i, h in enumerate(hints, 1):
        lines.append(f"[{i}] policy_uid={h.policy_uid}")
        lines.append(f"    field_path: {h.field_path or '(policy-level)'}")
        lines.append(f"    issue: {h.issue}")
        if h.suggested_fix:
            lines.append(f"    suggested_fix: {h.suggested_fix}")
        lines.append(f"    category: {h.odrl_template_key}")
        lines.append(f"    confidence: {h.confidence}")
        lines.append("")

    if warnings:
        lines.append("=== Global warnings ===")
        for w in warnings:
            lines.append(f"  - {w}")

    summary = "\n".join(lines)
    utterance = (
        f"Business semantic validation: {len(hints)} blocking issue(s) on "
        f"{len({h.policy_uid for h in hints})} policy uid(s). "
        "Repair business meaning and ODRL conceptual use; resubmit for semantic audit."
    )
    return {
        "semantic_correction_summary": summary,
        "semantic_warnings": list(warnings),
        "semantic_validation_reports": list(reports),
        "utterance": utterance,
    }


class ValidationDecision(Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REFORMULATE = "reformulate"
    STRUCTURAL_ERROR = "structural_error"


class RejectionReason(Enum):
    SELF_REFERENCE = "self_reference"
    DUPLICATE_EXPLICIT = "duplicate_explicit"
    B2P_CONTRADICTION = "b2p_contradiction"
    LLM_REJECTED = "llm_rejected"
    LOW_CONFIDENCE = "low_confidence"
    INVALID_RULE_TYPE = "invalid_rule_type"


@dataclass
class ValidationResult:
    """Outcome for one unmapped proposal (or structural error placeholder)."""

    proposal: Optional[UnmappedCaseProposal]
    decision: ValidationDecision
    decision_level: str
    reason: Optional[RejectionReason] = None
    explanation: str = ""
    llm_raw_response: Optional[str] = None
    reformulation_hint: Optional[str] = None
    hint: Optional[str] = None
    description: str = ""

    @property
    def is_accepted(self) -> bool:
        return self.decision == ValidationDecision.ACCEPTED


@dataclass
class ValidationReport:
    """Aggregate report for Agent 3 → Agent 4."""

    results: list[ValidationResult]
    accepted_unmapped_proposals: list[UnmappedCaseProposal] = field(default_factory=list)
    semantic_warnings: list[str] = field(default_factory=list)
    accepted: list[ValidationResult] = field(default_factory=list)
    rejected: list[ValidationResult] = field(default_factory=list)
    reformulate: list[ValidationResult] = field(default_factory=list)
    structural_errors: list[ValidationResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.accepted = [r for r in self.results if r.is_accepted]
        self.rejected = [r for r in self.results if r.decision == ValidationDecision.REJECTED]
        self.reformulate = [r for r in self.results if r.decision == ValidationDecision.REFORMULATE]
        self.structural_errors = [
            r for r in self.results if r.decision == ValidationDecision.STRUCTURAL_ERROR
        ]
        self.accepted_unmapped_proposals = [
            r.proposal for r in self.accepted if r.proposal is not None
        ]


class ConstraintValidator:
    """
    Agent 3 — B2P ambiguity resolution, unmapped proposal validation,
    semantic validation of generated ODRL (after the first A5 syntax pass).
    """

    MODEL = "gpt-4o"
    TEMPERATURE = 0.1

    AGENT_NAME = "agent3"
    MAX_REFORMULATE = 3
    MAX_SEMANTIC_LOOPS = 4

    def __init__(
        self,
        enriched_graph: Optional[EnrichedGraph] = None,
        api_key: Optional[str] = None,
        min_confidence: float = 0.70,
        *,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
    ):
        self.enriched_graph = enriched_graph
        self.min_confidence = min_confidence
        self._use_azure = False
        self._deployment: Optional[str] = None

        self._on_send_exception_handling: Optional[Callable[[AgentMessage], None]] = None
        self._on_send_agent4: Optional[Callable[[AgentMessage], None]] = None

        self._last_enriched_graph: Optional[EnrichedGraph] = None
        self._reformulate_sent_count: dict[tuple[str, str, str], int] = {}
        self._semantic_loop_count = 0
        self._last_semantic_validation_reports: list[dict[str, Any]] = []
        # Open Contract-Net PROPOSE envelope id (persists across REFORMULATED inform replies).
        self._last_open_propose_id: Optional[str] = None

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
            if not key:
                raise ValueError(
                    "Missing API key. Set OPENAI_API_KEY or Azure OpenAI environment variables."
                )
            self.client = OpenAI(api_key=key)

    def register_send_callback_exception_handling(self, fn: Callable[[AgentMessage], None]) -> None:
        """Register the exception handling agent for ``REFORMULATE`` (unmapped path)."""
        self._on_send_exception_handling = fn

    def register_send_callback_agent4(self, fn: Callable[[AgentMessage], None]) -> None:
        """Register Agent 4 for ``VALIDATION_DONE``, ``SEMANTIC_CORRECTION``, ``SEMANTIC_VALIDATED``."""
        self._on_send_agent4 = fn

    def send(self, msg: AgentMessage) -> None:
        """Route outbound messages by ``recipient``."""
        print(f"[Agent 3] ► SEND {msg}")
        routes = {
            UnmappedCaseFormulator.AGENT_NAME: self._on_send_exception_handling,
            "agent4": self._on_send_agent4,
        }
        fn = routes.get(msg.recipient)
        if fn:
            fn(msg)
        else:
            print(f"[Agent 3][WARN] No callback for recipient '{msg.recipient}'.")

    def receive(self, msg: AgentMessage) -> None:
        """
        Handle ``GRAPH_READY``, ``UNMAPPED_PROPOSALS``, or ``POLICIES_READY``.

        Parameters
        ----------
        msg
            Incoming pipeline message.
        """
        print(f"[Agent 3] ◄ RECEIVE {msg}")
        if msg.msg_type == MessageType.DELEGATION_AGREE:
            print("[Agent 3] Received delegation AGREE (informational).")
            return
        if msg.msg_type == MessageType.GRAPH_READY:
            self._handle_graph_ready(msg)
        elif msg.msg_type in (
            MessageType.UNMAPPED_PROPOSALS,
            MessageType.REFORMULATED_PROPOSALS,
        ):
            self._handle_unmapped_proposals(msg)
        elif msg.msg_type == MessageType.POLICIES_READY:
            self._handle_policies_ready(msg)
        else:
            print(f"[Agent 3][WARN] Unknown message type '{msg.msg_type.value}'.")

    def _emit_accept_proposal_batch(self, msg: AgentMessage, utterance: str) -> None:
        pid = msg.payload.get("proposal_message_id") or self._last_open_propose_id
        if not pid or not self._on_send_exception_handling:
            return
        self.send(
            AgentMessage(
                sender=self.AGENT_NAME,
                recipient=UnmappedCaseFormulator.AGENT_NAME,
                msg_type=MessageType.ACCEPT_PROPOSAL_BATCH,
                payload={"utterance": utterance, "acl_in_reply_to": pid},
                loop_turn=msg.loop_turn,
            )
        )

    def _emit_reject_proposal_batch(self, msg: AgentMessage, reason: str) -> None:
        pid = msg.payload.get("proposal_message_id") or self._last_open_propose_id
        if not pid or not self._on_send_exception_handling:
            return
        self.send(
            AgentMessage(
                sender=self.AGENT_NAME,
                recipient=UnmappedCaseFormulator.AGENT_NAME,
                msg_type=MessageType.REJECT_PROPOSAL_BATCH,
                payload={
                    "utterance": reason[:500],
                    "acl_in_reply_to": pid,
                    "action": "reject-unmapped-proposals",
                    "reason": reason[:2000],
                },
                loop_turn=msg.loop_turn,
            )
        )

    def _handle_graph_ready(self, msg: AgentMessage) -> None:
        """
        Entry from Agent 1. If uncovered patterns exist, wait for proposals from
        the exception handling agent (CFP emitted by Agent 1). Otherwise B2P resolution and ``VALIDATION_DONE``.
        """
        enriched: EnrichedGraph = msg.payload["enriched_graph"]
        self._last_enriched_graph = enriched
        unmapped = list(getattr(enriched, "unmapped_patterns", []) or [])

        if unmapped:
            print(
                f"[Agent 3] {len(unmapped)} unmapped pattern(s) — "
                "waiting for proposals from the exception handling agent (CFP emitted by Agent 1). "
                "I5 orchestrator (accept_all) will continue upon UNMAPPED_PROPOSALS."
            )
            return

        self._resolve_b2p_ambiguity_llm(enriched)
        report = ValidationReport(
            results=[],
            accepted_unmapped_proposals=[],
            semantic_warnings=[],
        )
        self.send(
            AgentMessage(
                sender=self.AGENT_NAME,
                recipient="agent4",
                msg_type=MessageType.VALIDATION_DONE,
                payload={
                    "validation_report": report,
                    "enriched_graph": enriched,
                },
                loop_turn=msg.loop_turn,
            )
        )

    def _handle_unmapped_proposals(self, msg: AgentMessage) -> None:
        """
        Process proposals from the exception handling agent: B2P ambiguity (T1), validate proposals,
        then emit ``VALIDATION_DONE`` or ``REFORMULATE`` (no graph mutation).
        """
        pid = (msg.payload or {}).get("proposal_message_id")
        if pid:
            self._last_open_propose_id = pid

        enriched: EnrichedGraph = msg.payload["enriched_graph"]
        raw_props = msg.payload.get("unmapped_proposals") or []
        proposals = [UnmappedCaseProposal.from_dict(d) for d in raw_props]
        self._last_enriched_graph = enriched

        self._resolve_b2p_ambiguity_llm(enriched)

        results: list[ValidationResult] = []
        for p in proposals:
            self._normalize_unmapped_proposal(p)
            results.append(self._llm_judge_unmapped(p, enriched))

        report = ValidationReport(results=results)

        n_acc = len(report.accepted_unmapped_proposals)
        n_rej = len(report.rejected)
        print(
            f"[Agent 3] Unmapped proposals report: {n_acc} accepted, "
            f"{n_rej} rejected, {len(report.reformulate)} reformulation(s)."
        )
        for r in results:
            if r.proposal:
                tag = r.proposal.pattern_type
                extra = ""
                if r.proposal.involved_fragment_ids:
                    extra = f" [fragments: {', '.join(r.proposal.involved_fragment_ids)}]"
                det = ""
                if r.reason and r.decision == ValidationDecision.REJECTED:
                    det = f" — {r.reason.value}"
                if r.explanation and r.decision == ValidationDecision.REJECTED:
                    det += f": {r.explanation[:200]}"
                print(
                    f"[Agent 3]   · {tag}{extra} → {r.decision.value} "
                    f"({r.decision_level}){det}"
                )

        if report.structural_errors:
            bits = [
                (vr.description or vr.explanation or "").strip()
                for vr in report.structural_errors[:8]
            ]
            bits = [b for b in bits if b]
            rej_reason = (
                "Structural issue detected in unmapped proposals; "
                "BPMN input is not modified. "
                + (" | ".join(bits)[:1900] if bits else "See validation report.")
            )
            self._emit_reject_proposal_batch(msg, rej_reason)
            print(
                "[Agent 3] Structural error(s) — reject-proposal to exception handling agent; "
                "input graph unchanged."
            )
            return

        for vr in report.reformulate:
            if not vr.proposal:
                continue
            key = (
                vr.proposal.fragment_id,
                vr.proposal.pattern_type,
                vr.proposal.gateway_name,
            )
            if self._reformulate_sent_count.get(key, 0) >= self.MAX_REFORMULATE:
                continue
            self._reformulate_sent_count[key] = self._reformulate_sent_count.get(key, 0) + 1
            if vr.proposal and self._on_send_exception_handling:
                rreason = vr.reformulation_hint or vr.explanation or "Reformulation requested for unmapped proposals."
                self._emit_reject_proposal_batch(msg, rreason)
                self.send(
                    AgentMessage(
                        sender=self.AGENT_NAME,
                        recipient=UnmappedCaseFormulator.AGENT_NAME,
                        msg_type=MessageType.REFORMULATE,
                        payload={
                            "hint": vr.reformulation_hint or vr.explanation,
                            "pattern_type": vr.proposal.pattern_type,
                            "gateway_name": vr.proposal.gateway_name,
                            "fragment_id": vr.proposal.fragment_id,
                            "utterance": "Please reformulate the proposal for this pattern with the given hint.",
                        },
                        loop_turn=msg.loop_turn,
                    )
                )
                return

        print("[Agent 3] Emitting VALIDATION_DONE to Agent 4")
        self._emit_accept_proposal_batch(
            msg,
            "Proposal batch accepted; proceeding to fragment-policy projection and semantic audit.",
        )
        self.send(
            AgentMessage(
                sender=self.AGENT_NAME,
                recipient="agent4",
                msg_type=MessageType.VALIDATION_DONE,
                payload={
                    "validation_report": report,
                    "enriched_graph": enriched,
                },
                loop_turn=msg.loop_turn,
            )
        )

    def _handle_policies_ready(self, msg: AgentMessage) -> None:
        """
        Semantic validation (policies already passed A5 syntax audit).
        On success, emit ``SEMANTIC_VALIDATED``; otherwise ``SEMANTIC_CORRECTION`` up to ``MAX_SEMANTIC_LOOPS``.
        """
        fp_results = msg.payload["fp_results"]
        enriched = msg.payload["enriched_graph"]
        self._last_enriched_graph = enriched

        n_pol = sum(len(fps.all_policies()) for fps in (fp_results or {}).values())
        print(
            f"[Agent 3] POLICIES_READY — semantic validation of {n_pol} policy/policies "
            f"(deterministic + business LLM on FPd/unmapped)…"
        )
        ok, hints, warnings = self._semantic_validate_policies(fp_results, enriched)
        print(
            f"[Agent 3] POLICIES_READY — semantic result: "
            f"{'OK' if ok else f'FAILURE ({len(hints)} hint(s))'}"
        )
        if ok:
            self._semantic_loop_count = 0
            self.send(
                AgentMessage(
                    sender=self.AGENT_NAME,
                    recipient="agent4",
                    msg_type=MessageType.SEMANTIC_VALIDATED,
                    payload={
                        "fp_results": fp_results,
                        "enriched_graph": enriched,
                        "semantic_warnings": warnings,
                    },
                    loop_turn=msg.loop_turn,
                )
            )
            return

        self._semantic_loop_count += 1
        if self._semantic_loop_count > self.MAX_SEMANTIC_LOOPS:
            print(
                f"[Agent 3][WARN] MAX_SEMANTIC_LOOPS ({self.MAX_SEMANTIC_LOOPS}) — "
                "forwarding with warnings."
            )
            self._semantic_loop_count = 0
            self.send(
                AgentMessage(
                    sender=self.AGENT_NAME,
                    recipient="agent4",
                    msg_type=MessageType.SEMANTIC_VALIDATED,
                    payload={
                        "fp_results": fp_results,
                        "enriched_graph": enriched,
                        "semantic_warnings": warnings
                        + [f"Semantic validation exceeded MAX_SEMANTIC_LOOPS ({self.MAX_SEMANTIC_LOOPS})."],
                    },
                    loop_turn=msg.loop_turn,
                )
            )
            return

        anchor = (msg.payload or {}).get("acl_failure_reply_target")
        if anchor:
            reason_bits = [f"{h.policy_uid}: {h.issue}" for h in hints[:10]]
            reason_txt = ("; ".join(reason_bits))[:2000] or (
                "Fragment policies failed semantic validation against the enriched graph."
            )
            self.send(
                AgentMessage(
                    sender=self.AGENT_NAME,
                    recipient="agent4",
                    msg_type=MessageType.SEMANTIC_VALIDATION_FAILURE,
                    payload={
                        "acl_in_reply_to": anchor,
                        "action": "semantic-validation-failed",
                        "reason": reason_txt,
                        "utterance": "Semantic validation failed; issuing correction request next.",
                        "fp_results": fp_results,
                        "enriched_graph": enriched,
                    },
                    loop_turn=msg.loop_turn,
                )
            )

        extra = _build_semantic_correction_payload(
            hints,
            warnings,
            validation_reports=getattr(self, "_last_semantic_validation_reports", None),
        )
        self.send(
            AgentMessage(
                sender=self.AGENT_NAME,
                recipient="agent4",
                msg_type=MessageType.SEMANTIC_CORRECTION,
                payload={
                    "fp_results": fp_results,
                    "enriched_graph": enriched,
                    "semantic_hints": [asdict(h) for h in hints],
                    "loop_turn": msg.loop_turn + 1,
                    "utterance": extra["utterance"],
                    "semantic_correction_summary": extra["semantic_correction_summary"],
                    "semantic_warnings": extra["semantic_warnings"],
                    "semantic_validation_reports": extra.get("semantic_validation_reports") or [],
                },
                loop_turn=msg.loop_turn + 1,
            )
        )

    def _normalize_unmapped_proposal(self, p: UnmappedCaseProposal) -> None:
        """Force a usable ODRL rule type; the LLM judge decides afterward."""
        rt = (p.odrl_rule_type or "").strip().lower()
        if rt not in ("permission", "prohibition", "obligation"):
            p.odrl_rule_type = "permission"

    def _llm_judge_unmapped(
        self,
        p: UnmappedCaseProposal,
        graph: EnrichedGraph,
    ) -> ValidationResult:
        """LLM judge for semantic coherence of an unmapped proposal."""
        b2p_json = json.dumps(graph.raw_b2p, ensure_ascii=False)
        user_prompt = f"""You validate an ODRL fragment-policy hint against B2P context.

Exception handling agent self-reported confidence: {p.confidence} (reference threshold {self.min_confidence} — do not reject solely for low confidence if the structural case is real, e.g. BPMN loops).

Proposal:
{json.dumps(p.to_dict(), ensure_ascii=False, indent=2)}

B2P policies JSON:
{b2p_json}

Respond ONLY with JSON:
{{
  "decision": "accepted" | "rejected" | "reformulate" | "structural_error",
  "explanation": "...",
  "reformulation_hint": "..."
}}
"""
        try:
            raw = self._call_llm(
                "You are an ODRL policy validator. Respond only with valid JSON, no markdown.",
                user_prompt,
            )
            data = json.loads(raw)
            d = str(data.get("decision", "reformulate")).lower()
            dec = {
                "accepted": ValidationDecision.ACCEPTED,
                "rejected": ValidationDecision.REJECTED,
                "reformulate": ValidationDecision.REFORMULATE,
                "structural_error": ValidationDecision.STRUCTURAL_ERROR,
            }.get(d, ValidationDecision.REFORMULATE)
            return ValidationResult(
                proposal=p,
                decision=dec,
                decision_level="llm",
                explanation=str(data.get("explanation", "")),
                reformulation_hint=data.get("reformulation_hint"),
                hint=data.get("reformulation_hint"),
                description=str(data.get("explanation", "")),
                llm_raw_response=raw,
            )
        except Exception as e:
            print(f"[Agent 3][WARN] LLM judge failed: {e}")
            return ValidationResult(
                proposal=p,
                decision=ValidationDecision.ACCEPTED,
                decision_level="llm",
                explanation=f"Accepted after judge error: {e}",
            )

    def _resolve_b2p_ambiguity_llm(self, graph: EnrichedGraph) -> None:
        """
        Phase 1 — resolve ambiguous B2P mappings using the LLM judge (mutates ``graph.b2p_mappings``).
        """
        ambiguous = self._collect_ambiguous_mappings(graph)
        if not ambiguous:
            return
        b2p_json = json.dumps(graph.raw_b2p, ensure_ascii=False)
        amb_json = json.dumps(ambiguous, ensure_ascii=False, indent=2)
        user_prompt = f"""The following BPMN activities have ambiguous B2P policy mappings.
For each activity, select the single most appropriate B2P policy UID,
or return an empty list if none apply.

Activities with ambiguous mappings: {amb_json}
Available B2P policies: {b2p_json}

Respond ONLY with:
{{
  "decisions": [
    {{
      "activity_name": "...",
      "selected_policy_uid": "..." | null,
      "confidence": 0.0,
      "reason": "..."
    }}
  ]
}}
"""
        try:
            raw = self._call_llm(
                "You are an ODRL and BPMN policy expert. Respond only with valid JSON, no markdown, no preamble.",
                user_prompt,
            )
            data = json.loads(raw)
            self._apply_b2p_decisions(graph, data.get("decisions") or [])
        except Exception as e:
            print(f"[Agent 3][WARN] B2P ambiguity resolution failed: {e}")

    def _collect_ambiguous_mappings(self, graph: EnrichedGraph) -> list[dict]:
        """Build list of activities with ambiguous B2P mappings."""
        ambiguous: list[dict] = []
        policy_to_activities: dict[str, list[str]] = {}
        for m in graph.b2p_mappings.values():
            if len(m.b2p_policy_ids) > 1:
                ambiguous.append(
                    {
                        "activity_name": m.activity_name,
                        "fragment_id": m.fragment_id,
                        "candidate_policy_uids": list(m.b2p_policy_ids),
                    }
                )
            for uid in m.b2p_policy_ids:
                policy_to_activities.setdefault(uid, []).append(m.activity_name)
        for uid, names in policy_to_activities.items():
            if len(set(names)) > 1:
                ambiguous.append(
                    {
                        "policy_uid_collision": uid,
                        "activities": list(set(names)),
                    }
                )
        return ambiguous

    def _apply_b2p_decisions(self, graph: EnrichedGraph, decisions: list[dict]) -> None:
        """Apply LLM B2P decisions to ``ActivityB2PMapping`` entries."""
        by_name = {m.activity_name: m for m in graph.b2p_mappings.values()}
        valid_uids = {p.get("uid") for p in graph.raw_b2p if p.get("uid")}
        for d in decisions:
            if not isinstance(d, dict):
                continue
            name = d.get("activity_name")
            if not name or name not in by_name:
                continue
            try:
                conf = float(d.get("confidence", 0))
            except (TypeError, ValueError):
                conf = 0.0
            if conf < 0.52:
                continue
            sel = d.get("selected_policy_uid")
            m = by_name[name]
            if sel in valid_uids:
                m.b2p_policy_ids = [sel]
                m.rule_types = self._rule_types_for_uids([sel], graph.raw_b2p)
            elif sel is None or sel == "":
                m.b2p_policy_ids = []
                m.rule_types = []

    def _rule_types_for_uids(self, uids: list[str], raw_b2p: list[dict]) -> list[str]:
        """Collect ODRL rule types present for the given policy UIDs."""
        types: list[str] = []
        for pol in raw_b2p:
            if pol.get("uid") not in uids:
                continue
            for rt in ("permission", "prohibition", "obligation"):
                if pol.get(rt) and rt not in types:
                    types.append(rt)
        return types

    def _semantic_validate_policies(self, fp_results: dict, graph: EnrichedGraph) -> tuple:
        """
        Semantic validation: deterministic layer (vocabulary, FPa↔B2P, batch)
        then business LLM judge (meaning, BPMN, intent — excluding JSON-LD syntax).

        Returns
        -------
        is_valid, semantic_hints, warnings
        """
        det_hints, det_warnings = run_deterministic_semantic_validation(fp_results, graph)
        llm_hints, llm_warnings, llm_reports = self._semantic_validate_policies_llm(
            fp_results, graph
        )
        self._last_semantic_validation_reports = llm_reports
        merged = merge_semantic_hints(det_hints, llm_hints)
        warnings = list(det_warnings) + list(llm_warnings)

        if merged:
            uids = list({h.get("policy_uid") for h in merged if h.get("policy_uid")})
            print(
                f"[Agent 3] Semantic validation: {len(merged)} hint(s) "
                f"(deterministic + LLM) on {len(uids)} policy uid(s) — {uids[:6]}"
                f"{' …' if len(uids) > 6 else ''}"
            )
            hints_out = [
                SemanticHint(
                    policy_uid=str(h.get("policy_uid", "")),
                    field_path=str(h.get("field_path", "")),
                    issue=str(h.get("issue", "")),
                    suggested_fix=str(h.get("suggested_fix", "")),
                    odrl_template_key=str(h.get("odrl_template_key", "")),
                    confidence=float(h.get("confidence", 0.0)),
                )
                for h in merged
            ]
            return False, hints_out, warnings
        return True, [], warnings

    def _semantic_validate_policies_llm(
        self, fp_results: dict, graph: EnrichedGraph
    ) -> tuple[list[dict], list[str], list[dict]]:
        """
        Business LLM judge for each generated policy (FPa + FPd).

        Returns
        -------
        hint_dicts, warnings, validation_reports
        """
        if not self.client:
            return [], ["Business semantic LLM validation skipped (no API client)."], []

        hints, warnings, reports = run_business_semantic_llm_validation(
            fp_results,
            graph,
            call_llm=self._call_llm,
        )
        if hints:
            uids = sorted({h.get("policy_uid") for h in hints if h.get("policy_uid")})
            print(
                f"[Agent 3] Business LLM semantic validation: {len(hints)} error(s) "
                f"on {len(uids)} policy/policies — {uids[:5]}{' …' if len(uids) > 5 else ''}"
            )
        return hints, warnings, reports

    def _call_llm(self, system: str, user: str) -> str:
        """Invoke the configured chat model."""
        model = self._deployment if self._use_azure else self.MODEL
        kwargs = {
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
            raise RuntimeError("Empty LLM response")
        return content

    def validate(
        self,
        enriched_graph: EnrichedGraph,
        proposals: Optional[list] = None,
    ) -> ValidationReport:
        """
        Standalone API: B2P ambiguity resolution plus optional unmapped proposal validation.

        Parameters
        ----------
        enriched_graph
            Output of Agent 1.
        proposals
            ``UnmappedCaseProposal`` instances or dicts; omit or pass ``[]`` to skip.

        Returns
        -------
        ValidationReport
            Full validation report for Agent 4.
        """
        raw_list = proposals or []
        self.enriched_graph = enriched_graph
        self._resolve_b2p_ambiguity_llm(enriched_graph)
        results: list[ValidationResult] = []
        for raw in raw_list:
            p = (
                UnmappedCaseProposal.from_dict(raw)
                if isinstance(raw, dict)
                else raw
            )
            self._normalize_unmapped_proposal(p)
            results.append(self._llm_judge_unmapped(p, enriched_graph))
        return ValidationReport(results=results)
