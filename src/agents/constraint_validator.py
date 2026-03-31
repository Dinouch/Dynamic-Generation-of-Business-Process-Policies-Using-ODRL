"""
constraint_validator.py — Agent 3 : Constraint Validator

Validates unsupported-case proposals from Agent 2, resolves B2P mapping ambiguity (LLM),
runs semantic validation on policies from Agent 4, and routes messages in the pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Optional

from openai import AzureOpenAI, OpenAI

from .structural_analyzer import (
    AgentMessage,
    EnrichedGraph,
    MessageType,
    SemanticHint,
)
from .unsupported_case_formulator import UnsupportedCaseProposal


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
    """Outcome for one unsupported proposal (or structural error placeholder)."""

    proposal: Optional[UnsupportedCaseProposal]
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
    accepted_unsupported_proposals: list[UnsupportedCaseProposal] = field(default_factory=list)
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
        self.accepted_unsupported_proposals = [
            r.proposal for r in self.accepted if r.proposal is not None
        ]


class ConstraintValidator:
    """
    Agent 3 — B2P ambiguity resolution, unsupported proposal validation,
    semantic validation of generated ODRL (Time 2).
    """

    MODEL = "gpt-4o"
    TEMPERATURE = 0.1

    AGENT_NAME = "agent3"
    MAX_STRUCTURAL_LOOPS = 1
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

        self._on_send_agent2: Optional[Callable[[AgentMessage], None]] = None
        self._on_send_agent1: Optional[Callable[[AgentMessage], None]] = None
        self._on_send_agent4: Optional[Callable[[AgentMessage], None]] = None

        self._structural_loop_count = 0
        self._last_enriched_graph: Optional[EnrichedGraph] = None
        self._reformulate_sent_count: dict[tuple[str, str, str], int] = {}
        self._semantic_loop_count = 0

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

    def register_send_callback_agent2(self, fn: Callable[[AgentMessage], None]) -> None:
        """Register Agent 2 for ``REFORMULATE`` and ``GRAPH_READY`` (unsupported path)."""
        self._on_send_agent2 = fn

    def register_send_callback_agent1(self, fn: Callable[[AgentMessage], None]) -> None:
        """Register Agent 1 for ``STRUCTURAL_UPDATE``."""
        self._on_send_agent1 = fn

    def register_send_callback_agent4(self, fn: Callable[[AgentMessage], None]) -> None:
        """Register Agent 4 for ``VALIDATION_DONE``, ``SEMANTIC_CORRECTION``, ``SEMANTIC_VALIDATED``."""
        self._on_send_agent4 = fn

    def send(self, msg: AgentMessage) -> None:
        """Route outbound messages by ``recipient``."""
        print(f"[Agent 3] ► SEND {msg}")
        routes = {
            "agent2": self._on_send_agent2,
            "agent1": self._on_send_agent1,
            "agent4": self._on_send_agent4,
        }
        fn = routes.get(msg.recipient)
        if fn:
            fn(msg)
        else:
            print(f"[Agent 3][WARN] No callback for recipient '{msg.recipient}'.")

    def receive(self, msg: AgentMessage) -> None:
        """
        Handle ``GRAPH_READY``, ``UNSUPPORTED_PROPOSALS``, or ``POLICIES_READY``.

        Parameters
        ----------
        msg
            Incoming pipeline message.
        """
        print(f"[Agent 3] ◄ RECEIVE {msg}")
        if msg.msg_type == MessageType.GRAPH_READY:
            self._handle_graph_ready(msg)
        elif msg.msg_type == MessageType.UNSUPPORTED_PROPOSALS:
            self._handle_unsupported_proposals(msg)
        elif msg.msg_type == MessageType.POLICIES_READY:
            self._handle_policies_ready(msg)
        else:
            print(f"[Agent 3][WARN] Unknown message type '{msg.msg_type.value}'.")

    def _handle_graph_ready(self, msg: AgentMessage) -> None:
        """
        Entry from Agent 1. Forwards to Agent 2 if there are unsupported patterns;
        otherwise resolves B2P ambiguity and emits ``VALIDATION_DONE``.
        """
        enriched: EnrichedGraph = msg.payload["enriched_graph"]
        self._last_enriched_graph = enriched
        unsupported = list(getattr(enriched, "unsupported_patterns", []) or [])

        if unsupported and self._on_send_agent2:
            print(f"[Agent 3] {len(unsupported)} unsupported pattern(s) — forwarding GRAPH_READY to Agent 2")
            self.send(
                AgentMessage(
                    sender=self.AGENT_NAME,
                    recipient="agent2",
                    msg_type=MessageType.GRAPH_READY,
                    payload={"enriched_graph": enriched},
                    loop_turn=msg.loop_turn,
                )
            )
            return

        self._resolve_b2p_ambiguity_llm(enriched)
        report = ValidationReport(
            results=[],
            accepted_unsupported_proposals=[],
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

    def _handle_unsupported_proposals(self, msg: AgentMessage) -> None:
        """
        Process proposals from Agent 2: B2P ambiguity (T1), validate proposals,
        then emit ``VALIDATION_DONE`` or ``REFORMULATE`` / ``STRUCTURAL_UPDATE``.
        """
        enriched: EnrichedGraph = msg.payload["enriched_graph"]
        raw_props = msg.payload.get("unsupported_proposals") or []
        proposals = [UnsupportedCaseProposal.from_dict(d) for d in raw_props]
        self._last_enriched_graph = enriched

        self._resolve_b2p_ambiguity_llm(enriched)

        results: list[ValidationResult] = []
        for p in proposals:
            self._normalize_unsupported_proposal(p)
            results.append(self._llm_judge_unsupported(p, enriched))

        report = ValidationReport(results=results)

        n_acc = len(report.accepted_unsupported_proposals)
        n_rej = len(report.rejected)
        print(
            f"[Agent 3] Rapport propositions unsupported : {n_acc} acceptée(s), "
            f"{n_rej} rejetée(s), {len(report.reformulate)} reformulation(s)."
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

        for vr in report.structural_errors:
            if self._structural_loop_count < self.MAX_STRUCTURAL_LOOPS:
                self._structural_loop_count += 1
                edge_dict = self._extract_edge_from_hint(vr.hint or vr.reformulation_hint or "")
                self.send(
                    AgentMessage(
                        sender=self.AGENT_NAME,
                        recipient="agent1",
                        msg_type=MessageType.STRUCTURAL_UPDATE,
                        payload={
                            "reason": vr.description or vr.explanation,
                            "affected_fragments": [p.fragment_id for p in proposals if proposals],
                            "hint": vr.hint or vr.reformulation_hint,
                            "implicit_edges_to_add": [edge_dict] if edge_dict else [],
                        },
                        loop_turn=msg.loop_turn + 1,
                    )
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
            if vr.proposal and self._on_send_agent2:
                self.send(
                    AgentMessage(
                        sender=self.AGENT_NAME,
                        recipient="agent2",
                        msg_type=MessageType.REFORMULATE,
                        payload={
                            "hint": vr.reformulation_hint or vr.explanation,
                            "pattern_type": vr.proposal.pattern_type,
                            "gateway_name": vr.proposal.gateway_name,
                            "fragment_id": vr.proposal.fragment_id,
                        },
                        loop_turn=msg.loop_turn,
                    )
                )
                return

        print("[Agent 3] Emitting VALIDATION_DONE to Agent 4")
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
        Semantic validation (Time 2). On success, emit ``SEMANTIC_VALIDATED``;
        on failure emit ``SEMANTIC_CORRECTION`` until ``MAX_SEMANTIC_LOOPS``.
        """
        fp_results = msg.payload["fp_results"]
        enriched = msg.payload["enriched_graph"]
        self._last_enriched_graph = enriched

        ok, hints, warnings = self._semantic_validate_policies(fp_results, enriched)
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
                },
                loop_turn=msg.loop_turn + 1,
            )
        )

    def _normalize_unsupported_proposal(self, p: UnsupportedCaseProposal) -> None:
        """Force un type de règle ODRL utilisable ; le juge LLM tranche ensuite."""
        rt = (p.odrl_rule_type or "").strip().lower()
        if rt not in ("permission", "prohibition", "obligation"):
            p.odrl_rule_type = "permission"

    def _llm_judge_unsupported(
        self,
        p: UnsupportedCaseProposal,
        graph: EnrichedGraph,
    ) -> ValidationResult:
        """LLM judge for semantic coherence of an unsupported proposal."""
        b2p_json = json.dumps(graph.raw_b2p, ensure_ascii=False)
        user_prompt = f"""You validate an ODRL fragment-policy hint against B2P context.

Agent 2 self-reported confidence: {p.confidence} (reference threshold {self.min_confidence} — do not reject solely for low confidence if the structural case is real, e.g. BPMN loops).

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
        Time 1 — resolve ambiguous B2P mappings using the LLM judge (mutates ``graph.b2p_mappings``).
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
        Run LLM semantic audit on generated policies (sampled / batched).

        Returns
        -------
        is_valid, semantic_hints (dicts), warnings (str)
        """
        warnings: list[str] = []
        all_hints: list[dict] = []
        for _frag_id, fps in fp_results.items():
            for pol in fps.all_policies():
                src = pol.get("_source_b2p")
                if not src:
                    continue
                b2p = next((p for p in graph.raw_b2p if p.get("uid") == src), None)
                if not b2p:
                    continue
                user_prompt = f"""Validate the following generated ODRL policy against its B2P source.

B2P source policy: {json.dumps(b2p, ensure_ascii=False)}
Generated ODRL policy: {json.dumps({k: v for k, v in pol.items() if not str(k).startswith('_')}, ensure_ascii=False)}

Check specifically:
1. Is the rule type (permission/prohibition/obligation) correct?
2. Are constraint operators correct (gt/gteq/lt/lteq/eq/neq)?
3. Is the temporal constraint correctly expressed?
4. Is the assigner/assignee preserved if present?
5. Are there any hallucinated constraints not in the B2P source?

For "suggested_fix": use ONLY the literal value to apply (e.g. gteq, eq, or a full http(s) IRI).
Do not write sentences like "set uid to ..." — put the IRI alone when fixing uid or target.

Respond ONLY with:
{{
  "is_valid": true | false,
  "issues": [
    {{
      "field_path": "permission[0].constraint[0].operator",
      "issue": "operator is gt but should be gteq",
      "suggested_fix": "gteq",
      "odrl_template_key": "constraint_operator",
      "confidence": 0.0
    }}
  ]
}}
"""
                try:
                    raw = self._call_llm(
                        "You are an ODRL policy auditor. You validate whether generated ODRL policies faithfully capture their source B2P rules. Respond only with valid JSON, no markdown, no preamble.",
                        user_prompt,
                    )
                    data = json.loads(raw)
                    if not data.get("is_valid", True):
                        for issue in data.get("issues") or []:
                            all_hints.append(
                                {
                                    "policy_uid": pol.get("uid"),
                                    "field_path": issue.get("field_path", ""),
                                    "issue": issue.get("issue", ""),
                                    "suggested_fix": issue.get("suggested_fix", ""),
                                    "odrl_template_key": issue.get("odrl_template_key", ""),
                                    "confidence": float(issue.get("confidence", 0.0)),
                                }
                            )
                except Exception as e:
                    warnings.append(f"Semantic check skipped for {pol.get('uid')}: {e}")

        if all_hints:
            uids = list({h.get("policy_uid") for h in all_hints if h.get("policy_uid")})
            print(
                f"[Agent 3] Validation sémantique (B2P ↔ ODRL) : {len(all_hints)} "
                f"hint(s) sur {len(uids)} policy uid(s) — {uids[:6]}"
                f"{' …' if len(uids) > 6 else ''}"
            )
            hints_out = [
                SemanticHint(
                    policy_uid=h["policy_uid"],
                    field_path=h["field_path"],
                    issue=h["issue"],
                    suggested_fix=h["suggested_fix"],
                    odrl_template_key=h["odrl_template_key"],
                    confidence=h["confidence"],
                )
                for h in all_hints
            ]
            return False, hints_out, warnings
        return True, [], warnings

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
        Standalone API: B2P ambiguity resolution plus optional unsupported proposal validation.

        Parameters
        ----------
        enriched_graph
            Output of Agent 1.
        proposals
            ``UnsupportedCaseProposal`` instances or dicts; omit or pass ``[]`` to skip.

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
                UnsupportedCaseProposal.from_dict(raw)
                if isinstance(raw, dict)
                else raw
            )
            self._normalize_unsupported_proposal(p)
            results.append(self._llm_judge_unsupported(p, enriched_graph))
        return ValidationReport(results=results)

    def _extract_edge_from_hint(self, hint: str) -> Optional[dict]:
        """Parse optional JSON edge description from a structural hint string."""
        if not hint:
            return None
        try:
            return json.loads(hint)
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            start = hint.index("{")
            end = hint.rindex("}") + 1
            return json.loads(hint[start:end])
        except (ValueError, json.JSONDecodeError):
            return None
