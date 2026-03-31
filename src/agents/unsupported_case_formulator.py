"""
unsupported_case_formulator.py — Agent 2 : Unsupported Case Formulator

Turns BPMN structural patterns that have no deterministic ODRL template into
LLM-proposed ODRL hints for Agent 4. Replaces the former implicit dependency
detector.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from openai import AzureOpenAI, OpenAI

from .structural_analyzer import (
    AgentMessage,
    EnrichedGraph,
    MessageType,
    UnsupportedPattern,
)


@dataclass
class UnsupportedCaseProposal:
    """
    One LLM-generated proposal for an unsupported BPMN pattern.

    Attributes
    ----------
    pattern_type
        Same identifier as ``UnsupportedPattern.pattern_type``.
    gateway_name, fragment_id
        Context from structural analysis.
    odrl_rule_type
        ``permission`` | ``prohibition`` | ``obligation``.
    hint_text
        Natural-language policy intent for Agent 4.
    odrl_structure_hint
        Partial skeleton (e.g. action, constraint sketch) for Agent 4.
    confidence
        Self-reported confidence in [0.0, 1.0].
    justification
        Short reasoning from the LLM.
    """

    pattern_type: str
    gateway_name: str
    fragment_id: str
    odrl_rule_type: str
    hint_text: str
    odrl_structure_hint: dict
    confidence: float
    justification: str
    involved_fragment_ids: list[str] = field(default_factory=list)
    involved_activity_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for message payloads."""
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "UnsupportedCaseProposal":
        """Deserialize from a message payload dict."""
        rt = str(d.get("odrl_rule_type", "permission")).strip().lower()
        if rt not in ("permission", "prohibition", "obligation"):
            rt = "permission"
        return UnsupportedCaseProposal(
            pattern_type=d.get("pattern_type", ""),
            gateway_name=d.get("gateway_name", ""),
            fragment_id=d.get("fragment_id", ""),
            odrl_rule_type=rt,
            hint_text=d.get("hint_text", ""),
            odrl_structure_hint=d.get("odrl_structure_hint") or {},
            confidence=float(d.get("confidence", 0.0)),
            justification=d.get("justification", ""),
            involved_fragment_ids=list(d.get("involved_fragment_ids") or []),
            involved_activity_names=list(d.get("involved_activity_names") or []),
        )


class UnsupportedCaseFormulator:
    """
    Agent 2 — formulates ODRL hints for unsupported BPMN patterns.

    Receives ``GRAPH_READY`` from Agent 3 (when ``unsupported_patterns`` is non-empty).
    Emits ``UNSUPPORTED_PROPOSALS`` to Agent 3.
    """

    MODEL = "gpt-4o"
    TEMPERATURE = 0.2

    AGENT_NAME = "agent2"
    MAX_REFORMULATE = 3

    _SYSTEM_PROMPT = (
        "You are a senior ODRL policy architect with deep expertise in BPMN process "
        "modelling and W3C ODRL 2.2. Given a BPMN structural pattern and its business "
        "context, you independently determine the correct ODRL formalisation — including "
        "the number of rules, their types, conditions, and inter-rule dependencies. "
        "You never ask for clarification. You respond only with valid JSON, no markdown, "
        "no preamble."
    )

    def __init__(
        self,
        covered_patterns: Optional[set[str]] = None,
        api_key: Optional[str] = None,
        *,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        enriched_graph: Optional[EnrichedGraph] = None,
    ):
        """
        Initialize the formulator.

        Parameters
        ----------
        covered_patterns
            Ignored at runtime — registry is enforced in Agent 1; kept for wiring compatibility.
        """
        _ = covered_patterns
        self.enriched_graph = enriched_graph
        self._use_azure = False
        self._deployment: Optional[str] = None

        self._on_send: Optional[Callable[[AgentMessage], None]] = None
        self._reformulate_count = 0
        self._last_graph: Optional[EnrichedGraph] = None
        self._last_proposals: list[UnsupportedCaseProposal] = []

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
                or self.MODEL
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
                    "Missing API key. Set OPENAI_API_KEY or Azure OpenAI env vars."
                )
            self.client = OpenAI(api_key=key)

    def register_send_callback(self, fn: Callable[[AgentMessage], None]) -> None:
        """Register callback for outbound messages (typically Agent 3)."""
        self._on_send = fn

    def send(self, msg: AgentMessage) -> None:
        """Emit a message via the registered callback."""
        print(f"[Agent 2] ► SEND {msg}")
        if self._on_send:
            self._on_send(msg)
        else:
            print(f"[Agent 2][WARN] No callback — message {msg.msg_type.value} not sent.")

    def receive(self, msg: AgentMessage) -> None:
        """
        Handle incoming messages.

        Parameters
        ----------
        msg
            ``GRAPH_READY`` or ``REFORMULATE`` from Agent 3.
        """
        print(f"[Agent 2] ◄ RECEIVE {msg}")
        if msg.msg_type == MessageType.GRAPH_READY:
            self._handle_graph_ready(msg)
        elif msg.msg_type == MessageType.REFORMULATE:
            self._handle_reformulate(msg)
        else:
            print(f"[Agent 2][WARN] Unknown message type '{msg.msg_type.value}'.")

    def _handle_graph_ready(self, msg: AgentMessage) -> None:
        """Process ``GRAPH_READY`` and emit ``UNSUPPORTED_PROPOSALS``."""
        enriched_graph: EnrichedGraph = msg.payload["enriched_graph"]
        self._last_graph = enriched_graph
        self._reformulate_count = 0

        unsupported = list(getattr(enriched_graph, "unsupported_patterns", []) or [])
        proposals: list[UnsupportedCaseProposal] = []

        if not unsupported:
            self._emit_unsupported_proposals(enriched_graph, proposals, msg.loop_turn)
            return

        b2p_json = json.dumps(enriched_graph.raw_b2p, ensure_ascii=False)

        for u in unsupported:
            props = self._formulate_one(u, b2p_json)
            if props:
                proposals.extend(props)

        self._last_proposals = proposals
        self._emit_unsupported_proposals(enriched_graph, proposals, msg.loop_turn)

    def _emit_unsupported_proposals(
        self,
        enriched_graph: EnrichedGraph,
        proposals: list[UnsupportedCaseProposal],
        loop_turn: int,
    ) -> None:
        """Send ``UNSUPPORTED_PROPOSALS`` to Agent 3."""
        self.send(
            AgentMessage(
                sender=self.AGENT_NAME,
                recipient="agent3",
                msg_type=MessageType.UNSUPPORTED_PROPOSALS,
                payload={
                    "enriched_graph": enriched_graph,
                    "unsupported_proposals": [p.to_dict() for p in proposals],
                },
                loop_turn=loop_turn,
            )
        )

    def _handle_reformulate(self, msg: AgentMessage) -> None:
        """
        Reformulate one or more proposals after Agent 3 rejection.

        Parameters
        ----------
        msg
            Payload may include ``hint``, ``pattern_type``, ``gateway_name``,
            ``fragment_id`` from Agent 3.
        """
        if self._reformulate_count >= self.MAX_REFORMULATE:
            print(
                f"[Agent 2][WARN] MAX_REFORMULATE ({self.MAX_REFORMULATE}) reached — "
                "re-sending last proposals."
            )
            if self._last_graph:
                self._emit_unsupported_proposals(
                    self._last_graph,
                    self._last_proposals,
                    msg.loop_turn,
                )
            return

        self._reformulate_count += 1
        hint = msg.payload.get("hint", "")
        pt = msg.payload.get("pattern_type", "")
        gw = msg.payload.get("gateway_name", "")
        fid = str(msg.payload.get("fragment_id", ""))

        if not self._last_graph:
            print("[Agent 2][WARN] REFORMULATE without stored graph — ignored.")
            return

        b2p_json = json.dumps(self._last_graph.raw_b2p, ensure_ascii=False)

        for u in self._last_graph.unsupported_patterns:
            if u.pattern_type == pt and u.gateway_name == gw and u.fragment_id == fid:
                up = UnsupportedPattern(
                    pattern_type=u.pattern_type,
                    gateway_id=u.gateway_id,
                    gateway_name=u.gateway_name,
                    fragment_id=u.fragment_id,
                    description=(
                        f"{u.description}\n\nValidator reformulation hint: {hint}"
                    ),
                    bpmn_semantic=u.bpmn_semantic,
                    involved_fragment_ids=list(u.involved_fragment_ids or []),
                    involved_activity_names=list(u.involved_activity_names or []),
                )
                new_props = self._formulate_one(up, b2p_json)
                if new_props:
                    self._last_proposals = [
                        p
                        for p in self._last_proposals
                        if not (
                            p.pattern_type == pt
                            and p.gateway_name == gw
                            and p.fragment_id == fid
                        )
                    ]
                    self._last_proposals.extend(new_props)
                break

        self._emit_unsupported_proposals(
            self._last_graph,
            self._last_proposals,
            msg.loop_turn + 1,
        )

    def _formulate_one(self, u, b2p_json: str) -> list[UnsupportedCaseProposal]:
        """
        Call the LLM once for a single ``UnsupportedPattern``.

        Parameters
        ----------
        u
            ``UnsupportedPattern`` instance from Agent 1.
        b2p_json
            JSON string of B2P policies.

        Returns
        -------
        Zero or more ``UnsupportedCaseProposal`` instances (one per policy in the
        LLM response). Empty when formulation fails and no fallback applies.
        """
        user_prompt = f"""Formalise the following BPMN pattern as one or more ODRL Fragment Policies.

Pattern type: {u.pattern_type}
Gateway / element name: {u.gateway_name}
Principal fragment: {u.fragment_id}
Involved fragments: {', '.join(u.involved_fragment_ids) or 'same as principal'}
Activities involved: {', '.join(u.involved_activity_names) or 'see description'}
BPMN semantic: {u.bpmn_semantic}
Structural description: {u.description}
B2P policies in scope: {b2p_json}

Respond ONLY with this JSON:
{{
  "policies": [
    {{
      "odrl_rule_type": "permission|prohibition|obligation",
      "hint_text": "What this policy expresses in plain English.",
      "odrl_structure_hint": {{
        "action": "...",
        "constraints": [
          {{"leftOperand": "...", "operator": "...", "rightOperand": "..."}}
        ],
        "target_activity": "..."
      }},
      "confidence": 0.0,
      "justification": "Why this rule type and structure."
    }}
  ]
}}
"""
        try:
            raw = self._call_llm(user_prompt)
            data = json.loads(raw)
        except Exception as e:
            print(f"[Agent 2][WARN] LLM/formulation failed for {u.pattern_type}: {e}")
            fb = self._fallback_proposal(u)
            if fb:
                print(
                    f"[Agent 2] Deterministic fallback proposal for "
                    f"'{u.pattern_type}' (no successful LLM call)."
                )
            return [fb] if fb else []

        if not isinstance(data, dict):
            fb = self._fallback_proposal(u)
            return [fb] if fb else []

        policies_data = data.get("policies")
        if not isinstance(policies_data, list):
            policies_data = []

        out: list[UnsupportedCaseProposal] = []
        for item in policies_data:
            if not isinstance(item, dict):
                continue
            rt = str(item.get("odrl_rule_type", "permission")).lower()
            if rt not in ("permission", "prohibition", "obligation"):
                rt = "permission"
            try:
                conf = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))
            out.append(
                UnsupportedCaseProposal(
                    pattern_type=u.pattern_type,
                    gateway_name=u.gateway_name,
                    fragment_id=u.fragment_id,
                    odrl_rule_type=rt,
                    hint_text=str(item.get("hint_text", "")),
                    odrl_structure_hint=item.get("odrl_structure_hint") or {},
                    confidence=conf,
                    justification=str(item.get("justification", "")),
                    involved_fragment_ids=list(u.involved_fragment_ids or []),
                    involved_activity_names=list(u.involved_activity_names or []),
                )
            )

        if not out:
            fb = self._fallback_proposal(u)
            return [fb] if fb else []
        return out

    def _fallback_proposal(self, u: UnsupportedPattern) -> Optional[UnsupportedCaseProposal]:
        """
        When the LLM is unavailable (timeout, quota, network), emit a minimal
        proposal for supported patterns (e.g. ``loop``) so Agents 3/4 can still
        produce a fragment policy.
        """
        if u.pattern_type != "loop":
            return None
        ids = list(u.involved_fragment_ids or [])
        if not ids and u.fragment_id and u.fragment_id != "unknown":
            ids = [u.fragment_id]
        if not ids:
            return None
        return UnsupportedCaseProposal(
            pattern_type=u.pattern_type,
            gateway_name=u.gateway_name,
            fragment_id=u.fragment_id,
            odrl_rule_type="permission",
            hint_text=(
                "Fallback policy for detected control loop "
                "(LLM formulation unavailable)."
            ),
            odrl_structure_hint={"action": {"@id": "odrl:enable"}},
            confidence=0.85,
            justification="Deterministic proposal — no successful LLM call.",
            involved_fragment_ids=ids,
            involved_activity_names=list(u.involved_activity_names or []),
        )

    def _call_llm(self, user_prompt: str) -> str:
        """Invoke OpenAI/Azure with JSON-oriented settings."""
        model = self._deployment if self._use_azure else self.MODEL
        kwargs = {
            "model": model,
            "temperature": self.TEMPERATURE,
            "messages": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        if not self._use_azure:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Empty LLM response")
        return content
