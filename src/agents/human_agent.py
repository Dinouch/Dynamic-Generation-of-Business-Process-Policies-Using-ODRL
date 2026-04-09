from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from communication.acl import ACLEnvelope, ACLPerformative


@dataclass
class HumanDecision:
    decision: str  # "agree" | "refuse"
    comment: str = ""
    selected_option_id: Optional[str] = None

    def to_content(self) -> dict[str, Any]:
        out: dict[str, Any] = {"decision": self.decision, "comment": self.comment}
        if self.selected_option_id:
            out["selected_option_id"] = self.selected_option_id
        # FIPA REFUSE / FAILURE tuple shape (action-expression, reason-proposition) for strict ACL checks
        out["action"] = "human_gate_review_unsupported_proposal"
        comment = (self.comment or "").strip()
        if comment:
            out["reason"] = comment
        else:
            out["reason"] = (
                "operator agreed to the proposed unsupported rule"
                if self.decision == "agree"
                else "operator refused the proposed unsupported rule"
            )
        return out


@runtime_checkable
class HumanDecisionBridge(Protocol):
    """Async bridge for web/API HITL (no terminal input)."""

    async def wait_for_decision(self, env: ACLEnvelope) -> HumanDecision: ...


class HumanAgent:
    """
    Minimal Human-in-the-Loop agent.

    This is intentionally simple (CLI) to keep Phase 1 focused:
    - receives an ACL envelope requesting validation/choice
    - prompts the user
    - replies with AGREE/REFUSE and a comment
    """

    AGENT_NAME = "human3"

    def __init__(
        self,
        *,
        timeout_s: Optional[float] = None,
        decision_bridge: Optional[HumanDecisionBridge] = None,
    ):
        self.timeout_s = timeout_s
        self._decision_bridge = decision_bridge
        self._pending: dict[str, ACLEnvelope] = {}

    async def handle(self, env: ACLEnvelope) -> Optional[ACLEnvelope]:
        self._pending[env.reply_with] = env

        if self._decision_bridge is not None:
            decision = await self._decision_bridge.wait_for_decision(env)
            perf = ACLPerformative.AGREE if decision.decision == "agree" else ACLPerformative.REFUSE
            return ACLEnvelope(
                performative=perf,
                sender=self.AGENT_NAME,
                receiver=env.sender,
                ontology=env.ontology,
                content=decision.to_content(),
                conversation_id=env.conversation_id,
                in_reply_to=env.reply_with,
            )

        prompt = self._build_prompt(env)
        # IMPORTANT:
        # Console input is blocking and not cancellable; run it in a DAEMON thread so the
        # process can still terminate after the pipeline continues in partial mode.
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[HumanDecision] = loop.create_future()

        def _runner() -> None:
            try:
                dec = self._ask_user(prompt)
                loop.call_soon_threadsafe(fut.set_result, dec)
            except Exception as e:
                loop.call_soon_threadsafe(fut.set_exception, e)

        t = threading.Thread(target=_runner, name="HITLInputThread", daemon=True)
        t.start()
        decision = await fut

        perf = ACLPerformative.AGREE if decision.decision == "agree" else ACLPerformative.REFUSE
        return ACLEnvelope(
            performative=perf,
            sender=self.AGENT_NAME,
            receiver=env.sender,
            ontology=env.ontology,
            content=decision.to_content(),
            conversation_id=env.conversation_id,
            in_reply_to=env.reply_with,
        )

    def _build_prompt(self, env: ACLEnvelope) -> str:
        c = env.content or {}
        title = c.get("title") or "Validation humaine requise"
        summary = c.get("summary") or ""
        options = c.get("options") or []
        lines = [
            "",
            "═" * 70,
            f"[HITL] {title}",
            "═" * 70,
        ]
        if summary:
            lines.append(summary)
        if options:
            lines.append("")
            lines.append("Options:")
            for o in options:
                oid = o.get("id", "?")
                label = o.get("label", "")
                lines.append(f"  - {oid} : {label}")
        lines.append("")
        lines.append("Répondre par: agree | refuse")
        lines.append("Vous pouvez coller un commentaire (ligne suivante).")
        lines.append("═" * 70)
        lines.append("")
        return "\n".join(lines)

    def _ask_user(self, prompt: str) -> HumanDecision:
        print(prompt)
        raw = (input("> decision: ").strip().lower() or "").strip()
        if raw not in {"agree", "refuse"}:
            raw = "refuse"
        comment = input("> comment: ").strip()

        # Optional: allow JSON for future structured responses
        if comment.startswith("{") and comment.endswith("}"):
            try:
                data = json.loads(comment)
                return HumanDecision(
                    decision=raw,
                    comment=str(data.get("comment", "")),
                    selected_option_id=data.get("selected_option_id"),
                )
            except Exception:
                pass

        return HumanDecision(decision=raw, comment=comment)

