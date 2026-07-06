"""
HITL bridge for the web API: human decisions arrive via HTTP instead of the terminal.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from api.hitl_humanize import build_hitl_narrative_async
from agents.human_agent import HumanDecision
from communication.acl import ACLEnvelope


def _safe_json_content(content: dict[str, Any]) -> dict[str, Any]:
    """Serialize ACL content for the frontend (non-JSON objects -> str)."""
    try:
        return json.loads(json.dumps(content or {}, default=str))
    except Exception:
        return {"raw": str(content)}


def envelope_to_wire_dict(env: ACLEnvelope) -> dict[str, Any]:
    """JSON representation of a FIPA-ACL envelope for SSE."""
    perf = env.performative.value if hasattr(env.performative, "value") else str(env.performative)
    return {
        "performative": perf,
        "sender": env.sender,
        "receiver": env.receiver,
        "ontology": env.ontology,
        "content": _safe_json_content(dict(env.content or {})),
        "language": env.language,
        "conversation_id": env.conversation_id,
        "reply_with": env.reply_with,
        "in_reply_to": env.in_reply_to,
        "timestamp": env.timestamp,
    }


class WebHitlBridge:
    """
    - ``wait_for_decision`` : called by ``HumanAgent``; notifies SSE and waits for POST /hitl.
    - ``submit`` : resolves the future when the user responds from the frontend.
    """

    def __init__(self) -> None:
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._futures: dict[str, asyncio.Future[HumanDecision]] = {}

    def pending_hitl_keys(self) -> list[str]:
        """For API logs (diagnostic when POST /hitl is rejected)."""
        return list(self._futures.keys())

    async def wait_for_decision(self, env: ACLEnvelope) -> HumanDecision:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[HumanDecision] = loop.create_future()
        rw_key = (env.reply_with or "").strip()
        self._futures[rw_key] = fut
        content_safe = _safe_json_content(dict(env.content or {}))
        human_narrative = await build_hitl_narrative_async(content_safe)
        await self._event_queue.put(
            {
                "type": "hitl",
                "reply_with": rw_key,
                "conversation_id": env.conversation_id,
                "content": content_safe,
                "human_narrative": human_narrative,
            }
        )
        return await fut

    def submit(self, reply_with: str, decision: str, comment: str = "") -> bool:
        key = (reply_with or "").strip()
        fut = self._futures.pop(key, None)
        if fut is None:
            # Legacy frontend compatibility:
            # some clients keep submitting the first `reply_with` even when the
            # backend has already moved to the next unmapped proposal.
            # In one-by-one HITL mode, there is at most one pending request, so
            # we can safely route the answer to that sole pending future.
            if len(self._futures) == 1:
                only_key = next(iter(self._futures))
                fut = self._futures.pop(only_key, None)
        if fut is None or fut.done():
            return False
        d = (decision or "").strip().lower()
        if d not in {"agree", "refuse"}:
            d = "refuse"
        fut.set_result(HumanDecision(decision=d, comment=comment or ""))
        return True

    async def emit_done(self, payload: dict[str, Any]) -> None:
        await self._event_queue.put({"type": "done", "payload": payload})

    async def emit_log(self, message: str) -> None:
        await self._event_queue.put({"type": "log", "message": message})

    async def emit_acl(self, env: ACLEnvelope) -> None:
        """Trace each message published on the bus (FIPA demo)."""
        await self._event_queue.put({"type": "acl", "envelope": envelope_to_wire_dict(env)})

    async def emit_error(self, message: str) -> None:
        await self._event_queue.put({"type": "error", "message": message})

    async def iter_events(self) -> AsyncIterator[dict[str, Any]]:
        """Consume the queue until a ``done`` or ``error`` event."""
        while True:
            item = await self._event_queue.get()
            yield item
            if item.get("type") in ("done", "error"):
                break

    async def pull_event_or_timeout(self, timeout_s: float) -> dict[str, Any] | None:
        """For SSE: ``None`` on timeout (send a keep-alive comment)."""
        try:
            return await asyncio.wait_for(self._event_queue.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None
