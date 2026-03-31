"""
Pont HITL pour l'API web : les décisions humaines arrivent via HTTP au lieu du terminal.
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
    """Sérialise le contenu ACL pour le frontend (objets non JSON -> str)."""
    try:
        return json.loads(json.dumps(content or {}, default=str))
    except Exception:
        return {"raw": str(content)}


def envelope_to_wire_dict(env: ACLEnvelope) -> dict[str, Any]:
    """Représentation JSON d'une enveloppe FIPA-ACL pour le SSE."""
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
    - ``wait_for_decision`` : appelé par ``HumanAgent`` ; notifie le SSE et attend POST /hitl.
    - ``submit`` : résout la future quand l'utilisateur répond depuis le frontend.
    """

    def __init__(self) -> None:
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._futures: dict[str, asyncio.Future[HumanDecision]] = {}

    def pending_hitl_keys(self) -> list[str]:
        """Pour logs API (diagnostic POST /hitl refusé)."""
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
        """Trace chaque message publié sur le bus (démo FIPA)."""
        await self._event_queue.put({"type": "acl", "envelope": envelope_to_wire_dict(env)})

    async def emit_error(self, message: str) -> None:
        await self._event_queue.put({"type": "error", "message": message})

    async def iter_events(self) -> AsyncIterator[dict[str, Any]]:
        """Consomme la file jusqu'à un événement ``done`` ou ``error``."""
        while True:
            item = await self._event_queue.get()
            yield item
            if item.get("type") in ("done", "error"):
                break

    async def pull_event_or_timeout(self, timeout_s: float) -> dict[str, Any] | None:
        """Pour SSE : ``None`` si timeout (envoyer un commentaire keep-alive)."""
        try:
            return await asyncio.wait_for(self._event_queue.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None
