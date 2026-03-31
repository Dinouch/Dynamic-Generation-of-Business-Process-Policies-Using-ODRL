from __future__ import annotations

import datetime as _dt
import uuid as _uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ACLPerformative(str, Enum):
    INFORM = "inform"
    REQUEST = "request"
    PROPOSE = "propose"
    CONFIRM = "confirm"
    AGREE = "agree"
    REFUSE = "refuse"
    FAILURE = "failure"
    NOT_UNDERSTOOD = "not-understood"
    QUERY_IF = "query-if"


@dataclass(frozen=True)
class ACLEnvelope:
    """
    Minimal FIPA-ACL-like envelope.

    Notes
    -----
    - ``content`` holds the domain payload (often the legacy AgentMessage.payload),
      plus workflow metadata such as ``status``.
    - ``conversation_id`` groups a full pipeline run.
    - ``reply_with`` / ``in_reply_to`` chain request/response pairs.
    """

    performative: ACLPerformative
    sender: str
    receiver: str
    ontology: str
    content: dict[str, Any]
    language: str = "json"
    conversation_id: str = field(default_factory=lambda: _uuid.uuid4().hex)
    reply_with: str = field(default_factory=lambda: _uuid.uuid4().hex)
    in_reply_to: Optional[str] = None
    timestamp: str = field(default_factory=lambda: _dt.datetime.now().isoformat())

    def with_status(self, status: str) -> "ACLEnvelope":
        c = dict(self.content or {})
        c["status"] = status
        return ACLEnvelope(
            performative=self.performative,
            sender=self.sender,
            receiver=self.receiver,
            ontology=self.ontology,
            content=c,
            language=self.language,
            conversation_id=self.conversation_id,
            reply_with=self.reply_with,
            in_reply_to=self.in_reply_to,
            timestamp=self.timestamp,
        )

