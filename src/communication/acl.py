from __future__ import annotations

import datetime as _dt
import logging
import uuid as _uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

_logger = logging.getLogger(__name__)

# Content flag: REQUEST messages that expect AGREE / REFUSE before substantive work.
FIPA_EXPECTS_AGREE_KEY = "fipa_expects_agree"


class ACLPerformative(str, Enum):
    """
    FIPA ACL–oriented performatives used on the async bus.

    Negotiation (unsupported formulation): CFP → PROPOSE → accept-proposal | reject-proposal.
    Delegation (graph analysis, semantic audit, syntax audit): REQUEST with fipa_expects_agree → AGREE → …
    Human gate: REQUEST (human-gate) → AGREE | REFUSE (unchanged).
    """

    INFORM = "inform"
    REQUEST = "request"
    CFP = "cfp"
    FAILURE = "failure"
    NOT_UNDERSTOOD = "not-understood"
    QUERY_IF = "query-if"
    PROPOSE = "propose"
    ACCEPT_PROPOSAL = "accept-proposal"
    REJECT_PROPOSAL = "reject-proposal"
    CONFIRM = "confirm"
    AGREE = "agree"
    REFUSE = "refuse"


@dataclass(frozen=True)
class ACLEnvelope:
    """
    FIPA-ACL-like envelope.

    - ``content`` holds domain payload and optional English ``utterance`` for logging / UX.
    - ``conversation_id`` groups a pipeline run.
    - ``reply_with`` / ``in_reply_to`` chain dialogue turns.
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


def _content_flag(env: ACLEnvelope, key: str) -> bool:
    c = env.content or {}
    return bool(c.get(key))


def _tracked_agree_request(env: ACLEnvelope) -> bool:
    """REQUEST performatives whose AGREE/REFUSE must be registered."""
    if env.performative != ACLPerformative.REQUEST:
        return False
    if env.ontology == "human-gate":
        return True
    return _content_flag(env, FIPA_EXPECTS_AGREE_KEY)


@dataclass
class ACLSemanticRegistry:
    """Tracks open commitments for strict FIPA-style precondition checks."""

    _open_agree_requests: set[str] = field(default_factory=set)
    _failure_reply_targets: set[str] = field(default_factory=set)
    _open_cfp: set[str] = field(default_factory=set)
    _open_proposals: set[str] = field(default_factory=set)

    def apply_publish_effects(self, env: ACLEnvelope) -> None:
        perf = env.performative
        rto = env.in_reply_to
        rw = env.reply_with

        if perf == ACLPerformative.REQUEST and _tracked_agree_request(env):
            self._open_agree_requests.add(rw)
            # Reformulate round: treat the REQUEST id as the anchor for the next PROPOSE
            # (same slot as a CFP for negotiation tracking).
            c = env.content or {}
            if env.ontology == "unsupported-formulation" and c.get("msg_type") == "reformulate":
                self._open_cfp.add(rw)
            return

        if perf == ACLPerformative.CFP:
            self._open_cfp.add(rw)
            return

        if perf == ACLPerformative.PROPOSE:
            if rto and rto in self._open_cfp:
                self._open_cfp.discard(rto)
                self._open_proposals.add(rw)
            return

        if perf in (ACLPerformative.ACCEPT_PROPOSAL, ACLPerformative.REJECT_PROPOSAL):
            if rto and rto in self._open_proposals:
                self._open_proposals.discard(rto)
            return

        if perf == ACLPerformative.AGREE:
            if rto and rto in self._open_agree_requests:
                self._open_agree_requests.discard(rto)
                self._failure_reply_targets.add(rto)
                self._failure_reply_targets.add(rw)
            return

        if perf == ACLPerformative.REFUSE:
            if rto and rto in self._open_agree_requests:
                self._open_agree_requests.discard(rto)
            return

        if perf == ACLPerformative.FAILURE and rto:
            self._failure_reply_targets.discard(rto)


def _tuple_content_ok(env: ACLEnvelope) -> bool:
    c = env.content or {}
    action = str(c.get("action", "")).strip()
    reason = str(c.get("reason", "")).strip()
    return bool(action and reason)


def validate_acl_semantics(env: ACLEnvelope, registry: ACLSemanticRegistry) -> None:
    """
    Preconditions before enqueueing an envelope (strict bus mode).

    - AGREE / REFUSE: ``in_reply_to`` must match an open tracked REQUEST.
    - PROPOSE: ``in_reply_to`` must match an open CFP.
    - ACCEPT-PROPOSAL / REJECT-PROPOSAL: ``in_reply_to`` must match an open PROPOSE.
    - FAILURE: ``in_reply_to`` must match a valid post-AGREE failure target.
    - REFUSE / REJECT-PROPOSAL / FAILURE: require ``action`` and ``reason`` in content when applicable.
    """
    perf = env.performative

    if perf == ACLPerformative.PROPOSE:
        rto = env.in_reply_to
        if not rto:
            raise ValueError("ACL propose: in_reply_to must reference a prior CFP (got None).")
        if rto not in registry._open_cfp:
            raise ValueError(
                f"ACL propose: in_reply_to={rto!r} does not match an open CFP. "
                f"Open CFPs: {sorted(registry._open_cfp)!r}."
            )
        return

    if perf in (ACLPerformative.ACCEPT_PROPOSAL, ACLPerformative.REJECT_PROPOSAL):
        rto = env.in_reply_to
        if not rto:
            raise ValueError(
                f"ACL {perf.value}: in_reply_to must reference a prior PROPOSE (got None)."
            )
        if rto not in registry._open_proposals:
            raise ValueError(
                f"ACL {perf.value}: in_reply_to={rto!r} does not match an open PROPOSE. "
                f"Open: {sorted(registry._open_proposals)!r}."
            )
        if perf == ACLPerformative.REJECT_PROPOSAL and not _tuple_content_ok(env):
            raise ValueError(
                "ACL reject-proposal: content must include non-empty string keys "
                "'action' and 'reason'."
            )
        return

    if perf == ACLPerformative.CONFIRM:
        _logger.warning(
            "ACL CONFIRM without tracked receiver uncertainty (FIPA FP requires context); "
            "conversation_id=%s sender=%s receiver=%s",
            env.conversation_id,
            env.sender,
            env.receiver,
        )
        return

    if perf in (ACLPerformative.AGREE, ACLPerformative.REFUSE):
        rto = env.in_reply_to
        if not rto:
            raise ValueError(
                f"ACL {perf.value}: in_reply_to must reference a prior tracked REQUEST (got None)."
            )
        if rto not in registry._open_agree_requests:
            raise ValueError(
                f"ACL {perf.value}: in_reply_to={rto!r} does not match an open tracked REQUEST. "
                f"Open: {sorted(registry._open_agree_requests)!r}."
            )
        if perf == ACLPerformative.REFUSE and not _tuple_content_ok(env):
            raise ValueError(
                "ACL refuse: content must include non-empty string keys "
                "'action' and 'reason' (action-expression, reason-proposition)."
            )
        return

    if perf == ACLPerformative.FAILURE:
        rto = env.in_reply_to
        if not rto:
            raise ValueError(
                "ACL failure: in_reply_to must reference a prior AGREE (or accepted REQUEST id)."
            )
        if rto not in registry._failure_reply_targets:
            raise ValueError(
                f"ACL failure: in_reply_to={rto!r} is not a valid failure target. "
                f"Valid: {sorted(registry._failure_reply_targets)!r}."
            )
        if not _tuple_content_ok(env):
            raise ValueError(
                "ACL failure: content must include non-empty string keys 'action' and 'reason'."
            )
