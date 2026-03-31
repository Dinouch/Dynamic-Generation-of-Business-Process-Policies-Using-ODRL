from __future__ import annotations

from typing import Any, Optional

from agents.structural_analyzer import AgentMessage, MessageType

from .acl import ACLEnvelope, ACLPerformative


_MSGTYPE_TO_ACL: dict[MessageType, tuple[ACLPerformative, str]] = {
    MessageType.GRAPH_READY: (ACLPerformative.INFORM, "graph-structural"),
    MessageType.STRUCTURAL_UPDATE: (ACLPerformative.REQUEST, "graph-structural"),
    MessageType.UNSUPPORTED_PROPOSALS: (ACLPerformative.PROPOSE, "unsupported-formulation"),
    MessageType.REFORMULATE: (ACLPerformative.REQUEST, "unsupported-formulation"),
    MessageType.VALIDATION_DONE: (ACLPerformative.INFORM, "validation"),
    MessageType.POLICIES_READY: (ACLPerformative.INFORM, "policy-projection"),
    MessageType.SEMANTIC_CORRECTION: (ACLPerformative.REQUEST, "semantic-audit"),
    MessageType.SEMANTIC_VALIDATED: (ACLPerformative.CONFIRM, "semantic-audit"),
    MessageType.SYNTAX_CORRECTION: (ACLPerformative.REQUEST, "odrl-syntax-audit"),
    MessageType.ODRL_VALID: (ACLPerformative.INFORM, "odrl-syntax-audit"),
    MessageType.ODRL_SYNTAX_ERROR: (ACLPerformative.FAILURE, "odrl-syntax-audit"),
}


def agent_message_to_acl(
    msg: AgentMessage,
    *,
    conversation_id: Optional[str] = None,
    in_reply_to: Optional[str] = None,
    status: Optional[str] = None,
) -> ACLEnvelope:
    perf, ont = _MSGTYPE_TO_ACL.get(msg.msg_type, (ACLPerformative.INFORM, "legacy"))
    content: dict[str, Any] = dict(msg.payload or {})
    content["loop_turn"] = msg.loop_turn
    content["msg_type"] = msg.msg_type.value
    if status:
        content["status"] = status
    return ACLEnvelope(
        performative=perf,
        sender=msg.sender,
        receiver=msg.recipient,
        ontology=ont,
        content=content,
        language="json",
        conversation_id=conversation_id or "",
        in_reply_to=in_reply_to,
    )


def acl_to_agent_message(env: ACLEnvelope) -> AgentMessage:
    # Best-effort: rely on content["msg_type"] when present; default to GRAPH_READY.
    msg_type_val = str((env.content or {}).get("msg_type", MessageType.GRAPH_READY.value))
    mtype = next((t for t in MessageType if t.value == msg_type_val), MessageType.GRAPH_READY)
    payload = dict(env.content or {})
    # remove envelope metadata
    payload.pop("status", None)
    payload.pop("msg_type", None)
    loop_turn = int(payload.pop("loop_turn", 0) or 0)
    return AgentMessage(
        sender=env.sender,
        recipient=env.receiver,
        msg_type=mtype,
        payload=payload,
        loop_turn=loop_turn,
    )

