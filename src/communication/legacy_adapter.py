from __future__ import annotations

from typing import Any, Optional

from agents.structural_analyzer import AgentMessage, MessageType

from .acl import ACLEnvelope, ACLPerformative, FIPA_EXPECTS_AGREE_KEY


def _strip_acl_internal(d: dict[str, Any]) -> dict[str, Any]:
    """Remove routing metadata from copied payload (utterance is kept)."""
    out = dict(d)
    for k in (
        "acl_in_reply_to",
        "acl_envelope_reply_with",
        "cfp_call_id",
        "proposal_message_id",
        "syntax_audit_request_id",
        "projection_request_id",
        "semantic_correction_request_id",
        "syntax_correction_request_id",
        "_acl_ontology",
        "_acl_reply_with",
    ):
        out.pop(k, None)
    return out


def agent_message_to_acl(
    msg: AgentMessage,
    *,
    conversation_id: Optional[str] = None,
    in_reply_to: Optional[str] = None,
    status: Optional[str] = None,
) -> ACLEnvelope:
    """
    Build an ACL envelope from a legacy ``AgentMessage``.

    Uses ``msg.payload`` keys:
    - ``acl_in_reply_to``: envelope ``in_reply_to`` (overrides parameter).
    - ``fipa_expects_agree`` on REQUEST delegations (semantic / syntax / reformulate).
    """
    p = dict(msg.payload or {})
    eff_in_reply = in_reply_to or p.pop("acl_in_reply_to", None)
    reply_with_override = p.pop("_acl_reply_with", None)

    # --- Explicit delegation performatives (AGREE / REFUSE) ---
    if msg.msg_type == MessageType.DELEGATION_AGREE:
        ont = str(p.pop("_acl_ontology", "graph-structural"))
        utterance = str(p.pop("utterance", "") or "I accept the delegated task and will proceed.")
        content = _strip_acl_internal(p)
        content["utterance"] = utterance
        env_kw: dict[str, Any] = dict(
            performative=ACLPerformative.AGREE,
            sender=msg.sender,
            receiver=msg.recipient,
            ontology=ont,
            content=content,
            language="json",
            conversation_id=conversation_id or "",
            in_reply_to=eff_in_reply,
        )
        if reply_with_override:
            env_kw["reply_with"] = reply_with_override
        return ACLEnvelope(**env_kw)

    if msg.msg_type == MessageType.DELEGATION_REFUSE:
        ont = str(p.pop("_acl_ontology", "semantic-audit"))
        content = _strip_acl_internal(p)
        if "action" not in content:
            content["action"] = "refuse-delegated-task"
        if "reason" not in content:
            content["reason"] = str(content.get("utterance") or "Unable to complete the delegated task.")
        content.setdefault(
            "utterance",
            str(content.get("reason") or "Refusing delegated task."),
        )
        env_kw = dict(
            performative=ACLPerformative.REFUSE,
            sender=msg.sender,
            receiver=msg.recipient,
            ontology=ont,
            content=content,
            language="json",
            conversation_id=conversation_id or "",
            in_reply_to=eff_in_reply,
        )
        if reply_with_override:
            env_kw["reply_with"] = reply_with_override
        return ACLEnvelope(**env_kw)

    # --- Standard msg_type → performative ---
    perf: ACLPerformative
    ont: str
    extra_content: dict[str, Any] = {}

    if msg.msg_type == MessageType.GRAPH_READY:
        perf, ont = ACLPerformative.INFORM, "graph-structural"
    elif msg.msg_type == MessageType.CFP_UNSUPPORTED:
        perf, ont = ACLPerformative.CFP, "unsupported-formulation"
    elif msg.msg_type == MessageType.UNSUPPORTED_PROPOSALS:
        perf, ont = ACLPerformative.PROPOSE, "unsupported-formulation"
    elif msg.msg_type == MessageType.REFORMULATED_PROPOSALS:
        perf, ont = ACLPerformative.INFORM, "unsupported-formulation"
    elif msg.msg_type == MessageType.REFORMULATE:
        perf, ont = ACLPerformative.REQUEST, "unsupported-formulation"
        extra_content[FIPA_EXPECTS_AGREE_KEY] = True
    elif msg.msg_type == MessageType.ACCEPT_PROPOSAL_BATCH:
        perf, ont = ACLPerformative.ACCEPT_PROPOSAL, "unsupported-formulation"
    elif msg.msg_type == MessageType.REJECT_PROPOSAL_BATCH:
        perf, ont = ACLPerformative.REJECT_PROPOSAL, "unsupported-formulation"
    elif msg.msg_type == MessageType.VALIDATION_DONE:
        perf, ont = ACLPerformative.REQUEST, "semantic-audit"
        extra_content[FIPA_EXPECTS_AGREE_KEY] = True
        extra_content.setdefault(
            "utterance",
            "Project fragment policies from this validation report for semantic audit.",
        )
    elif msg.msg_type == MessageType.POLICIES_READY:
        perf, ont = ACLPerformative.INFORM, "policy-projection"
    elif msg.msg_type == MessageType.SEMANTIC_CORRECTION:
        perf, ont = ACLPerformative.REQUEST, "semantic-audit"
        extra_content[FIPA_EXPECTS_AGREE_KEY] = True
        extra_content.setdefault(
            "utterance",
            "Apply these semantic corrections to the projected policies.",
        )
    elif msg.msg_type == MessageType.SEMANTIC_VALIDATED:
        perf, ont = ACLPerformative.INFORM, "semantic-audit"
    elif msg.msg_type == MessageType.SEMANTIC_VALIDATION_FAILURE:
        perf, ont = ACLPerformative.FAILURE, "semantic-audit"
    elif msg.msg_type == MessageType.SYNTAX_AUDIT_REQUEST:
        perf, ont = ACLPerformative.REQUEST, "odrl-syntax-audit"
        extra_content[FIPA_EXPECTS_AGREE_KEY] = True
        extra_content.setdefault(
            "utterance",
            "Validate ODRL syntax for these fragment policies.",
        )
    elif msg.msg_type == MessageType.SYNTAX_CORRECTION:
        perf, ont = ACLPerformative.REQUEST, "odrl-syntax-audit"
        extra_content[FIPA_EXPECTS_AGREE_KEY] = True
        extra_content.setdefault(
            "utterance",
            "Repair the reported ODRL syntax issues.",
        )
    elif msg.msg_type == MessageType.ODRL_VALID:
        perf, ont = ACLPerformative.INFORM, "odrl-syntax-audit"
    elif msg.msg_type == MessageType.ODRL_SYNTAX_ERROR:
        perf, ont = ACLPerformative.INFORM, "odrl-syntax-audit"
    elif msg.msg_type == MessageType.ODRL_SYNTAX_FAILURE:
        perf, ont = ACLPerformative.FAILURE, "odrl-syntax-audit"
    else:
        perf, ont = ACLPerformative.INFORM, "legacy"

    content: dict[str, Any] = _strip_acl_internal(p)
    content.update(extra_content)
    content["loop_turn"] = msg.loop_turn
    content["msg_type"] = msg.msg_type.value
    if status:
        content["status"] = status
    env_kw = dict(
        performative=perf,
        sender=msg.sender,
        receiver=msg.recipient,
        ontology=ont,
        content=content,
        language="json",
        conversation_id=conversation_id or "",
        in_reply_to=eff_in_reply,
    )
    if reply_with_override:
        env_kw["reply_with"] = reply_with_override
    return ACLEnvelope(**env_kw)


def acl_to_agent_message(env: ACLEnvelope) -> AgentMessage:
    """Map an ACL envelope to the legacy agent view (``AgentMessage``)."""
    payload = dict(env.content or {})
    loop_turn = int(payload.pop("loop_turn", 0) or 0)
    msg_type_val = str(payload.pop("msg_type", "") or "")
    status = payload.pop("status", None)

    perf = env.performative
    ont = env.ontology

    # REQUEST from pipeline → structural analysis task
    if (
        perf == ACLPerformative.REQUEST
        and env.sender == "pipeline"
        and env.receiver == "agent1"
        and ont == "graph-structural"
    ):
        mtype = MessageType.ANALYZE_GRAPH_TASK
        payload["request_message_id"] = env.reply_with
        payload.setdefault(
            "utterance",
            str(payload.get("utterance") or "Analyze this business process graph."),
        )
        if status:
            payload["status"] = status
        return AgentMessage(
            sender=env.sender,
            recipient=env.receiver,
            msg_type=mtype,
            payload=payload,
            loop_turn=loop_turn,
        )

    if perf == ACLPerformative.CFP and ont == "unsupported-formulation":
        payload["cfp_call_id"] = env.reply_with
        mtype = MessageType.CFP_UNSUPPORTED
    elif perf == ACLPerformative.REQUEST and ont == "unsupported-formulation":
        payload["reformulate_request_id"] = env.reply_with
        mtype = MessageType.REFORMULATE
    elif perf == ACLPerformative.PROPOSE and ont == "unsupported-formulation":
        payload["proposal_message_id"] = env.reply_with
        try:
            mtype = MessageType(msg_type_val) if msg_type_val else MessageType.UNSUPPORTED_PROPOSALS
        except ValueError:
            mtype = MessageType.UNSUPPORTED_PROPOSALS
    elif perf == ACLPerformative.INFORM and ont == "unsupported-formulation":
        if msg_type_val == MessageType.REFORMULATED_PROPOSALS.value:
            mtype = MessageType.REFORMULATED_PROPOSALS
        else:
            try:
                mtype = MessageType(msg_type_val) if msg_type_val else MessageType.REFORMULATED_PROPOSALS
            except ValueError:
                mtype = MessageType.REFORMULATED_PROPOSALS
    elif perf == ACLPerformative.REQUEST and ont == "semantic-audit":
        intent = str(payload.get("intent") or payload.get("msg_type") or "")
        if "semantic_correction" in intent or msg_type_val == MessageType.SEMANTIC_CORRECTION.value:
            payload["semantic_correction_request_id"] = env.reply_with
            mtype = MessageType.SEMANTIC_CORRECTION
        else:
            payload["projection_request_id"] = env.reply_with
            mtype = MessageType.VALIDATION_DONE
    elif perf == ACLPerformative.INFORM and ont == "semantic-audit":
        mtype = MessageType.SEMANTIC_VALIDATED
    elif perf == ACLPerformative.FAILURE and ont == "semantic-audit":
        mtype = MessageType.SEMANTIC_VALIDATION_FAILURE
    elif perf == ACLPerformative.REQUEST and ont == "odrl-syntax-audit":
        if msg_type_val == MessageType.SYNTAX_CORRECTION.value or "syntax_correction" in str(
            payload.get("intent", "")
        ):
            payload["syntax_correction_request_id"] = env.reply_with
            mtype = MessageType.SYNTAX_CORRECTION
        else:
            payload["syntax_audit_request_id"] = env.reply_with
            mtype = MessageType.SYNTAX_AUDIT_REQUEST
    elif perf == ACLPerformative.INFORM and ont == "odrl-syntax-audit":
        if msg_type_val == MessageType.ODRL_SYNTAX_ERROR.value:
            mtype = MessageType.ODRL_SYNTAX_ERROR
        else:
            mtype = MessageType.ODRL_VALID
    elif perf == ACLPerformative.FAILURE and ont == "odrl-syntax-audit":
        mtype = MessageType.ODRL_SYNTAX_FAILURE
    elif msg_type_val:
        try:
            mtype = MessageType(msg_type_val)
        except ValueError:
            mtype = MessageType.GRAPH_READY
    else:
        mtype = MessageType.GRAPH_READY

    if status:
        payload["status"] = status

    return AgentMessage(
        sender=env.sender,
        recipient=env.receiver,
        msg_type=mtype,
        payload=payload,
        loop_turn=loop_turn,
    )
