"""
Sequential syntax loop Agent 4 ↔ Agent 5 (profile **I3**).
"""

from __future__ import annotations

from typing import Any, Optional

from agents.Agent_3.constraint_validator import ValidationReport
from agents.Agent_4.policy_projection_agent import PolicyProjectionAgent
from agents.policy_auditor import PolicyAuditor
from agents.structural_analyzer import AgentMessage, EnrichedGraph, MessageType


def run_syntax_correction_loop(
    enriched: EnrichedGraph,
    validation_report: ValidationReport,
    projector: PolicyProjectionAgent,
    auditor: PolicyAuditor,
    *,
    increment_profile: str = "I3",
) -> dict[str, Any]:
    """
    Trigger ``VALIDATION_DONE`` on Agent 4 and route messages until
    ``ODRL_VALID`` / ``ODRL_SYNTAX_ERROR`` reach ``pipeline`` (no Agent 3 semantic pass).
    """
    terminal: list[AgentMessage] = []

    def route(msg: AgentMessage) -> None:
        recipient = msg.recipient
        if recipient == PolicyAuditor.AGENT_NAME:
            auditor.receive(msg)
        elif recipient == PolicyProjectionAgent.AGENT_NAME:
            projector.receive(msg)
        elif recipient == "pipeline" and msg.msg_type in (
            MessageType.ODRL_VALID,
            MessageType.ODRL_SYNTAX_ERROR,
        ):
            terminal.append(msg)
        elif recipient == "agent3":
            print(
                f"[incremental I3] Message to Agent 3 ignored ({msg.msg_type.value}) "
                "— profile without semantic validation."
            )

    projector.register_send_callback(route)
    auditor.register_send_callback(route)
    projector._increment_stop_before_semantic = True  # noqa: SLF001
    projector._increment_profile = increment_profile  # noqa: SLF001

    projector.receive(
        AgentMessage(
            sender="pipeline",
            recipient=projector.AGENT_NAME,
            msg_type=MessageType.VALIDATION_DONE,
            payload={
                "validation_report": validation_report,
                "enriched_graph": enriched,
                "status": f"incremental_{increment_profile}",
                "increment_profile": increment_profile,
            },
            loop_turn=0,
        )
    )

    if terminal:
        last = terminal[-1]
        print(
            f"[incremental I3] Syntax loop finished: {last.msg_type.value} "
            f"(is_valid={(last.payload or {}).get('is_valid')})"
        )
    return projector._last_fp_results or {}  # noqa: SLF001
