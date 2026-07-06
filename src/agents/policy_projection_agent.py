"""
Compatibility shim: live implementation in ``agents.Agent_4``.

Deterministic ODRL templates live in ``agents.Agent_4.odrl_deterministic_templates``.
"""

from .Agent_4.odrl_deterministic_templates import FragmentPolicySet
from .Agent_4.policy_projection_agent import PolicyProjectionAgent

__all__ = ["FragmentPolicySet", "PolicyProjectionAgent"]
