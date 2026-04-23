"""
pipeline_registry.py — Single source of truth for BPMN pattern coverage (Agent 4).

This module lists every structural pattern type that Agent 4 can translate using
deterministic ODRL templates. When a new template is added to Agent 4, add the
corresponding pattern_type string here so Agent 1 tags anything else as unsupported
and routes it through the exception handling agent.

Agent 1 and Agent 4 both import this module; keep it synchronized with
policy_projection_agent template logic.
"""

from typing import FrozenSet

# Patterns Agent 4 handles via deterministic templates (must match pattern_type strings).
COVERED_PATTERNS: FrozenSet[str] = frozenset(
    {
        # Gateway forks
        "fork_xor",
        "fork_and",
        "fork_or",
        # Gateway joins
        "join_xor",
        "join_and",
        # Flow / graph (also detected as structural notes)
        "sequence",
        "message",
    }
)

# BPMN 2.0 semantics for unsupported pattern_type values (English, one sentence each).
# Used verbatim in exception handling agent LLM prompts.
BPMN_SEMANTICS: dict[str, str] = {
    "fork_event_based_exclusive": (
        "An Event-Based Gateway (exclusive instantiation) waits for one of several "
        "following events; the first event to occur determines which path is taken."
    ),
    "fork_event_based_parallel": (
        "An Event-Based Gateway (parallel instantiation) waits for all expected "
        "events before the process continues along the merged path."
    ),
    "fork_complex": (
        "A Complex Gateway uses a custom activation expression to decide which "
        "combination of outgoing sequence flows are enabled."
    ),
    "join_or": (
        "An inclusive (OR) join merges multiple incoming sequence flows when "
        "the gateway's join condition is satisfied."
    ),
    "default_flow": (
        "A default sequence flow is taken when no other conditional outgoing flow "
        "from the same exclusive or inclusive gateway evaluates to true."
    ),
    "conditional_flow": (
        "A conditional sequence flow carries an explicit condition and is taken "
        "when that condition is true, without an intermediate splitting gateway."
    ),
    "event_subprocess": (
        "An Event Sub-Process is a subprocess triggered by an event while its "
        "parent activity or process is active."
    ),
    "compensation": (
        "Compensation associates a boundary compensation event with a handler "
        "activity that reverses or undoes completed work."
    ),
    "loop_activity": (
        "A standard loop activity repeats its inner behavior until a loop condition "
        "no longer holds or a maximum number of iterations is reached."
    ),
    "multi_instance_sequential": (
        "A multi-instance activity with sequential multi-loop executes its inner "
        "behavior a fixed or data-driven number of times, one after another."
    ),
    "multi_instance_parallel": (
        "A multi-instance activity with parallel multi-loop executes its inner "
        "behavior for each instance concurrently."
    ),
    "ad_hoc_subprocess": (
        "An ad-hoc subprocess contains activities that may be executed in any order "
        "subject to completion conditions, without a fixed control flow."
    ),
    "call_activity": (
        "A call activity invokes a reusable global or local process or subprocess "
        "definition as a subordinate execution."
    ),
    "sync": (
        "A synchronization point merges parallel branches so execution continues "
        "only after required branches have completed."
    ),
    "loop": (
        "A control-flow cycle in the process graph indicates iterative or cyclic "
        "execution until an exit condition is met."
    ),
}
