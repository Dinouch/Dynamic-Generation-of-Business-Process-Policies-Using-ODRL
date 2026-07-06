"""
Shared prompts — BPMN → rule activation model → ODRL compilation.

The generator and validator share the same mental model:
  BPMN activity → governable rule/asset (existing B2P or activity asset)
  gateway → conditional activation relation (not a "gateway policy")
  event → branch rule activation condition

Intermediate reasoning is latent; only the final JSON is exposed.
"""

from __future__ import annotations

__all__ = [
    "COMPILATION_MENTAL_MODEL",
    "SEMANTIC_HIERARCHY_LAYERS",
    "GENERATOR_COMPILATION_PIPELINE",
    "ANTI_STRUCTURAL_SHORTCUTS",
    "RULE_AND_TARGET_BINDING",
    "GENERATOR_INTERNAL_REASONING",
    "VALIDATOR_INTERNAL_RECONSTRUCTION",
    "VALIDATOR_DETERMINISTIC_TEMPLATE_MODE",
    "SURGICAL_CORRECTION_FOR_TEMPLATES",
    "PATTERN_TRANSLATION_HEURISTICS",
    "heuristics_for_pattern",
    "generator_user_prompt_body",
    "semantic_repair_user_prompt_body",
    "validator_user_prompt_preamble",
    "validator_user_prompt_epilogue",
    "validator_user_prompt_for_deterministic_template",
]

# ── Mental model: compilation, not literal translation ─────────────────────

COMPILATION_MENTAL_MODEL = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOU ARE A STRUCTURAL COMPILER — NOT A TEXT TRANSLATOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are NOT translating BPMN diagram elements directly into ODRL fields.
You are NOT answering: "which ODRL policy describes this gateway?"

You ARE compiling BPMN operational semantics into a governance-oriented
rule-activation model:

  BPMN semantics
    → identify control-flow pattern
    → derive operational semantics (who/what becomes executable when)
    → derive governance semantics (permission / prohibition / obligation)
    → derive rule activation semantics (which rule is conditionally enabled)
    → express activation as ODRL meta-rules (permissions with constraints)

Mental chain for every branch:

  event or business condition
    → activates a BPMN branch
    → branch enables ONE governable rule (target)
    → ODRL policy governs the conditional enablement of that rule

Policies do NOT represent gateways.
Policies govern conditional enablement of branch-specific rules derived from BPMN.
""".strip()

SEMANTIC_HIERARCHY_LAYERS = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEMANTIC HIERARCHY (mandatory)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Level 1 — BPMN activity (executable work unit in the process)
Level 2 — Business rule (governance unit: who may/must/must-not act)
Level 3 — ODRL target (reference to an existing asset IRI or an existing B2P rule uid)
Level 4 — Conditional enablement (constraints on when that target/rule applies)

Mapping rules:
• Each downstream branch activity → its own governable target (asset IRI).
• Do NOT invent new rule uids under http://example.com/rules/ — only use valid_b2p_rule_uids.
• Constraints MAY be created to express pattern-specific activation (event, time, count, …).
• Never collapse the hierarchy by pointing every rule at the gateway or at one generic asset.
""".strip()

GENERATOR_COMPILATION_PIPELINE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTERNAL COMPILATION PIPELINE (execute in order — do not output)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before writing any JSON, complete these steps internally:

1. Identify the BPMN control-flow pattern (pattern_type, gateway role, branches).
2. Determine which activities become conditionally executable (one per branch path).
3. Map each executable branch activity to ONE governable target (existing asset IRI).
4. Determine which events or business conditions activate each branch's rule.
5. Model gateway semantics as conditional rule enablement — NOT as a gateway target.
6. Generate ODRL permissions/prohibitions/obligations governing activation of those rules.
7. Preserve branch exclusivity and event semantics (XOR / event-based: mutually exclusive activations).
8. Avoid collapsing multiple branches into a single target unless BPMN explicitly converges
   on one activity before the governed step.

For event-based exclusive gateways, ask:
  "Which rule becomes enabled when event E occurs?" — once per branch, not once per gateway.

For XOR gateways, ask:
  "Which rule is enabled under condition C?" — distinct constraint + distinct target per branch.
""".strip()

ANTI_STRUCTURAL_SHORTCUTS = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORBIDDEN STRUCTURAL SHORTCUTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LLMs tend to compress structure. You must resist that.

Do NOT:
• Describe the whole gateway in a single ODRL rule when multiple branches exist.
• Reuse the same target for every branch unless BPMN converges to one activity.
• Invent assets (documentation-outcome-record, gateway-name-as-asset, …).
• Invent rule uids not present in valid_b2p_rule_uids.
• Use BPMN routing vocabulary in ODRL fields (gateway, fork, exclusive, firstEventWins, …).
• Use non-ODRL actions (enable, trigger, nextPolicy, …) when compiling NEW unmapped policies.

Note: deterministic Agent 4 templates (XOR/AND/OR/sequence/message) may use project-specific
activation actions (odrl:enable, nextPolicy) — do NOT flag or replace those during template repair.

Do NOT compress multiple BPMN branches into one policy target unless the graph explicitly
converges before the governed activity.

Distinct downstream branch activities MUST produce distinct rule targets.

A superficially compact ODRL document is NOT a success criterion.
Preserving branch-level activation semantics IS.
""".strip()

RULE_AND_TARGET_BINDING = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES, ASSETS, AND B2P BINDING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Prefer target = http://example.com/asset/<activity_slug> for branch activities listed in
  valid_asset_targets / bpmn_activity_slugs.
• Use target = http://example.com/rules/<uid> ONLY when reusing an existing B2P rule uid
  from valid_b2p_rule_uids — never mint new rule IRIs.
• Each ODRL rule in the policy MUST have at least one constraint (or meaningful refinement).
• Hints from the exception-handling agent explain WHY the pattern is unmapped — they are
  not literal values to paste into constraints.
""".strip()

# ── Internal reasoning (generator) — do not reproduce in output ─────

GENERATOR_INTERNAL_REASONING = f"""
{COMPILATION_MENTAL_MODEL}

{SEMANTIC_HIERARCHY_LAYERS}

{GENERATOR_COMPILATION_PIPELINE}

{ANTI_STRUCTURAL_SHORTCUTS}

Take time to reason carefully and internally before producing the final policy.
Do not autocomplete the first ODRL-shaped JSON you can imagine.

Only after the full compilation pipeline is complete, compose the final odrl_policy.
""".strip()

# ── Double-check (validator) — do not reproduce in output ──────

VALIDATOR_INTERNAL_RECONSTRUCTION = f"""
{COMPILATION_MENTAL_MODEL}

{SEMANTIC_HIERARCHY_LAYERS}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTERNAL RECONSTRUCTION (execute before verdict — do not output)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Independently re-run the compilation pipeline on the BPMN context, then compare
the generated policy to your reconstruction:

1. Re-identify the control-flow pattern from ENRICHED BPMN GRAPH.
2. List branch activities that should be conditionally executable.
3. For each branch, state which event/condition should enable which target rule.
4. Check whether the ODRL policy models gateway semantics as rule activation
   (not as a gateway-shaped target).
5. Verify distinct branch activities → distinct targets (unless explicit convergence).
6. Verify no invented assets/rules; constraints are operational, not BPMN routing jargon.
7. Decide whether a BPM analyst would accept this as faithful governance compilation.

Reject policies that only look ODRL-shaped but fail the activation model.
""".strip()

# ── Pattern translation heuristics ───────────────────────────────────

PATTERN_TRANSLATION_HEURISTICS = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN → RULE ACTIVATION (compilation heuristics)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Apply the section matching pattern_type. Think "which rule is enabled when …?"

fork_xor / XOR gateway:
  Operational: exactly one branch fires.
  Compilation: one ODRL rule per branch activity; mutually exclusive constraints
  (event, purpose, or business predicate); each rule.target = that branch's activity asset.
  Never target the gateway node.

fork_event_based_exclusive / event-based exclusive:
  Operational: first qualifying event determines the path.
  Compilation: one rule per event-triggered branch; constraint.event (or equivalent)
  expresses the business trigger (deadline elapsed, documents received, …).
  Each rule enables the downstream activity asset for that event.
  Do NOT model "the gateway" as target.

fork_and / parallel fork:
  Operational: all branches proceed concurrently.
  Compilation: separate rules (or parallel permissions) per branch activity;
  do not merge parallel paths into one vague rule.

fork_or / inclusive OR:
  Operational: one or more branches may activate.
  Compilation: rules may overlap; constraints should not falsely imply mutual exclusion.

join_* / join gateway:
  Operational: synchronization before continuing.
  Compilation: governance on the synchronized activity (merge point asset), not on join gateway id.

loop / cycle_detected:
  Operational: repeated execution until exit condition.
  Compilation: obligation/prohibition on re-entry; recurrence or termination constraint;
  target = activity in the loop body (from involved_activity_names).

sequence / flow_sequence:
  Operational: ordered handoff A then B.
  Compilation: precondition or ordered constraints; targets = A then B activity assets.

message / message_flow:
  Operational: message triggers work on receiving activity.
  Compilation: permission/obligation on receiving activity; event = message received.

conditional_flow / default_flow:
  Operational: conditional vs default path.
  Compilation: business predicate constraints mirroring the condition — not sequenceFlow ids.

default (unknown pattern):
  Compile to WHO may/must/must-not do WHAT on WHICH existing asset WHEN which condition holds.
  Prefer multiple precise branch rules over one generic gateway rule.
""".strip()


def heuristics_for_pattern(pattern_type: str) -> str:
    """Targeted reminder for the primary pattern being processed."""
    pt = (pattern_type or "").strip().lower()
    if not pt:
        return PATTERN_TRANSLATION_HEURISTICS
    return (
        f"Primary pattern_type for this compilation task: \"{pt}\".\n"
        f"Apply the matching section in PATTERN → RULE ACTIVATION first.\n\n"
        f"{PATTERN_TRANSLATION_HEURISTICS}"
    )


def generator_user_prompt_body(
    *,
    payload_json: str,
    requested_rule_type: str,
    base_uri: str,
    pattern_type: str,
) -> str:
    """
    Generator user prompt body: compilation first, JSON last.
    """
    return f"""Compile the unmapped BPMN pattern below into ONE ODRL 2.2 fragment policy
using structural compilation — not direct element-by-element translation.

{GENERATOR_INTERNAL_REASONING}

{RULE_AND_TARGET_BINDING}

{heuristics_for_pattern(pattern_type)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT DATA (BPMN snapshot + hints)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{payload_json}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT ODRL OUTPUT RULES (after internal compilation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Complete all 8 compilation steps internally before writing JSON.
• Each rule MUST have at least one constraint (or action refinement).
• target MUST be {base_uri}/asset/<slug> for an existing activity in valid_asset_targets,
  OR {base_uri}/rules/<uid> from valid_b2p_rule_uids — never invent IRIs.
• action ∈ strict ODRL 2.2 vocabulary (use, transfer, read, modify, delete, execute, …).
• Top-level rule kind: exactly one of permission | prohibition | obligation matching
  "{requested_rule_type}".
• If no coherent governance rule exists after compilation, use prohibition + action use
  on a valid branch activity asset — do not fabricate permissions on invented targets.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT (FINAL STEP ONLY — no reasoning text)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After internal compilation is complete, output ONE JSON object and nothing else (no markdown):

{{
  "odrl_policy": {{
    "@context": "http://www.w3.org/ns/odrl.jsonld",
    "@type": "Set",
    "uid": "{base_uri}/policy:...",
    ...
  }}
}}

Do NOT include problem_interpretation, validation_notes, compilation_trace, or reasoning.
Only the final odrl_policy object."""


SURGICAL_CORRECTION_FOR_TEMPLATES = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SURGICAL CORRECTION ONLY — deterministic templates
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your task is SURGICAL CORRECTION only — not rewriting.

## Immutable elements — DO NOT modify under any circumstances:
- The overall structure and keys of the policy (@type, uid, permission/prohibition/obligation arrays, action, target, assignee, duty, consequence, etc.)
- The operand type and URI of any constraint (leftOperand value and namespace)
- The rightOperand value unless it is logically inconsistent with the operator
- Any value extracted from the BPMN context (IDs, labels, thresholds, parameters)
- Template-specific actions (odrl:enable, nextPolicy) and internal /rules/ targets linking FPa rules

## Allowed corrections — ONLY these:
- Replace a semantically incorrect operator within a constraint or action refinement
  (e.g., replace gt with lt if the logic requires it)
- Add a missing @type or @context declaration if absent
- Fix a malformed datatype annotation (e.g., xsd:integer vs xsd:string)
- Normalize an operator or leftOperand token to strict ODRL vocabulary when it is a typo
  (e.g., odrl:gt → gt) without changing semantic intent

## Operator correction rules:
When correcting an operator, you MUST:
1. Identify why the current operator is semantically wrong relative to the constraint intent
2. Replace it with the minimal fix (prefer the inverse: gt→lt, gteq→lteq, etc.)
3. Leave the rightOperand unchanged unless its type is syntactically invalid

## Hard constraints:
- If a constraint already makes semantic sense in its context, do NOT touch it
- Never restructure, reorder, or rename any JSON-LD key
- Never add or remove rules or constraints — only fix existing ones
- Never replace odrl:enable / nextPolicy with odrl:use or other actions
- Never collapse multiple rules into one generic business narrative
- Output the corrected policy in the exact same format as the input
- Your output MUST be compliant with strict ODRL vocabulary for operators and leftOperands
- Your output must differ from the input by the minimum number of tokens necessary
  to make the policy semantically valid. Any change beyond that is an error.
""".strip()


VALIDATOR_DETERMINISTIC_TEMPLATE_MODE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETERMINISTIC TEMPLATE MODE (Agent 4 pre-built FPd)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This policy was emitted by a deterministic ODRL template (XOR/AND/OR/sequence/message).
The structural compilation is ALREADY CORRECT by construction.

You must NOT reject or request restructuring because:
- actions use odrl:enable or nextPolicy (project activation model)
- targets reference /rules/ uids linking branch FPa rules (not only B2P uids)
- rule kinds or counts differ from unmapped-pattern heuristics

Validate ONLY:
□ Each constraint/refinement operator ∈ strict ODRL vocabulary (eq, neq, gt, lt, gteq, lteq, …)
□ Each leftOperand ∈ strict ODRL vocabulary (event, product, purpose, dateTime, …)
□ Operators are semantically consistent with their rightOperand (no impossible gt+string misuse)
□ No BPMN routing vocabulary in constraint values (gateway, fork, sequenceFlow, …)

Do NOT flag as errors:
- Template structure (permissions with enable + refinement, sequence duty/nextPolicy, etc.)
- Internal /rules/ targets that wire branch activation
- Supposed "branch collapse" or "missing asset target" when the template pattern is XOR/flow

Return "valid" unless a constraint operator or leftOperand is outside ODRL vocabulary
or semantically incoherent. Use "warning" only for weak but acceptable constraints.
""".strip()


def validator_user_prompt_for_deterministic_template() -> str:
    """Lightweight validator prompt: deterministic template constraints only."""
    return f"""{VALIDATOR_DETERMINISTIC_TEMPLATE_MODE}

{SURGICAL_CORRECTION_FOR_TEMPLATES}

You are performing constraint-level verification of a deterministic template policy.
Do NOT recompile BPMN into a new policy shape.
Do NOT validate JSON syntax or JSON-LD syntax."""


def semantic_repair_user_prompt_body(
    *,
    summary: str,
    validation_reports_json: str,
    warnings_json: str,
    hints_json: str,
    bundles_json: str,
    template_mode: bool = False,
) -> str:
    """Semantic repair prompt body for Agent 4 (surgical for templates)."""
    mode_block = (
        SURGICAL_CORRECTION_FOR_TEMPLATES
        if template_mode
        else (
            "Apply structural compilation fixes for unmapped patterns only. "
            "Prefer minimal edits; do not rewrite unrelated fields."
        )
    )
    return f"""Apply semantic corrections using the validation report below.

{mode_block}

Apply ONLY the fixes indicated by STRUCTURED HINTS field_path values.
If a hint has an empty field_path, do not guess — leave that aspect unchanged.

VALIDATION REPORT (human-readable):
{summary}

STRUCTURED VALIDATION REPORTS (per policy):
{validation_reports_json}

WARNINGS:
{warnings_json}

STRUCTURED HINTS (apply ONLY these targeted fixes):
{hints_json}

POLICIES TO REPAIR (public ODRL + metadata; internal _* keys are re-attached after merge):
{bundles_json}

Respond with exactly:
{{
  "corrected": [
    {{"policy_uid": "<uid matching input>", "odrl": {{ "@context": "http://www.w3.org/ns/odrl.jsonld", "...": "..." }}}}
  ]
}}
Include one entry per policy_uid in bundles.
Each `odrl` must preserve the input structure; change only fields referenced in hints
(or operator/leftOperand/@context fixes allowed above for template mode).
"""


def validator_user_prompt_preamble() -> str:
    """Initial block of the validator prompt (before data)."""
    return f"""{VALIDATOR_INTERNAL_RECONSTRUCTION}

{ANTI_STRUCTURAL_SHORTCUTS}

{PATTERN_TRANSLATION_HEURISTICS}

You are performing a critical second-pass verification of a compiled fragment policy.
The generator ran an internal compilation pipeline; you must independently recompile
from BPMN context and reject policies that only autocomplete ODRL shape.

You must NOT validate JSON syntax or JSON-LD syntax."""


def validator_user_prompt_epilogue() -> str:
    """JSON verdict — always the last section of the prompt."""
    return """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL VERDICT (OUTPUT THIS ONLY AFTER INTERNAL RECOMPILATION)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return:
- "valid" only if branch activation semantics, targets, and constraints match compiled BPMN logic.
- "warning" if mostly valid but structurally weak (e.g. missing distinct branch discrimination).
- "invalid" if gateway-as-target, branch collapse, invented IRIs, or BPMN routing in ODRL fields.

Output ONE JSON object only (no markdown, no reasoning text):

{
  "verdict": "valid" | "invalid" | "warning",
  "errors": ["..."],
  "warnings": ["..."]
}"""
