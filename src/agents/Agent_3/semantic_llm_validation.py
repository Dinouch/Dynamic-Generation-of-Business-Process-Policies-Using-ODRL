"""
LLM business semantic validation (Agent 3).

Complements the deterministic layer: no JSON-LD / SHACL, focus on business meaning,
BPMN↔ODRL consistency, and conceptual use of the ODRL vocabulary.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from ..structural_analyzer import EnrichedGraph
from ..bpmn_odrl_reasoning_prompts import (
    VALIDATOR_INTERNAL_RECONSTRUCTION,
    validator_user_prompt_epilogue,
    validator_user_prompt_for_deterministic_template,
    validator_user_prompt_preamble,
)
from ..Agent_4.odrl_deterministic_templates import is_deterministic_template_fpd
from .semantic_deterministic_validation import (
    collect_known_asset_iris,
    collect_known_b2p_rule_uids,
)

VALIDATOR_SYSTEM = f"""
You are a strict ODRL 2.2 semantic validator for BPMN-compiled fragment policies.

Your role is NOT to validate JSON syntax or JSON-LD syntax.

You verify that the generator performed structural compilation:
  BPMN pattern → branch activities → conditional rule activation → ODRL constraints.

{VALIDATOR_INTERNAL_RECONSTRUCTION}

Evaluate only:
- compilation fidelity (branch-level activation, not gateway compression)
- semantic correctness and business meaning
- target hierarchy (existing assets / B2P rules only)
- ODRL conceptual compliance (not JSON shape)
- logical coherence and alignment with business intent

You are deliberately critical and adversarial.
Assume the policy is semantically incorrect until proven otherwise.

Reject policies that autocomplete ODRL shape without branch-level activation semantics.

Do not output your internal reconstruction or reasoning.
After internal analysis, output ONLY one JSON object with keys: verdict, errors, warnings.
Do not propose fixes in the output.
""".strip()

_BPMN_FORBIDDEN_IN_ODRL = re.compile(
    r"\b(gateway|gatewaytype|exclusivegateway|parallelgateway|inclusivegateway|"
    r"join|fork|sequenceflow|routing|firsteventwins|path\s+selection|event_based|"
    r"exclusive\s+gateway|parallel\s+gateway)\b",
    re.IGNORECASE,
)


def build_enriched_graph_context(graph: EnrichedGraph, fragment_id: str) -> dict[str, Any]:
    """Business BPMN snapshot for the fragment (no raw internal IDs)."""
    ctx = (graph.fragment_contexts or {}).get(fragment_id)
    out: dict[str, Any] = {
        "fragment_id": fragment_id,
        "activities": [],
        "connections": [],
        "unmapped_patterns": [],
        "bpmn_activity_slugs": [],
    }
    if ctx:
        out["activities"] = list(ctx.activities or [])
        for conn in ctx.connections or []:
            entry = {
                "from": getattr(conn, "from_activity", None),
                "to": getattr(conn, "to_activity", None),
                "type": getattr(conn, "connection_type", None),
                "condition": getattr(conn, "condition", None),
            }
            out["connections"].append({k: v for k, v in entry.items() if v is not None})

    slugs: set[str] = set(out["activities"])
    for fid, fctx in (graph.fragment_contexts or {}).items():
        slugs.update(fctx.activities or [])
    out["bpmn_activity_slugs"] = sorted(slugs)

    for up in getattr(graph, "unmapped_patterns", None) or []:
        if up.fragment_id != fragment_id and fragment_id not in (up.involved_fragment_ids or []):
            continue
        out["unmapped_patterns"].append(
            {
                "pattern_type": up.pattern_type,
                "gateway_name": up.gateway_name,
                "bpmn_semantic": up.bpmn_semantic,
                "description": up.description,
                "involved_activity_names": list(up.involved_activity_names or []),
            }
        )

    out["b2p_policy_uids"] = [p.get("uid") for p in (graph.raw_b2p or []) if p.get("uid")]
    out["valid_asset_targets"] = sorted(collect_known_asset_iris(graph))
    out["valid_b2p_rule_uids"] = sorted(collect_known_b2p_rule_uids(graph))
    out["target_reference_rules"] = (
        "Generated policies must NOT invent assets or rules. "
        "target must be either (1) http://example.com/asset/<activity_slug> where <activity_slug> "
        "is exactly one of bpmn_activity_slugs (canonical IRI listed in valid_asset_targets), or "
        "(2) http://example.com/rules/<ruleName> matching one of valid_b2p_rule_uids exactly."
    )
    return out


def build_reference_fragments(
    fp_results: dict,
    fragment_id: str,
    exclude_policy_uid: str,
    graph: EnrichedGraph,
) -> list[dict[str, Any]]:
    """Other batch policies (same fragment + B2P sources) as style references."""
    refs: list[dict[str, Any]] = []
    fps = fp_results.get(fragment_id)
    if not fps:
        return refs

    b2p_by_uid = {p.get("uid"): p for p in (graph.raw_b2p or []) if p.get("uid")}

    for pol in fps.all_policies():
        uid = pol.get("uid")
        if not uid or uid == exclude_policy_uid:
            continue
        pub = {k: v for k, v in pol.items() if not str(k).startswith("_")}
        entry: dict[str, Any] = {
            "policy_uid": uid,
            "policy_type": pol.get("_type"),
            "odrl": pub,
        }
        src = pol.get("_source_b2p")
        if src and src in b2p_by_uid:
            entry["b2p_source"] = b2p_by_uid[src]
        refs.append(entry)

    return refs[:8]


def resolve_business_intent(pol: dict, graph: EnrichedGraph) -> str:
    """Business intent declared at generation time or inferred from metadata."""
    for key in ("_business_intent", "_hint_text"):
        v = pol.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    src = pol.get("_source_b2p")
    if src:
        for p in graph.raw_b2p or []:
            if p.get("uid") == src:
                parts = []
                for rt in ("permission", "prohibition", "obligation"):
                    for rule in p.get(rt) or []:
                        if isinstance(rule, dict) and rule.get("action"):
                            parts.append(f"{rt}: action={rule.get('action')}")
                if parts:
                    return f"B2P governance for activity '{pol.get('_activity', '?')}' — " + "; ".join(
                        parts[:3]
                    )
                return f"B2P policy uid {src} mapped to activity '{pol.get('_activity', '?')}'."

    pattern = pol.get("_unmapped_pattern")
    if pattern:
        return (
            f"Unmapped BPMN pattern '{pattern}' "
            f"(gateway: {pol.get('_gateway_name', 'n/a')})."
        )
    return "No explicit business intent metadata; infer only from ODRL fields and BPMN context."


def build_validator_user_prompt(
    generated_policy: dict[str, Any],
    business_intent: str,
    enriched_graph_ctx: dict[str, Any],
    reference_fragments: list[dict[str, Any]],
) -> str:
    odrl_public = {k: v for k, v in generated_policy.items() if not str(k).startswith("_")}
    if is_deterministic_template_fpd(generated_policy):
        return f"""{validator_user_prompt_for_deterministic_template()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA FOR CONSTRAINT-LEVEL VERIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GENERATED POLICY (deterministic template — structure is immutable):
{json.dumps(odrl_public, ensure_ascii=False, indent=2)}

TEMPLATE METADATA (do not rewrite; context only):
{json.dumps({k: v for k, v in generated_policy.items() if str(k).startswith("_")}, ensure_ascii=False, indent=2)}

ENRICHED BPMN GRAPH:
{json.dumps(enriched_graph_ctx, ensure_ascii=False, indent=2)}

{validator_user_prompt_epilogue()}
"""

    return f"""{validator_user_prompt_preamble()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA FOR VERIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GENERATED POLICY:
{json.dumps(odrl_public, ensure_ascii=False, indent=2)}

BUSINESS INTENT (from generation / hints):
{business_intent}

ENRICHED BPMN GRAPH:
{json.dumps(enriched_graph_ctx, ensure_ascii=False, indent=2)}

REFERENCE ODRL FRAGMENTS:
{json.dumps(reference_fragments, ensure_ascii=False, indent=2)}

Complete internal reconstruction before applying the checks below.

==================================================
MANDATORY VALIDATION CHECKS (after internal reconstruction)
==================================================

1. TARGET CONSISTENCY WITH BUSINESS PROCESS (STRICT IRI RULES)

The generator must NOT create new assets or rules. Targets are references only.

Check carefully:

□ If target starts with http://example.com/asset/ — it MUST equal one entry in
  ENRICHED BPMN GRAPH → valid_asset_targets (one existing BPMN activity slug).
  Invented slugs such as documentation-outcome-record when no such activity exists → ERROR.

□ If target starts with http://example.com/rules/ — it MUST equal one entry in
  valid_b2p_rule_uids (existing B2P rule uid). The generator must not mint new rule IRIs → ERROR.

□ Does the chosen target match the business intent and the correct activity in scope?

If the target does not exist in valid_asset_targets or valid_b2p_rule_uids → ERROR.

==================================================

2. SEMANTIC ALIGNMENT WITH BUSINESS INTENT

Determine whether the policy preserves the operational meaning
of the business intent.

Check carefully:

□ Does the policy express the same governance rule as the intent?
□ Does the selected rule type (permission/prohibition/duty)
  match the intent?
□ Does the selected ODRL action represent the intended behavior?
□ Are important business conditions missing?
□ Does the policy introduce constraints not implied by the intent?
□ Does the policy weaken, distort, or over-restrict the original meaning?
□ Could a business analyst reasonably say:
  "Yes, this policy expresses the intended business rule"?

If the policy is technically valid but semantically misaligned
with the business intent → ERROR.

==================================================

3. BUSINESS PROCESS CONSISTENCY

Reason about the operational role of the policy inside the BPMN process.

Check carefully:

□ Does the policy correspond to a realistic governance rule
  in this business process?
□ Does the timing or context implied by the policy make sense?
□ Does the policy regulate something operationally meaningful?
□ Is the policy attached to the correct process concern
  (privacy, retention, payment, consent, auditability, etc.)?
□ Does the policy confuse BPMN control-flow logic
  with access-control or governance logic?

BPMN structural concepts MUST NOT appear as policy semantics.

Forbidden examples include:
- gateway
- gatewayType
- exclusiveGateway
- parallelGateway
- join
- fork
- sequenceFlow
- routing logic
- firstEventWins
- path selection

If BPMN execution logic is embedded into ODRL semantics → ERROR.

==================================================

4. ODRL CONCEPTUAL COMPLIANCE

Validate semantic compliance with ODRL concepts.

Check carefully:

□ Is the action semantically valid in ODRL?
□ Is the leftOperand semantically meaningful in ODRL?
□ Are constraints conceptually coherent?
□ Are operators used consistently with their operands?
□ Does the policy avoid semantic misuse of ODRL fields?
□ Are permissions/prohibitions/duties used correctly?

Examples of semantic misuse:
- using actions as targets
- using events as assets
- using BPMN elements as constraint values
- using process routing concepts as purposes
- attaching constraints that are semantically unrelated to the action

If ODRL concepts are semantically misused → ERROR.

==================================================

5. LOGICAL COHERENCE

Check whether the policy creates impossible or contradictory conditions.

Check carefully:

□ Are there multiple constraints that create impossible conditions?
□ Are there contradictory equality constraints?
□ Does the policy contain mutually exclusive conditions?
□ Would the rule ever be satisfiable in practice?

Example:
purpose = fraudDetection
AND
purpose = marketing

This creates an impossible conjunction and MUST be considered invalid.

==================================================

6. CONSISTENCY WITH REFERENCE FRAGMENTS

Compare the generated fragment with the reference fragments.

Check carefully:

□ Does the generated policy remain semantically compatible
  with the reference fragment style and abstraction level?
□ Does it preserve the same policy modeling logic?
□ Does it introduce concepts absent from the reference policy family?
□ Does it hallucinate governance semantics unrelated to the examples?

The generated fragment does NOT need to copy the reference fragments,
but it MUST remain semantically coherent with them.

7. STRUCTURAL COMPILATION / RULE ACTIVATION (critical)

After recompiling BPMN logic internally, verify:

□ Gateway semantics are modeled as conditional enablement of branch rules — NOT as
  target = gateway name / invented gateway asset.
□ For fork_xor / fork_event_based_exclusive: each branch activity has a distinguishable
  activation story (distinct target and/or distinct event/condition constraint).
□ Distinct downstream branch activities → distinct targets (unless explicit BPMN convergence).
□ Multiple ODRL rules in one policy must NOT share the same target unless branches converged.
□ Constraints express operational triggers (event, deadline, business predicate) — not
  gatewayType, firstEventWins, exclusive, sequenceFlow, routing, path selection.
□ No invented http://example.com/asset/* or http://example.com/rules/* IRIs.
□ For event-based patterns: ask "which rule is enabled when event E?" — reject if the
  policy only describes the gateway holistically.

8. BRANCH AND PATTERN SEMANTICS (summary)

□ Generated rules reflect reconstructed operational logic.
□ Exclusive branches are mutually distinguishable (not duplicate conditions on same target).
□ Policy structure matches pattern (XOR / event-based → typically one rule per branch path).
□ A BPM analyst would accept this as compiled governance, not compressed ODRL autocomplete.

{validator_user_prompt_epilogue()}
"""


def _infer_field_path(error_text: str) -> str:
    """Light heuristic to point Agent 4 to an ODRL field."""
    t = error_text.lower()
    for token in ("target", "action", "leftoperand", "rightoperand", "operator", "assignee", "assigner"):
        if token in t.replace(" ", ""):
            for rt in ("permission", "prohibition", "obligation"):
                if rt in t:
                    return f"{rt}[0].{token}"
            return token
    return ""


def verdict_to_hints_and_report(
    policy_uid: str,
    verdict_data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """
    Convert LLM output into Agent 4 hints + structured report.

    Returns
    -------
    hints, warnings, report_dict
    """
    verdict = str(verdict_data.get("verdict", "invalid")).strip().lower()
    errors = [str(e) for e in (verdict_data.get("errors") or []) if e]
    warns = [str(w) for w in (verdict_data.get("warnings") or []) if w]

    report = {
        "policy_uid": policy_uid,
        "verdict": verdict,
        "errors": errors,
        "warnings": warns,
    }

    hints: list[dict[str, Any]] = []
    if verdict == "invalid":
        for err in errors:
            hints.append(
                {
                    "policy_uid": policy_uid,
                    "field_path": _infer_field_path(err),
                    "issue": err,
                    "suggested_fix": "",
                    "odrl_template_key": "business_semantic",
                    "confidence": 0.92,
                }
            )
        if not errors:
            hints.append(
                {
                    "policy_uid": policy_uid,
                    "field_path": "",
                    "issue": "Semantic validation failed without detailed error messages.",
                    "suggested_fix": "",
                    "odrl_template_key": "business_semantic",
                    "confidence": 0.85,
                }
            )

    warning_msgs = list(warns)
    if verdict == "warning" and not warning_msgs:
        warning_msgs.append(
            f"Policy {policy_uid}: validator returned warning without detail."
        )

    return hints, warning_msgs, report


def quick_bpmn_leak_check(pol: dict) -> list[str]:
    """Light local check: BPMN terms in public ODRL fields."""
    leaks: list[str] = []
    pub = json.dumps({k: v for k, v in pol.items() if not str(k).startswith("_")})
    if _BPMN_FORBIDDEN_IN_ODRL.search(pub):
        leaks.append(
            "BPMN structural vocabulary detected inside ODRL policy fields "
            "(gateway, fork, join, sequenceFlow, routing, etc.)."
        )
    return leaks


def policy_needs_llm_business_semantic(pol: dict) -> bool:
    """
    B2P-aligned FPa: deterministic layer is sufficient.
    Deterministic FPd templates: constraints checked deterministically (no LLM recompilation).
    Business LLM mainly for unmapped FPd (LLM generation).
    """
    if is_deterministic_template_fpd(pol):
        return False
    if pol.get("_unmapped_pattern"):
        return True
    if pol.get("_type") == "FPd":
        return False
    if pol.get("_source_b2p"):
        return False
    return True


def run_business_semantic_llm_validation(
    fp_results: dict,
    graph: EnrichedGraph,
    call_llm: Callable[[str, str], str],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """
    Validate each generated policy via the business LLM judge.

    Returns
    -------
    hints, warnings, validation_reports
    """
    all_hints: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    all_reports: list[dict[str, Any]] = []
    llm_targets = 0
    skipped_fpa_b2p = 0

    for frag_id, fps in fp_results.items():
        for pol in fps.all_policies():
            pol_uid = str(pol.get("uid") or "")
            if not pol_uid:
                continue

            if not policy_needs_llm_business_semantic(pol):
                skipped_fpa_b2p += 1
                continue

            leaks = quick_bpmn_leak_check(pol)
            if leaks:
                for leak in leaks:
                    all_hints.append(
                        {
                            "policy_uid": pol_uid,
                            "field_path": "",
                            "issue": leak,
                            "suggested_fix": "",
                            "odrl_template_key": "business_semantic",
                            "confidence": 0.94,
                        }
                    )
                all_reports.append(
                    {
                        "policy_uid": pol_uid,
                        "verdict": "invalid",
                        "errors": leaks,
                        "warnings": [],
                        "source": "deterministic_bpmn_leak",
                        "fragment_id": frag_id,
                    }
                )
                continue

            llm_targets += 1
            print(
                f"[Agent 3] LLM semantic validation ({llm_targets}) — "
                f"{pol_uid[:72]}{'…' if len(pol_uid) > 72 else ''}"
            )
            business_intent = resolve_business_intent(pol, graph)
            graph_ctx = build_enriched_graph_context(graph, frag_id)
            refs = build_reference_fragments(fp_results, frag_id, pol_uid, graph)
            user_prompt = build_validator_user_prompt(pol, business_intent, graph_ctx, refs)

            try:
                raw = call_llm(VALIDATOR_SYSTEM, user_prompt)
                data = json.loads(raw)
            except Exception as e:
                all_warnings.append(f"Business semantic LLM skipped for {pol_uid}: {e}")
                continue

            hints, warns, report = verdict_to_hints_and_report(pol_uid, data)
            report["fragment_id"] = frag_id
            report["business_intent"] = business_intent
            report["source"] = "llm_business_semantic"
            all_reports.append(report)

            if hints:
                all_hints.extend(hints)
            all_warnings.extend(warns)

    if llm_targets or skipped_fpa_b2p:
        print(
            f"[Agent 3] LLM semantic validation complete: {llm_targets} call(s), "
            f"{skipped_fpa_b2p} B2P FPa skipped (deterministic only)."
        )

    return all_hints, all_warnings, all_reports
