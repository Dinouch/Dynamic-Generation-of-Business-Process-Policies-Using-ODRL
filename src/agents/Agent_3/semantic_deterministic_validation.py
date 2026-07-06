"""
Deterministic and batch-level semantic validation (Agent 3).

- FPa ↔ B2P alignment (_source_b2p): rule types, operators, parties, target vs activity.
- Global coherence: targeted contradictions, FPd vs graph, rule/policy references, activity homonymy.

No RDF/SHACL syntax validation; complements the LLM judge in constraint_validator.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from orchestration.graph import ActivityNode

from ..structural_analyzer import EnrichedGraph
from ..Agent_4.odrl_deterministic_templates import is_deterministic_template_fpd

BASE_URI = "http://example.com"

RULE_TYPES = ("permission", "prohibition", "obligation")

# Strict ODRL vocabulary checks (W3C ODRL Vocabulary & Expression 2.2).
# Intentionally disallow profile extensions in this project mode.
_ODRL_ALLOWED_OPERATORS = {
    "eq",
    "neq",
    "gt",
    "gteq",
    "lt",
    "lteq",
    "ispartof",
    "haspart",
    "isallof",
    "isanyof",
    "isnoneof",
    "isA".lower(),
}

# Keep this list focused on standard ODRL actions commonly used in generated policies.
# Non-standard actions (eg. enable, trigger, nextPolicy) should be rejected here.
_ODRL_ALLOWED_ACTIONS = {
    "use",
    "transfer",
    "read",
    "play",
    "display",
    "execute",
    "print",
    "reproduce",
    "distribute",
    "stream",
    "synchronize",
    "attribute",
    "compensate",
    "sharealike",
    "derive",
    "modify",
    "delete",
    "copy",
    "install",
    "move",
    "write",
}

_ODRL_ALLOWED_LEFT_OPERANDS = {
    "event",
    "dateTime".lower(),
    "elapsedTime".lower(),
    "count",
    "purpose",
    "recipient",
    "spatial",
    "industry",
    "language",
    "media",
    "meteredTime".lower(),
    "payAmount".lower(),
    "percentage",
    "product",
    "timeInterval".lower(),
    "unitOfCount".lower(),
    "version",
    "fileFormat".lower(),
}


def _uri_asset(activity_name: str) -> str:
    """Aligned with policy_projection_agent.uri_asset (avoids circular import)."""
    slug = activity_name.replace(" ", "_").replace("-", "_")
    return f"{BASE_URI}/asset/{slug}"


_ASSET_PREFIX = f"{BASE_URI}/asset/"
_RULES_PREFIX = f"{BASE_URI}/rules/"


def _normalize_iri_ref(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        raw = value.get("@id") or value.get("id")
        return str(raw).strip() if raw is not None else ""
    return ""


def collect_known_asset_iris(graph: EnrichedGraph) -> set[str]:
    """
    Allowed IRIs ``http://example.com/asset/<slug>``: one per BPMN activity in the enriched graph.
    Generators must not invent assets.
    """
    names: set[str] = set(_graph_activity_names(graph))
    for ctx in (graph.fragment_contexts or {}).values():
        names.update(ctx.activities or [])
    return {_uri_asset(n) for n in names if n}


def collect_known_b2p_rule_uids(graph: EnrichedGraph) -> set[str]:
    """Rule UIDs already defined in B2P policies (``uid`` field of ODRL rules)."""
    uids: set[str] = set()
    for pol in graph.raw_b2p or []:
        if not isinstance(pol, dict):
            continue
        for rt in RULE_TYPES:
            for rule in pol.get(rt) or []:
                if not isinstance(rule, dict):
                    continue
                uid = rule.get("uid")
                if isinstance(uid, str) and uid.strip():
                    uids.add(uid.strip())
    return uids


def _suggest_asset_for_policy(pol: dict, known_assets: set[str]) -> str:
    """First plausible canonical asset for Agent 4 correction."""
    for act in pol.get("_activities") or []:
        if isinstance(act, str) and act.strip():
            cand = _uri_asset(act.strip())
            if cand in known_assets:
                return cand
    act = pol.get("_activity")
    if isinstance(act, str) and act.strip():
        cand = _uri_asset(act.strip())
        if cand in known_assets:
            return cand
    return sorted(known_assets)[0] if known_assets else ""


def _rule_has_meaningful_constraints(rule: dict) -> bool:
    """At least one constraint (direct or refinement under action)."""
    return len(_collect_constraint_like_entries(rule)) > 0


def _validate_target_iri(
    iri: str,
    *,
    known_assets: set[str],
    known_b2p_rules: set[str],
    field_path: str,
    policy_uid: str,
    suggested_asset: str,
) -> list[dict]:
    """Validate ``target``: /asset/ → existing activity; /rules/ → existing B2P uid."""
    if not iri or not iri.startswith("http"):
        return []

    hints: list[dict] = []
    if iri.startswith(_ASSET_PREFIX):
        if iri not in known_assets:
            slug = iri[len(_ASSET_PREFIX) :]
            hints.append(
                {
                    "policy_uid": policy_uid,
                    "field_path": field_path,
                    "issue": (
                        f"target '{iri}' references an invented asset (slug '{slug}'). "
                        "Only enriched-graph activities are allowed under "
                        f"{_ASSET_PREFIX}<activity> — the generator must not create assets."
                    ),
                    "suggested_fix": suggested_asset,
                    "odrl_template_key": "target_asset_activity",
                    "confidence": 0.98,
                }
            )
        return hints

    if iri.startswith(_RULES_PREFIX):
        if iri not in known_b2p_rules:
            hints.append(
                {
                    "policy_uid": policy_uid,
                    "field_path": field_path,
                    "issue": (
                        f"target '{iri}' is not the uid of an existing B2P rule. "
                        "References under /rules/ must reuse a uid already present "
                        "in enriched_graph.raw_b2p (e.g. http://example.com/rules/largeAmountRule) "
                        "— the generator must not create rules."
                    ),
                    "suggested_fix": sorted(known_b2p_rules)[0] if known_b2p_rules else "",
                    "odrl_template_key": "target_b2p_rule_uid",
                    "confidence": 0.98,
                }
            )
        return hints

    return hints


def strict_target_reference_checks(
    fp_results: dict,
    graph: EnrichedGraph,
) -> tuple[list[dict], list[str]]:
    """
    Required ``target`` references:
    - ``.../asset/<slug>`` → existing activity in the enriched graph;
    - ``.../rules/<...>`` → uid of an existing B2P rule.

    FPd policies must have at least one constraint per rule (no empty rules).
    """
    hints: list[dict] = []
    warnings: list[str] = []
    known_assets = collect_known_asset_iris(graph)
    known_b2p_rules = collect_known_b2p_rule_uids(graph)

    if not known_assets:
        warnings.append(
            "No activity found in the enriched graph — /asset/ target validation impossible."
        )

    for frag_id, fps in fp_results.items():
        for pol in fps.all_policies():
            pol_uid = str(pol.get("uid") or "")
            if not pol_uid:
                continue
            is_fpd = pol.get("_type") == "FPd"
            is_template_fpd = is_deterministic_template_fpd(pol)
            suggested_asset = _suggest_asset_for_policy(pol, known_assets)

            for rt, idx, rule in _iter_rule_entries_with_paths(pol):
                if not _rule_has_meaningful_constraints(rule):
                    if is_fpd and not is_template_fpd:
                        hints.append(
                            {
                                "policy_uid": pol_uid,
                                "field_path": f"{rt}[{idx}].constraint",
                                "issue": (
                                    "FPd rule without constraint: a decision fragment policy "
                                    "must express at least one condition (constraint or refinement) — "
                                    "a bare rule is invalid."
                                ),
                                "suggested_fix": "",
                                "odrl_template_key": "fpd_missing_constraint",
                                "confidence": 0.97,
                            }
                        )

                if is_template_fpd:
                    continue

                targets: list[tuple[str, str]] = []
                t = _normalize_iri_ref(rule.get("target"))
                if t:
                    targets.append((f"{rt}[{idx}].target", t))
                for di, duty in enumerate(rule.get("duty") or []):
                    if isinstance(duty, dict):
                        dt = _normalize_iri_ref(duty.get("target"))
                        if dt:
                            targets.append((f"{rt}[{idx}].duty[{di}].target", dt))

                for field_path, iri in targets:
                    hints.extend(
                        _validate_target_iri(
                            iri,
                            known_assets=known_assets,
                            known_b2p_rules=known_b2p_rules,
                            field_path=field_path,
                            policy_uid=pol_uid,
                            suggested_asset=suggested_asset,
                        )
                    )

    return hints, warnings


def _norm_operator(op: Any) -> str:
    if op is None:
        return ""
    if isinstance(op, dict):
        v = op.get("@id") or op.get("id")
        if isinstance(v, str):
            return v.split(":")[-1].strip().lower() if ":" in v else v.strip().lower()
        return str(v).strip().lower() if v is not None else ""
    return str(op).strip().lower()


def _norm_vocab_token(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, dict):
        raw = v.get("@id") or v.get("id") or v.get("value")
        if isinstance(raw, str):
            t = raw.strip()
        else:
            t = str(raw).strip() if raw is not None else ""
    else:
        t = str(v).strip()
    if t.startswith("http://www.w3.org/ns/odrl/2/"):
        t = t.rsplit("/", 1)[-1]
    elif ":" in t:
        t = t.split(":")[-1]
    return t.strip().lower()


def _iter_rule_entries_with_paths(pol: dict) -> list[tuple[str, int, dict]]:
    out: list[tuple[str, int, dict]] = []
    for rt in RULE_TYPES:
        rules = pol.get(rt)
        if not isinstance(rules, list):
            continue
        for i, rule in enumerate(rules):
            if isinstance(rule, dict):
                out.append((rt, i, rule))
    return out


def strict_odrl_vocabulary_checks(fp_results: dict) -> tuple[list[dict], list[str]]:
    """
    Deterministic strict checks against ODRL vocabulary terms.

    Enforced policy for this project:
    - No profile extension allowed.
    - Only ODRL vocabulary terms accepted for actions/operators/leftOperand.
    """
    hints: list[dict] = []
    warnings: list[str] = []

    for frag_id, fps in fp_results.items():
        for pol in fps.all_policies():
            pol_uid = str(pol.get("uid", ""))
            if not pol_uid:
                continue

            if is_deterministic_template_fpd(pol):
                continue

            if pol.get("profile"):
                hints.append(
                    {
                        "policy_uid": pol_uid,
                        "field_path": "profile",
                        "issue": "ODRL profile extensions are disabled in strict vocabulary mode.",
                        "suggested_fix": "",
                        "odrl_template_key": "constraint_right_operand",
                        "confidence": 0.99,
                    }
                )

            for rt, i, rule in _iter_rule_entries_with_paths(pol):
                action = rule.get("action")
                action_path = f"{rt}[{i}].action"

                def _append_action_hint(raw_action: Any) -> None:
                    tok = _norm_vocab_token(raw_action)
                    if not tok:
                        return
                    if tok not in _ODRL_ALLOWED_ACTIONS:
                        hints.append(
                            {
                                "policy_uid": pol_uid,
                                "field_path": action_path,
                                "issue": (
                                    f"Action '{tok}' is outside strict ODRL vocabulary "
                                    "(no profile extension allowed)."
                                ),
                                "suggested_fix": "use",
                                "odrl_template_key": "action_id",
                                "confidence": 0.98,
                            }
                        )

                if isinstance(action, list):
                    for a in action:
                        if isinstance(a, dict) and "rdf:value" in a:
                            _append_action_hint(a.get("rdf:value"))
                        else:
                            _append_action_hint(a)
                elif isinstance(action, dict) and "rdf:value" in action:
                    _append_action_hint(action.get("rdf:value"))
                else:
                    _append_action_hint(action)

                constraints = rule.get("constraint")
                if isinstance(constraints, dict):
                    constraints = [constraints]
                if isinstance(constraints, list):
                    for ci, c in enumerate(constraints):
                        if not isinstance(c, dict):
                            continue
                        op = _norm_vocab_token(c.get("operator"))
                        if op and op not in _ODRL_ALLOWED_OPERATORS:
                            hints.append(
                                {
                                    "policy_uid": pol_uid,
                                    "field_path": f"{rt}[{i}].constraint[{ci}].operator",
                                    "issue": (
                                        f"Operator '{op}' is outside strict ODRL vocabulary."
                                    ),
                                    "suggested_fix": "eq",
                                    "odrl_template_key": "constraint_operator",
                                    "confidence": 0.97,
                                }
                            )

                        lo = _norm_vocab_token(c.get("leftOperand"))
                        if lo and lo not in _ODRL_ALLOWED_LEFT_OPERANDS:
                            # Use "event" as a neutral fallback operand for process constraints.
                            hints.append(
                                {
                                    "policy_uid": pol_uid,
                                    "field_path": f"{rt}[{i}].constraint[{ci}].leftOperand",
                                    "issue": (
                                        f"leftOperand '{lo}' is outside strict ODRL vocabulary "
                                        "(likely profile-specific/custom term)."
                                    ),
                                    "suggested_fix": "event",
                                    "odrl_template_key": "constraint_left_operand",
                                    "confidence": 0.96,
                                }
                            )

            if any(str(k).startswith("bpmn:") for k in pol.keys()):
                warnings.append(
                    f"[{frag_id}] Policy '{pol_uid}' contains BPMN-prefixed fields at top level; "
                    "strict ODRL vocabulary mode ignores non-ODRL terms."
                )

    return hints, warnings


def _append_constraint_vocab_hints(
    hints: list[dict],
    *,
    pol_uid: str,
    field_path: str,
    constraint: dict,
) -> None:
    op = _norm_vocab_token(constraint.get("operator"))
    if op and op not in _ODRL_ALLOWED_OPERATORS:
        hints.append(
            {
                "policy_uid": pol_uid,
                "field_path": f"{field_path}.operator",
                "issue": f"Operator '{op}' is outside strict ODRL vocabulary.",
                "suggested_fix": "eq",
                "odrl_template_key": "constraint_operator",
                "confidence": 0.97,
            }
        )
    lo = _norm_vocab_token(constraint.get("leftOperand"))
    if lo and lo not in _ODRL_ALLOWED_LEFT_OPERANDS:
        hints.append(
            {
                "policy_uid": pol_uid,
                "field_path": f"{field_path}.leftOperand",
                "issue": (
                    f"leftOperand '{lo}' is outside strict ODRL vocabulary "
                    "(likely profile-specific/custom term)."
                ),
                "suggested_fix": "product",
                "odrl_template_key": "constraint_left_operand",
                "confidence": 0.96,
            }
        )


def strict_template_constraint_vocabulary_checks(
    fp_results: dict,
) -> tuple[list[dict], list[str]]:
    """
    For deterministic FPd templates: check ONLY operators and leftOperands
    of constraints / refinements — never action, target, or structure.
    """
    hints: list[dict] = []
    warnings: list[str] = []

    for _frag_id, fps in fp_results.items():
        for pol in fps.all_policies():
            if not is_deterministic_template_fpd(pol):
                continue
            pol_uid = str(pol.get("uid") or "")
            if not pol_uid:
                continue

            for rt, i, rule in _iter_rule_entries_with_paths(pol):
                constraints = rule.get("constraint")
                if isinstance(constraints, dict):
                    constraints = [constraints]
                if isinstance(constraints, list):
                    for ci, c in enumerate(constraints):
                        if isinstance(c, dict):
                            _append_constraint_vocab_hints(
                                hints,
                                pol_uid=pol_uid,
                                field_path=f"{rt}[{i}].constraint[{ci}]",
                                constraint=c,
                            )

                actions = rule.get("action")
                if actions is None:
                    continue
                if not isinstance(actions, list):
                    actions = [actions]
                for ai, a in enumerate(actions):
                    if not isinstance(a, dict):
                        continue
                    refs = a.get("refinement")
                    if isinstance(refs, dict):
                        refs = [refs]
                    if isinstance(refs, list):
                        for ri, ref in enumerate(refs):
                            if isinstance(ref, dict):
                                _append_constraint_vocab_hints(
                                    hints,
                                    pol_uid=pol_uid,
                                    field_path=f"{rt}[{i}].action[{ai}].refinement[{ri}]",
                                    constraint=ref,
                                )

    return hints, warnings


def _collect_constraint_like_entries(rule: dict) -> list[dict]:
    """Direct constraints + refinements under action[]."""
    out: list[dict] = []
    if not isinstance(rule, dict):
        return out
    c = rule.get("constraint")
    if isinstance(c, list):
        out.extend(x for x in c if isinstance(x, dict))
    elif isinstance(c, dict):
        out.append(c)
    actions = rule.get("action")
    if actions is None:
        return out
    if not isinstance(actions, list):
        actions = [actions]
    for a in actions:
        if not isinstance(a, dict):
            continue
        refs = a.get("refinement")
        if isinstance(refs, list):
            out.extend(x for x in refs if isinstance(x, dict))
        elif isinstance(refs, dict):
            out.append(refs)
    return out


def _operators_sequence(rule: dict) -> list[str]:
    return [_norm_operator(x.get("operator")) for x in _collect_constraint_like_entries(rule)]


def _operator_field_paths(rule_type: str, rule_idx: int, rule: dict) -> list[str]:
    """JSON paths aligned with the actual rule structure (constraint vs refinement)."""
    paths: list[str] = []
    c = rule.get("constraint")
    if isinstance(c, list):
        for i, item in enumerate(c):
            if isinstance(item, dict) and "operator" in item:
                paths.append(f"{rule_type}[{rule_idx}].constraint[{i}].operator")
    elif isinstance(c, dict) and "operator" in c:
        paths.append(f"{rule_type}[{rule_idx}].constraint[0].operator")
    actions = rule.get("action")
    if not isinstance(actions, list):
        actions = [actions] if isinstance(actions, dict) else []
    for ai, a in enumerate(actions):
        if not isinstance(a, dict):
            continue
        refs = a.get("refinement")
        if isinstance(refs, list):
            for ri, item in enumerate(refs):
                if isinstance(item, dict) and "operator" in item:
                    paths.append(
                        f"{rule_type}[{rule_idx}].action[{ai}].refinement[{ri}].operator"
                    )
        elif isinstance(refs, dict) and "operator" in refs:
            paths.append(f"{rule_type}[{rule_idx}].action[{ai}].refinement[0].operator")
    return paths


def _activity_names_ambiguous(graph: EnrichedGraph) -> set[str]:
    """Activity names shared by multiple activity_id values (homonymy)."""
    by_name: dict[str, set[str]] = defaultdict(set)
    for m in graph.b2p_mappings.values():
        by_name[m.activity_name].add(m.activity_id)
    return {name for name, ids in by_name.items() if len(ids) > 1}


def _graph_activity_names(graph: EnrichedGraph) -> set[str]:
    names: set[str] = set()
    for node in graph.graph.all_nodes():
        if isinstance(node, ActivityNode):
            names.add(node.name)
    return names


def _build_uid_indexes(fp_results: dict) -> tuple[set[str], set[str]]:
    policy_uids: set[str] = set()
    rule_uids: set[str] = set()
    for fps in fp_results.values():
        for pol in fps.all_policies():
            pu = pol.get("uid")
            if isinstance(pu, str) and pu:
                policy_uids.add(pu)
            for rt in RULE_TYPES:
                for rule in pol.get(rt) or []:
                    if isinstance(rule, dict):
                        ru = rule.get("uid")
                        if isinstance(ru, str) and ru:
                            rule_uids.add(ru)
    return policy_uids, rule_uids


def _collect_string_refs(obj: Any, out: list[str]) -> None:
    if isinstance(obj, str):
        if obj.startswith("http://") or obj.startswith("https://"):
            out.append(obj)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("_"):
                continue
            _collect_string_refs(v, out)
        return
    if isinstance(obj, list):
        for x in obj:
            _collect_string_refs(x, out)


def _fpd_ref_warnings(
    fpd: dict,
    frag_id: str,
    policy_uids: set[str],
    rule_uids: set[str],
    known_b2p_rules: set[str] | None = None,
) -> list[str]:
    """References /rules/ or /policy: outside batch and B2P (informational)."""
    refs: list[str] = []
    body = {k: v for k, v in fpd.items() if not str(k).startswith("_")}
    _collect_string_refs(body, refs)
    seen = set()
    w: list[str] = []
    uid_fpd = fpd.get("uid", "?")
    b2p_rules = known_b2p_rules or set()
    for r in refs:
        if r in seen:
            continue
        seen.add(r)
        if r in policy_uids or r in rule_uids or r in b2p_rules:
            continue
        if r.startswith(_RULES_PREFIX) or "/policy:" in r:
            w.append(
                f"[FPd {frag_id}] Policy '{uid_fpd}' references unknown UID "
                f"(neither generated batch nor B2P): {r}"
            )
    return w


def deterministic_fpa_b2p_alignment(
    pol: dict,
    frag_id: str,
    b2p: dict,
    graph: EnrichedGraph,
) -> tuple[list[dict], list[str]]:
    """
    Compare an FPa (ODRL) to its B2P source.

    Returns (hints for Agent 4, informational warnings).
    """
    hints: list[dict] = []
    warnings: list[str] = []
    pol_uid = pol.get("uid") or ""
    activity = pol.get("_activity") or ""

    expected_target = _uri_asset(activity) if activity else ""

    for rt in RULE_TYPES:
        b_rules = b2p.get(rt) if isinstance(b2p.get(rt), list) else []
        o_rules = pol.get(rt) if isinstance(pol.get(rt), list) else []
        if b_rules and not o_rules:
            warnings.append(
                f"FPa {pol_uid}: B2P defines '{rt}' but the generated policy does not contain this rule type."
            )
            continue
        if not b_rules:
            continue
        n = min(len(b_rules), len(o_rules))
        for i in range(n):
            br = b_rules[i] if isinstance(b_rules[i], dict) else {}
            orule = o_rules[i] if isinstance(o_rules[i], dict) else {}

            ot = orule.get("target")
            bt = br.get("target")
            if isinstance(bt, str) and isinstance(ot, str) and bt and ot and bt != ot:
                hints.append(
                    {
                        "policy_uid": pol_uid,
                        "field_path": f"{rt}[{i}].target",
                        "issue": "ODRL target differs from the target in the source B2P rule.",
                        "suggested_fix": bt,
                        "odrl_template_key": "constraint_right_operand",
                        "confidence": 0.9,
                    }
                )
            elif (
                activity
                and expected_target
                and isinstance(ot, str)
                and ot
                and ot != expected_target
                and (not isinstance(bt, str) or not bt or bt == expected_target)
            ):
                hints.append(
                    {
                        "policy_uid": pol_uid,
                        "field_path": f"{rt}[{i}].target",
                        "issue": (
                            f"ODRL target ({ot}) does not match the canonical asset for "
                            f"activity '{activity}'."
                        ),
                        "suggested_fix": expected_target,
                        "odrl_template_key": "constraint_right_operand",
                        "confidence": 0.85,
                    }
                )

            for field in ("assigner", "assignee"):
                bv, ov = br.get(field), orule.get(field)
                if isinstance(bv, str) and bv and isinstance(ov, str) and ov and bv != ov:
                    hints.append(
                        {
                            "policy_uid": pol_uid,
                            "field_path": f"{rt}[{i}].{field}",
                            "issue": f"{field} does not match the B2P source.",
                            "suggested_fix": bv,
                            "odrl_template_key": "constraint_right_operand",
                            "confidence": 0.9,
                        }
                    )

            b_ops = _operators_sequence(br)
            o_ops = _operators_sequence(orule)
            if b_ops and o_ops and b_ops != o_ops:
                o_paths = _operator_field_paths(rt, i, orule)
                for j, (bo, oo) in enumerate(zip(b_ops, o_ops)):
                    if bo != oo:
                        fpath = o_paths[j] if j < len(o_paths) else f"{rt}[{i}].constraint[{j}].operator"
                        hints.append(
                            {
                                "policy_uid": pol_uid,
                                "field_path": fpath,
                                "issue": f"ODRL operator '{oo}' ≠ B2P '{bo}'.",
                                "suggested_fix": bo,
                                "odrl_template_key": "constraint_operator",
                                "confidence": 0.95,
                            }
                        )
                        break

    return hints, warnings


def batch_semantic_checks(
    fp_results: dict,
    graph: EnrichedGraph,
) -> tuple[list[dict], list[str]]:
    """
    Batch coherence: FPa target contradictions, FPd vs graph, references, XOR.

    Hints are limited to cases with a clear ODRL path; the rest go to warnings.
    """
    hints: list[dict] = []
    warnings: list[str] = []

    policy_uids, rule_uids = _build_uid_indexes(fp_results)
    known_b2p_rules = collect_known_b2p_rule_uids(graph)
    act_in_graph = _graph_activity_names(graph)
    amb_names = _activity_names_ambiguous(graph)
    if amb_names:
        warnings.append(
            "Activity homonymy (same name, multiple activity_id): "
            f"{', '.join(sorted(amb_names)[:12])}"
            f"{' …' if len(amb_names) > 12 else ''} — FPa/FPd alignment risk."
        )

    for frag_id, fps in fp_results.items():
        for pol in fps.fpa_policies:
            if pol.get("_source_b2p"):
                continue
            uid = pol.get("uid", "?")
            warnings.append(
                f"[{frag_id}] FPa '{uid}' without _source_b2p — minimal / default policy, "
                "not aligned to an explicit B2P entry."
            )

    target_modes: dict[str, set[str]] = defaultdict(set)
    for frag_id, fps in fp_results.items():
        for pol in fps.fpa_policies:
            for rt in ("permission", "prohibition"):
                for rule in pol.get(rt) or []:
                    if not isinstance(rule, dict):
                        continue
                    t = rule.get("target")
                    if isinstance(t, str) and t:
                        target_modes[t].add(rt)
    for tgt, modes in target_modes.items():
        if "permission" in modes and "prohibition" in modes:
            warnings.append(
                f"Global coherence: target '{tgt}' has both permission and prohibition in FPa policies."
            )

    seq_pairs = {
        (c.from_activity, c.to_activity)
        for c in graph.connections
        if c.connection_type in ("sequence", "message")
    }

    for frag_id, fps in fp_results.items():
        for fpd in fps.fpd_policies:
            uid = fpd.get("uid", "?")

            for act in fpd.get("_activities") or []:
                if act not in act_in_graph:
                    warnings.append(
                        f"[FPd {frag_id} '{uid}'] Activity '{act}' missing from enriched graph."
                    )

            if fpd.get("_flow") == "sequence":
                acts = fpd.get("_activities") or []
                if len(acts) >= 2:
                    pair = (acts[0], acts[1])
                    if pair not in seq_pairs:
                        warnings.append(
                            f"[FPd {frag_id} '{uid}'] Sequence {acts[0]} → {acts[1]} "
                            "not found as sequence/message connection in graph."
                        )

            if fpd.get("_gateway") == "XOR":
                conds = fpd.get("_conditions") or []
                if len(conds) >= 2 and conds[0] == conds[1]:
                    warnings.append(
                        f"[FPd {frag_id} '{uid}'] XOR: identical conditions ({conds[0]!r})."
                    )

            warnings.extend(
                _fpd_ref_warnings(
                    fpd, frag_id, policy_uids, rule_uids, known_b2p_rules=known_b2p_rules
                )
            )

    return hints, warnings


_EXCLUSIVE_BRANCH_PATTERNS = frozenset({"fork_xor", "fork_event_based_exclusive"})


def _slugify_activity(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_").lower()


def _target_is_gateway_asset(gateway_name: str, target_iri: str) -> bool:
    """Detect a target pointing at the gateway instead of a branch activity."""
    if not gateway_name or not target_iri:
        return False
    if not target_iri.startswith(_ASSET_PREFIX):
        return False
    slug = target_iri[len(_ASSET_PREFIX) :].lower()
    gw_slug = _slugify_activity(gateway_name)
    return slug == gw_slug or gateway_name.replace("-", "_").lower() == slug


def _target_matches_branch_activity(target_iri: str, branch_activities: set[str]) -> bool:
    if not target_iri.startswith(_ASSET_PREFIX) or not branch_activities:
        return True
    slug = target_iri[len(_ASSET_PREFIX) :].lower()
    for act in branch_activities:
        if _slugify_activity(act) == slug:
            return True
    return False


def rule_activation_compilation_checks(
    fp_results: dict,
    graph: EnrichedGraph,
) -> tuple[list[dict], list[str]]:
    """
    Verify the compilation model: conditional activation per branch,
    not gateway compression / single target for multiple rules.
    """
    hints: list[dict] = []
    warnings: list[str] = []

    branch_by_key: dict[tuple[str, str, str], set[str]] = {}
    for up in getattr(graph, "unmapped_patterns", None) or []:
        acts = set(up.involved_activity_names or [])
        keys = [(up.fragment_id, up.gateway_name or "", up.pattern_type or "")]
        for fid in up.involved_fragment_ids or []:
            keys.append((fid, up.gateway_name or "", up.pattern_type or ""))
        for key in keys:
            if acts:
                branch_by_key.setdefault(key, set()).update(acts)

    for frag_id, fps in fp_results.items():
        for pol in fps.all_policies():
            pol_uid = str(pol.get("uid") or "")
            if not pol_uid:
                continue

            if is_deterministic_template_fpd(pol):
                continue

            pattern = str(pol.get("_unmapped_pattern") or "").strip().lower()
            gateway = str(pol.get("_gateway_name") or "").strip()
            if pattern not in _EXCLUSIVE_BRANCH_PATTERNS and not gateway:
                continue

            rules = _iter_rule_entries_with_paths(pol)
            if not rules:
                continue

            target_entries: list[tuple[str, str]] = []
            for rt, idx, rule in rules:
                t = _normalize_iri_ref(rule.get("target"))
                if t:
                    target_entries.append((f"{rt}[{idx}].target", t))

            if not target_entries:
                continue

            branch_acts = branch_by_key.get((frag_id, gateway, pattern), set())

            for field_path, t in target_entries:
                if gateway and _target_is_gateway_asset(gateway, t):
                    hints.append(
                        {
                            "policy_uid": pol_uid,
                            "field_path": field_path,
                            "issue": (
                                "Invalid compilation: target points at BPMN gateway "
                                f"'{gateway}' instead of a governable branch activity. "
                                "Policies activate branch rules; they do not describe the gateway."
                            ),
                            "suggested_fix": "",
                            "odrl_template_key": "business_semantic",
                            "confidence": 0.96,
                        }
                    )
                elif branch_acts and not _target_matches_branch_activity(t, branch_acts):
                    hints.append(
                        {
                            "policy_uid": pol_uid,
                            "field_path": field_path,
                            "issue": (
                                "Target does not match any expected branch activity for this "
                                f"pattern '{pattern}' (gateway '{gateway}'). "
                                f"Expected branch activities: {sorted(branch_acts)}."
                            ),
                            "suggested_fix": "",
                            "odrl_template_key": "business_semantic",
                            "confidence": 0.9,
                        }
                    )

            target_iris = [t for _, t in target_entries]
            unique_targets = set(target_iris)

            if len(rules) > 1 and len(unique_targets) == 1:
                hints.append(
                    {
                        "policy_uid": pol_uid,
                        "field_path": "permission[0].target",
                        "issue": (
                            "Structural compression forbidden: multiple ODRL rules share "
                            "the same target while the BPMN pattern has distinct branches. "
                            "Each branch must have its own target (activity asset)."
                        ),
                        "suggested_fix": "",
                        "odrl_template_key": "business_semantic",
                        "confidence": 0.95,
                    }
                )

            if len(rules) > 1 and len(unique_targets) < len(target_iris):
                hints.append(
                    {
                        "policy_uid": pol_uid,
                        "field_path": "",
                        "issue": (
                            "Undistinguished branches: rules in the same policy reuse "
                            "identical targets without explicit BPMN convergence."
                        ),
                        "suggested_fix": "",
                        "odrl_template_key": "business_semantic",
                        "confidence": 0.88,
                    }
                )

    return hints, warnings


def run_deterministic_semantic_validation(
    fp_results: dict,
    graph: EnrichedGraph,
) -> tuple[list[dict], list[str]]:
    """
    Entry point: FPa↔B2P per policy, then batch checks.

    Returns
    -------
    hints
        Dicts compatible with SemanticHint / Agent 4 semantic_hints.
    warnings
        Informational messages (homonymy, structural absences, etc.).
    """
    all_hints: list[dict] = []
    all_warnings: list[str] = []

    b2p_by_uid = {p.get("uid"): p for p in graph.raw_b2p if p.get("uid")}

    for frag_id, fps in fp_results.items():
        for pol in fps.fpa_policies:
            src = pol.get("_source_b2p")
            if not src:
                continue
            b2p = b2p_by_uid.get(src)
            if not b2p:
                all_warnings.append(
                    f"[{frag_id}] FPa '{pol.get('uid')}': _source_b2p '{src}' not found in raw_b2p."
                )
                continue
            h, w = deterministic_fpa_b2p_alignment(pol, frag_id, b2p, graph)
            all_hints.extend(h)
            all_warnings.extend(w)

    h2, w2 = batch_semantic_checks(fp_results, graph)
    all_hints.extend(h2)
    all_warnings.extend(w2)

    h3, w3 = strict_odrl_vocabulary_checks(fp_results)
    all_hints.extend(h3)
    all_warnings.extend(w3)

    h3t, w3t = strict_template_constraint_vocabulary_checks(fp_results)
    all_hints.extend(h3t)
    all_warnings.extend(w3t)

    h4, w4 = strict_target_reference_checks(fp_results, graph)
    all_hints.extend(h4)
    all_warnings.extend(w4)

    h5, w5 = rule_activation_compilation_checks(fp_results, graph)
    all_hints.extend(h5)
    all_warnings.extend(w5)

    return all_hints, all_warnings


def merge_semantic_hints(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """Deduplicate by (policy_uid, field_path, suggested_fix)."""
    seen: set[tuple[str, str, str]] = set()
    out: list[dict] = []
    for h in primary + secondary:
        key = (
            str(h.get("policy_uid", "")),
            str(h.get("field_path", "")),
            str(h.get("suggested_fix", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out
