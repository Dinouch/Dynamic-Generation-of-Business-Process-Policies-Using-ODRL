"""
Policy Decision Point (PDP) ODRL simplifié pour l'exécution runtime.

Évalue permissions/prohibitions FPa et permissions ``odrl:enable`` des FPd
contre un ``ExecutionContext`` (variables métier, horloge, règles actives).
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Optional

from .models import ActivityDecision, ExecutionContext


def _normalize_operand(value: Any) -> Any:
    if isinstance(value, dict):
        if "@value" in value:
            return value["@value"]
        if "@id" in value:
            return value["@id"]
    return value


def _action_type(action: Any) -> str:
    if isinstance(action, str):
        return action.replace("odrl:", "")
    if isinstance(action, dict):
        val = action.get("rdf:value") or action.get("@id") or action
        return _action_type(val)
    return str(action)


def _iter_rule_actions(policy: dict) -> list[tuple[str, dict, dict]]:
    """Yields (rule_type, rule_dict, action_dict)."""
    for rt in ("permission", "prohibition", "obligation"):
        for rule in policy.get(rt) or []:
            if not isinstance(rule, dict):
                continue
            actions = rule.get("action") or []
            if not isinstance(actions, list):
                actions = [actions]
            for act in actions:
                if isinstance(act, dict):
                    yield rt, rule, act


def evaluate_constraint(constraint: dict, ctx: ExecutionContext) -> bool:
    lo = str(constraint.get("leftOperand") or "").lower()
    op = str(constraint.get("operator") or "eq").lower()
    ro = _normalize_operand(constraint.get("rightOperand"))

    if lo in ("datetime", "date", "time"):
        ro_raw = constraint.get("rightOperand")
        ro_type = ""
        if isinstance(ro_raw, dict):
            ro_type = str(ro_raw.get("@type") or "")
        ro_val = _normalize_operand(ro_raw)

        if "time" in ro_type.lower() or (
            isinstance(ro_val, str) and len(ro_val) <= 8 and ":" in str(ro_val)
        ):
            try:
                parts = str(ro_val).split(":")
                ro_time = time(int(parts[0]), int(parts[1]))
                ctx_time = ctx.now.time().replace(tzinfo=None)
                if op in ("gteq", "ge", ">="):
                    return ctx_time >= ro_time
                if op in ("lteq", "le", "<="):
                    return ctx_time <= ro_time
                if op == "eq":
                    return ctx_time == ro_time
            except (ValueError, IndexError):
                return True

        ctx_val = ctx.now.date() if isinstance(ctx.now, datetime) else ctx.now
        if isinstance(ro, str):
            try:
                ro_date = date.fromisoformat(ro[:10])
            except ValueError:
                return False
        else:
            ro_date = ro
        if op in ("gteq", "ge", ">="):
            return ctx_val >= ro_date
        if op in ("lteq", "le", "<="):
            return ctx_val <= ro_date
        if op == "eq":
            return ctx_val == ro_date
        return False

    if lo in ("spatial",):
        ctx_val = ctx.get("spatial") or "http://example.com/location/InternalNetwork"
        if op in ("eq", "="):
            return str(ctx_val) == str(ro) or str(ro).endswith(str(ctx_val).split("/")[-1])
        return False

    if lo == "event" and ro in ("odrl:policyUsage", "http://www.w3.org/ns/odrl/2.2/policyUsage"):
        return ctx.get("policy_usage_count", 0) > 0

    ctx_val = ctx.get(lo)
    if ctx_val is None:
        ctx_val = ctx.get("product") if lo == "product" else None

    if op in ("eq", "="):
        return str(ctx_val) == str(ro)
    if op in ("neq", "!="):
        return str(ctx_val) != str(ro)
    if op == "gt":
        try:
            return float(ctx_val) > float(ro)
        except (TypeError, ValueError):
            return False
    if op in ("gteq", "ge"):
        try:
            return float(ctx_val) >= float(ro)
        except (TypeError, ValueError):
            return False
    if op in ("lt",):
        try:
            return float(ctx_val) < float(ro)
        except (TypeError, ValueError):
            return False
    if op in ("lteq", "le"):
        try:
            return float(ctx_val) <= float(ro)
        except (TypeError, ValueError):
            return False
    return str(ctx_val) == str(ro)


def _constraints_hold(action: dict, ctx: ExecutionContext) -> bool:
    refinements = action.get("refinement") or action.get("constraint") or []
    if isinstance(refinements, dict):
        refinements = [refinements]
    if not refinements:
        return True
    return all(
        evaluate_constraint(c, ctx) for c in refinements if isinstance(c, dict)
    )


def _rule_id(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1] if uri else ""


class OdrlPolicyDecisionPoint:
    """Évalue FPa (exécution d'activité) et applique FPd (activation de règles)."""

    def __init__(self, rule_activity_index: Optional[dict[str, str]] = None):
        self.rule_activity_index = rule_activity_index or {}

    def apply_fpd_policies(
        self, fpd_policies: list[dict], ctx: ExecutionContext
    ) -> set[str]:
        """Met à jour ``ctx.enabled_rules`` selon les FPd dont les contraintes matchent."""
        newly_enabled: set[str] = set()
        for pol in fpd_policies:
            for _rt, rule, action in _iter_rule_actions(pol):
                if _action_type(action) != "enable":
                    continue
                if not _constraints_hold(action, ctx):
                    continue
                target = str(rule.get("target") or "")
                rid = _rule_id(target)
                if rid:
                    ctx.enabled_rules.add(rid)
                    newly_enabled.add(rid)
        return newly_enabled

    def can_execute_activity(
        self,
        activity: str,
        fpa_policies: list[dict],
        ctx: ExecutionContext,
        *,
        require_enabled_rule: bool = True,
        is_start_activity: bool = False,
    ) -> ActivityDecision:
        """
        Décide si une activité peut s'exécuter.

        - Les prohibitions ``odrl:execute`` bloquent si leurs contraintes sont satisfaites.
        - Les permissions ``odrl:execute`` autorisent si contraintes OK.
        - Si une règle gouverne l'activité, elle doit être dans ``enabled_rules`` (sauf 1ère activité).
        """
        governing: list[str] = []
        for pol in fpa_policies:
            for rt, rule, action in _iter_rule_actions(pol):
                at = _action_type(action)
                if at not in ("execute", "trigger"):
                    continue
                rid = _rule_id(str(rule.get("uid") or rule.get("target") or ""))
                if rid:
                    governing.append(rid)

        if require_enabled_rule and governing and not is_start_activity:
            if not any(g in ctx.enabled_rules for g in governing):
                return ActivityDecision(
                    activity=activity,
                    allowed=False,
                    reason=f"Aucune règle active parmi {governing}",
                    governing_rules=governing,
                )

        for pol in fpa_policies:
            for rt, rule, action in _iter_rule_actions(pol):
                at = _action_type(action)
                if rt == "prohibition" and at == "execute":
                    if _constraints_hold(action, ctx):
                        return ActivityDecision(
                            activity=activity,
                            allowed=False,
                            reason="Prohibition ODRL active (contrainte satisfaite)",
                            governing_rules=governing,
                        )
                if rt in ("permission", "obligation") and at in ("execute", "trigger"):
                    if is_start_activity:
                        continue
                    if not _constraints_hold(action, ctx):
                        return ActivityDecision(
                            activity=activity,
                            allowed=False,
                            reason="Permission/obligation ODRL : contrainte non satisfaite",
                            governing_rules=governing,
                        )

        return ActivityDecision(
            activity=activity,
            allowed=True,
            reason="Autorisé",
            governing_rules=governing,
        )
