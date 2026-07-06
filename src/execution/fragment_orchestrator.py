"""
Orchestrateur de fragments : parcours du graphe BPMN avec gouvernance ODRL.

Utilise ``fragments_enhanced.json`` (connexions, conditions) et le PDP ODRL.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .models import ExecutionContext, ExecutionResult, ExecutionStep, RuntimeMode
from .odrl_pdp import OdrlPolicyDecisionPoint
from .policy_loader import (
    PolicyBundle,
    build_activity_fpa_index,
    build_rule_activity_index,
)

logger = logging.getLogger(__name__)


class FragmentOrchestrator:
    """Exécution simulée pas-à-pas, fragment par fragment."""

    def __init__(self, bundle: PolicyBundle, pdp: Optional[OdrlPolicyDecisionPoint] = None):
        self.bundle = bundle
        self.pdp = pdp or OdrlPolicyDecisionPoint(build_rule_activity_index(bundle))
        self.activity_fpa = build_activity_fpa_index(bundle)
        self._fpd_all = [
            p for fps in bundle.fp_results.values() for p in fps.fpd_policies
        ]
        self._adjacency = self._build_adjacency()

    def _build_adjacency(self) -> dict[str, list[dict[str, Any]]]:
        adj: dict[str, list[dict[str, Any]]] = {}
        for conn in self.bundle.fragments_enhanced.get("connections") or []:
            src = conn.get("from")
            if not src:
                continue
            adj.setdefault(src, []).append(conn)
        for flow in self.bundle.bp_model.get("flows") or []:
            src = flow.get("from")
            if not src:
                continue
            entry = {
                "from": flow.get("from"),
                "to": flow.get("to"),
                "type": flow.get("type", "sequence"),
                "condition": flow.get("condition"),
                "gateway": flow.get("gateway"),
            }
            if entry not in adj.get(src, []):
                adj.setdefault(src, []).append(entry)
        return adj

    def _start_activities(self) -> list[str]:
        starts = [
            a["name"]
            for a in self.bundle.bp_model.get("activities") or []
            if a.get("start")
        ]
        if starts:
            return starts
        order = self.bundle.global_activity_order
        return [order[0]] if order else []

    def _is_gateway(self, node: str) -> bool:
        gw_names = {g.get("name") for g in self.bundle.bp_model.get("gateways") or []}
        return node in gw_names

    def _fpd_for_gateway(self, gateway_name: str) -> list[dict]:
        out: list[dict] = []
        for pol in self._fpd_all:
            gw = pol.get("_gateway_name") or pol.get("_gateway") or ""
            if str(gw).lower() == str(gateway_name).lower():
                out.append(pol)
            elif pol.get("_gateway") == "XOR" and gateway_name in str(pol.get("_conditions", "")):
                out.append(pol)
        if out:
            return out
        slug = gateway_name.replace("-", "_")
        for pol in self._fpd_all:
            uid = str(pol.get("uid", "")).lower()
            if gateway_name.lower() in uid or slug in uid:
                out.append(pol)
        return out

    def _apply_gateway_fpd(self, gateway: str, ctx: ExecutionContext) -> None:
        subset = self._fpd_for_gateway(gateway)
        if subset:
            self.pdp.apply_fpd_policies(subset, ctx)
        else:
            self.pdp.apply_fpd_policies(self._fpd_all, ctx)

    def _apply_message_fpd(self, from_act: str, to_act: str, ctx: ExecutionContext) -> None:
        fa = from_act.replace("-", "_")
        ta = to_act.replace("-", "_")
        subset = [
            p
            for p in self._fpd_all
            if p.get("_flow") == "message"
            or ("MSG_" in str(p.get("uid", "")) and fa in str(p.get("uid", "")))
            or (fa in str(p.get("uid", "")) and ta in str(p.get("uid", "")))
        ]
        if subset:
            self.pdp.apply_fpd_policies(subset, ctx)

    def _resolve_gateway_targets(
        self, gateway: str, ctx: ExecutionContext
    ) -> list[str]:
        """Choisit les branches sortantes d'une passerelle selon les variables."""
        self._apply_gateway_fpd(gateway, ctx)
        outs: list[str] = []
        for edge in self._adjacency.get(gateway, []):
            dest = edge.get("to")
            if not dest or self._is_gateway(dest):
                continue
            cond = edge.get("condition")
            if cond is None:
                outs.append(dest)
                continue
            if str(ctx.get("product")) == str(cond):
                outs.append(dest)
            elif str(ctx.get(edge.get("gateway", ""))) == str(cond):
                outs.append(dest)
            elif str(ctx.variables.get(cond, "")) == "true":
                outs.append(dest)
        return outs

    def _next_nodes(self, current: str, ctx: ExecutionContext) -> list[str]:
        if self._is_gateway(current):
            return self._resolve_gateway_targets(current, ctx)

        next_nodes: list[str] = []
        for edge in self._adjacency.get(current, []):
            dest = edge.get("to")
            if not dest:
                continue
            if self._is_gateway(dest):
                next_nodes.extend(self._resolve_gateway_targets(dest, ctx))
            else:
                cond = edge.get("condition")
                if cond is None or str(ctx.get("product")) == str(cond):
                    next_nodes.append(dest)
        seen: set[str] = set()
        unique: list[str] = []
        for n in next_nodes:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        return unique

    def run(
        self,
        *,
        initial_variables: Optional[dict[str, Any]] = None,
        max_steps: int = 64,
    ) -> ExecutionResult:
        ctx = ExecutionContext(variables=dict(initial_variables or {}))
        ctx.set(
            "spatial",
            ctx.get("spatial") or "http://example.com/location/InternalNetwork",
        )
        steps: list[ExecutionStep] = []
        starts = self._start_activities()
        if not starts:
            return ExecutionResult(
                success=False,
                mode=RuntimeMode.SIMULATION,
                error="Aucune activité de départ trouvée",
            )

        for act in starts:
            for pol in self.activity_fpa.get(act, []):
                for rt in ("permission", "prohibition", "obligation"):
                    for rule in pol.get(rt) or []:
                        if isinstance(rule, dict) and rule.get("uid"):
                            rid = str(rule["uid"]).rstrip("/").split("/")[-1]
                            ctx.enabled_rules.add(rid)

        queue: list[str] = list(starts)
        visited_exec: set[str] = set()
        step_idx = 0

        while queue and step_idx < max_steps:
            activity = queue.pop(0)
            if activity in visited_exec and activity not in starts:
                continue
            if self._is_gateway(activity):
                queue.extend(self._next_nodes(activity, ctx))
                continue

            frag_id = self.bundle.fragment_for_activity(activity) or "?"
            ctx.active_fragment_id = frag_id
            fpa = self.activity_fpa.get(activity, [])
            is_start = activity in starts and activity not in ctx.completed_activities

            decision = self.pdp.can_execute_activity(
                activity,
                fpa,
                ctx,
                is_start_activity=is_start,
            )
            decision.fragment_id = frag_id

            if not decision.allowed:
                return ExecutionResult(
                    success=False,
                    mode=RuntimeMode.SIMULATION,
                    steps=steps,
                    final_context=ctx,
                    error=f"Activité bloquée : {activity} — {decision.reason}",
                    summary={"blocked_activity": activity, "reason": decision.reason},
                )

            ctx.completed_activities.append(activity)
            visited_exec.add(activity)
            ctx.set("policy_usage_count", ctx.get("policy_usage_count", 0) + 1)

            for edge in self._adjacency.get(activity, []):
                gw = edge.get("gateway")
                if gw and self._is_gateway(str(edge.get("to", ""))):
                    pass
                elif gw:
                    self._apply_gateway_fpd(str(gw), ctx)
                if edge.get("type") == "message":
                    self._apply_message_fpd(activity, str(edge.get("to", "")), ctx)

            steps.append(
                ExecutionStep(
                    step_index=step_idx,
                    activity=activity,
                    fragment_id=frag_id,
                    decision=decision,
                    enabled_rules_after=sorted(ctx.enabled_rules),
                    variables_snapshot=dict(ctx.variables),
                )
            )
            step_idx += 1

            for nxt in self._next_nodes(activity, ctx):
                if nxt not in queue:
                    queue.append(nxt)

            ends = {
                a["name"]
                for a in self.bundle.bp_model.get("activities") or []
                if a.get("end")
            }
            if activity in ends and not queue:
                break

        return ExecutionResult(
            success=True,
            mode=RuntimeMode.SIMULATION,
            steps=steps,
            final_context=ctx,
            summary={
                "completed_activities": list(ctx.completed_activities),
                "enabled_rules": sorted(ctx.enabled_rules),
                "step_count": len(steps),
            },
        )
