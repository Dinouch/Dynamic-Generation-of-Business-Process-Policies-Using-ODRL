"""
Moteur d'exécution principal : simulation locale ou runtime Camunda + PEP ODRL.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from agents.Agent_4.odrl_deterministic_templates import FragmentPolicySet

from .camunda_client import CamundaClientError, CamundaRestClient
from .fragment_orchestrator import FragmentOrchestrator
from .models import ExecutionContext, ExecutionResult, RuntimeMode
from .odrl_pdp import OdrlPolicyDecisionPoint
from .policy_loader import (
    PolicyBundle,
    build_activity_fpa_index,
    build_rule_activity_index,
    load_policy_bundle,
    load_policies_from_export_dir,
)

logger = logging.getLogger(__name__)


def _slug_to_task_name(activity_slug: str) -> str:
    """Heuristique : task Camunda souvent en camelCase depuis le slug BPMN."""
    parts = activity_slug.split("-")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _task_matches_activity(task: dict, activity_slug: str) -> bool:
    name = (task.get("name") or "").lower().replace(" ", "-")
    tid = (task.get("taskDefinitionKey") or task.get("activityId") or "").lower()
    slug = activity_slug.lower()
    camel = _slug_to_task_name(activity_slug).lower()
    return slug in name or slug in tid or camel in name or camel in tid


class ExecutionEngine:
    """
    Point d'entrée pour exécuter un scénario après génération des politiques.

    Exemple (simulation)::

        engine = ExecutionEngine.from_scenario("src/dataset/scenario018")
        result = engine.run_simulation(initial_variables={"product": "complete"})
    """

    def __init__(self, bundle: PolicyBundle, mode: RuntimeMode = RuntimeMode.SIMULATION):
        self.bundle = bundle
        self.mode = mode
        self.pdp = OdrlPolicyDecisionPoint(build_rule_activity_index(bundle))
        self.activity_fpa = build_activity_fpa_index(bundle)

    @classmethod
    def from_scenario(
        cls,
        scenario_dir: str | Path,
        *,
        policies_dir: Optional[str | Path] = None,
        mode: RuntimeMode = RuntimeMode.SIMULATION,
    ) -> "ExecutionEngine":
        bundle = load_policy_bundle(scenario_dir, policies_dir=policies_dir)
        if not bundle.fp_results:
            logger.warning(
                "Aucune politique ODRL chargée pour %s — l'exécution ODRL sera limitée.",
                scenario_dir,
            )
        return cls(bundle=bundle, mode=mode)

    @classmethod
    def from_export(
        cls,
        scenario_dir: str | Path,
        policies_export_dir: str | Path,
        *,
        mode: RuntimeMode = RuntimeMode.SIMULATION,
    ) -> "ExecutionEngine":
        bundle = load_policy_bundle(scenario_dir, policies_dir=policies_export_dir)
        return cls(bundle=bundle, mode=mode)

    @classmethod
    def from_fp_results(
        cls,
        scenario_dir: str | Path,
        fp_results: dict[str, FragmentPolicySet],
        *,
        mode: RuntimeMode = RuntimeMode.SIMULATION,
    ) -> "ExecutionEngine":
        bundle = load_policy_bundle(scenario_dir, policies_dir=None)
        bundle.fp_results = fp_results
        bundle.policies_flat = [
            p for fps in fp_results.values() for p in fps.all_policies()
        ]
        return cls(bundle=bundle, mode=mode)

    def run_simulation(
        self,
        *,
        initial_variables: Optional[dict[str, Any]] = None,
        max_steps: int = 64,
    ) -> ExecutionResult:
        orchestrator = FragmentOrchestrator(self.bundle, self.pdp)
        return orchestrator.run(
            initial_variables=initial_variables,
            max_steps=max_steps,
        )

    def run_camunda(
        self,
        *,
        bpmn_path: str | Path,
        process_definition_key: str,
        initial_variables: Optional[dict[str, Any]] = None,
        deployment_name: str = "bpfragment-odrl",
        max_task_rounds: int = 32,
        client: Optional[CamundaRestClient] = None,
    ) -> ExecutionResult:
        """
        Déploie le BPMN, démarre une instance, complète les tâches utilisateur
        uniquement si le PEP ODRL autorise l'activité correspondante.
        """
        camunda = client or CamundaRestClient()
        if not camunda.health():
            return ExecutionResult(
                success=False,
                mode=RuntimeMode.CAMUNDA_7,
                error=(
                    f"Camunda injoignable à {camunda.base_url}. "
                    "Déployez sur AKS ou lancez un conteneur local (voir deploy/camunda-aks/README.md)."
                ),
            )

        try:
            camunda.deploy_bpmn(bpmn_path, deployment_name)
            instance = camunda.start_process(
                process_definition_key,
                variables=initial_variables,
            )
        except CamundaClientError as e:
            return ExecutionResult(
                success=False,
                mode=RuntimeMode.CAMUNDA_7,
                error=str(e),
            )

        instance_id = instance.get("id")
        ctx = ExecutionContext(variables=dict(initial_variables or {}))
        fpd_all = [
            p for fps in self.bundle.fp_results.values() for p in fps.fpd_policies
        ]
        self.pdp.apply_fpd_policies(fpd_all, ctx)

        from .fragment_orchestrator import FragmentOrchestrator

        sim_helper = FragmentOrchestrator(self.bundle, self.pdp)
        starts = sim_helper._start_activities()

        steps = []
        for round_i in range(max_task_rounds):
            tasks = camunda.list_tasks(process_instance_id=instance_id)
            if not tasks:
                break
            progressed = False
            for task in tasks:
                activity = self._infer_activity_from_task(task, starts)
                if not activity:
                    logger.warning("Tâche Camunda non mappée : %s", task)
                    continue
                fpa = self.activity_fpa.get(activity, [])
                is_start = activity in starts and activity not in ctx.completed_activities
                decision = self.pdp.can_execute_activity(
                    activity, fpa, ctx, is_start_activity=is_start
                )
                if not decision.allowed:
                    return ExecutionResult(
                        success=False,
                        mode=RuntimeMode.CAMUNDA_7,
                        steps=steps,
                        final_context=ctx,
                        camunda_process_instance_id=instance_id,
                        error=f"PEP ODRL : tâche « {activity} » refusée — {decision.reason}",
                        summary={"blocked_task": task, "activity": activity},
                    )
                try:
                    camunda.complete_task(task["id"], variables=ctx.variables)
                except CamundaClientError as e:
                    return ExecutionResult(
                        success=False,
                        mode=RuntimeMode.CAMUNDA_7,
                        error=str(e),
                        camunda_process_instance_id=instance_id,
                    )
                ctx.completed_activities.append(activity)
                ctx.set("policy_usage_count", ctx.get("policy_usage_count", 0) + 1)
                self.pdp.apply_fpd_policies(fpd_all, ctx)
                from .models import ExecutionStep

                steps.append(
                    ExecutionStep(
                        step_index=round_i,
                        activity=activity,
                        fragment_id=self.bundle.fragment_for_activity(activity) or "?",
                        decision=decision,
                        enabled_rules_after=sorted(ctx.enabled_rules),
                        variables_snapshot=dict(ctx.variables),
                    )
                )
                progressed = True
            if not progressed:
                break

        remaining = camunda.list_tasks(process_instance_id=instance_id)
        success = len(remaining) == 0
        return ExecutionResult(
            success=success,
            mode=RuntimeMode.CAMUNDA_7,
            steps=steps,
            final_context=ctx,
            camunda_process_instance_id=instance_id,
            error=None if success else f"Tâches restantes : {len(remaining)}",
            summary={
                "completed_activities": list(ctx.completed_activities),
                "remaining_tasks": len(remaining),
            },
        )

    def _infer_activity_from_task(
        self, task: dict, start_activities: list[str]
    ) -> Optional[str]:
        for act in self.bundle.global_activity_order:
            if _task_matches_activity(task, act):
                return act
        name = (task.get("name") or "").lower()
        for act in start_activities:
            if act.replace("-", " ") in name or act in name:
                return act
        m = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
        if m in self.bundle.global_activity_order:
            return m
        return None

    def run(
        self,
        *,
        initial_variables: Optional[dict[str, Any]] = None,
        bpmn_path: Optional[str | Path] = None,
        process_definition_key: Optional[str] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        if self.mode == RuntimeMode.SIMULATION:
            return self.run_simulation(initial_variables=initial_variables, **kwargs)
        if self.mode in (RuntimeMode.CAMUNDA_7, RuntimeMode.CAMUNDA_8):
            if not bpmn_path or not process_definition_key:
                return ExecutionResult(
                    success=False,
                    mode=self.mode,
                    error="mode Camunda requiert bpmn_path et process_definition_key",
                )
            return self.run_camunda(
                bpmn_path=bpmn_path,
                process_definition_key=process_definition_key,
                initial_variables=initial_variables,
                **kwargs,
            )
        return ExecutionResult(
            success=False,
            mode=self.mode,
            error=f"Mode non supporté : {self.mode}",
        )
