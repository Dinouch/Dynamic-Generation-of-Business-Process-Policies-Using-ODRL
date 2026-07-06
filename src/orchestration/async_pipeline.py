from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

from agents.pipeline_registry import COVERED_PATTERNS
from agents.structural_analyzer import StructuralAnalyzer, AgentMessage, MessageType
from agents.exception_handling_agent import UnmappedCaseFormulator
from agents.Agent_3.constraint_validator import ConstraintValidator, ValidationReport
from agents.policy_projection_agent import PolicyProjectionAgent
from agents.policy_auditor import PolicyAuditor, AuditReport
from agents.human_agent import HumanAgent, HumanDecisionBridge

from communication.acl import ACLEnvelope, ACLPerformative, FIPA_EXPECTS_AGREE_KEY
from communication.bus import AsyncBus, PublishHook


def _coerce_fp_results(raw: Any) -> dict[str, Any]:
    """Normalize ``fp_results`` (FragmentPolicySet or serialized dict) for ``export()``."""
    from agents.Agent_4.odrl_deterministic_templates import FragmentPolicySet

    if not isinstance(raw, dict) or not raw:
        return {}
    out: dict[str, Any] = {}
    for fid, fps in raw.items():
        frag_id = str(fid)
        if isinstance(fps, FragmentPolicySet):
            out[frag_id] = fps
            continue
        if isinstance(fps, dict) and callable(getattr(fps, "all_policies", None)):
            out[frag_id] = fps
            continue
        if isinstance(fps, dict):
            out[frag_id] = FragmentPolicySet(
                fragment_id=frag_id,
                fpa_policies=[
                    p for p in (fps.get("fpa_policies") or []) if isinstance(p, dict)
                ],
                fpd_policies=[
                    p for p in (fps.get("fpd_policies") or []) if isinstance(p, dict)
                ],
            )
    return out


def _export_fp_results_to_disk(
    scenario_id: str,
    stage: str,
    fp_results: Any,
    *,
    enriched_graph: Any = None,
) -> Optional[str]:
    """Write ``output/{scenario}/odrl_fragment_policies/{stage}/``; return the path or ``None``."""
    coerced = _coerce_fp_results(fp_results)
    if not coerced:
        return None
    if not any(fps.all_policies() for fps in coerced.values()):
        return None
    out_dir = _scenario_odrl_stage_dir(scenario_id, stage)
    try:
        if out_dir.is_dir():
            shutil.rmtree(out_dir, ignore_errors=True)
        exporter = PolicyProjectionAgent(
            enriched_graph=enriched_graph,
            validation_report=None,
        )
        exporter.export(coerced, output_dir=str(out_dir), replace_existing=True)
        print(f"[AsyncPipeline] Export ODRL ({stage}) -> {out_dir}")
        return str(out_dir)
    except Exception as e:
        print(f"[AsyncPipeline][WARN] Export {stage} failed: {e}")
        return None


def _pick_export_stage(status: str, *, is_valid: bool) -> str:
    st = (status or "").strip()
    if st in ("partial_template_only", "final_validated", "final_merged", "generated"):
        if st == "final_merged":
            return "final_validated"
        return st
    return "final_validated" if is_valid else "partial_template_only"


def _infer_process_display_name(
    bp_model: dict,
    scenario_id: str,
    override: Optional[str],
) -> str:
    if override and str(override).strip():
        return str(override).strip()
    for k in ("name", "title", "processName", "process_name"):
        v = bp_model.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    sid = (scenario_id or "scenario").replace("_", " ").strip()
    return f"Process scenario — {sid.title()}"


@dataclass
class PipelineAsyncResult:
    is_valid: bool
    report: AuditReport
    summary: dict
    syntax_score: float
    msg_type: str
    status: str  # partial_template_only | final_merged | final_validated
    manifest: Optional[dict] = None
    scenario_id: str = ""
    odrl_export_paths: Optional[dict[str, str]] = None


class AsyncPipelineOrchestrator:
    """
    Phase 1-5 implementation:
    - asyncio bus + async orchestration
    - HITL gate at Agent 3 for unmapped proposals (batch)
    - partial template-only continuation on timeout
    - merge additively with unmapped-generated FPd once human validates

    Notes
    -----
    This keeps existing agent logic mostly intact and orchestrates the new behavior
    at the pipeline level.
    """

    def __init__(
        self,
        *,
        bp_model: dict,
        fragments: list[dict],
        b2p_policies: list[dict],
        human_timeout_s: float = 120.0,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        human_agent: Optional[HumanAgent] = None,
        acl_trace: Optional[PublishHook] = None,
        scenario_id: str = "scenario1",
        process_title: Optional[str] = None,
        accept_all_unmapped: bool = False,
        enable_hitl: bool = True,
        max_wall_s: Optional[float] = None,
        increment_profile: str = "I5",
    ):
        self._increment_profile = (increment_profile or "I5").strip().upper()
        self.bp_model = bp_model
        self.fragments = fragments
        self.b2p_policies = b2p_policies
        self.human_timeout_s = human_timeout_s
        self._scenario_id = (scenario_id or "scenario1").strip()
        self._process_display_name = _infer_process_display_name(
            bp_model, self._scenario_id, process_title
        )

        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self.azure_deployment = azure_deployment

        self.bus = AsyncBus(on_publish=acl_trace)
        self._accept_all_unmapped = bool(accept_all_unmapped)
        self._enable_hitl = bool(enable_hitl)
        self.human = human_agent if self._enable_hitl else None

        raw_max = os.environ.get("PIPELINE_ASYNC_MAX_WALL_S", "").strip()
        if max_wall_s is not None:
            self._max_wall_s = float(max_wall_s)
        elif raw_max:
            try:
                self._max_wall_s = float(raw_max)
            except ValueError:
                self._max_wall_s = 3600.0
        else:
            self._max_wall_s = 3600.0

        self._conversation_id = os.urandom(8).hex()
        self._final_result: Optional[PipelineAsyncResult] = None
        self._partial_result: Optional[PipelineAsyncResult] = None
        self._partial_fp_results: Optional[dict[str, Any]] = None
        self._last_exportable_fp_results: Optional[dict[str, Any]] = None
        self._enriched_graph: Optional[Any] = None
        self._pending_unmapped_env: Optional[ACLEnvelope] = None
        self._unmapped_queue: list[dict] = []
        self._unmapped_accepted: list[dict] = []
        self._unmapped_pending_by_request: dict[str, dict] = {}
        self._auto_close_after_partial = os.environ.get("HITL_AUTOCLOSE_AFTER_PARTIAL", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self._odrl_export_paths: dict[str, str] = {}

    async def run(self) -> PipelineAsyncResult:
        loop = asyncio.get_running_loop()
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        azure_endpoint = self.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_version = self.azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
        azure_deploy = self.azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")

        agent1 = StructuralAnalyzer(
            self.bp_model,
            self.fragments,
            self.b2p_policies,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_version,
            azure_deployment=azure_deploy,
        )
        exception_handling_agent = UnmappedCaseFormulator(
            covered_patterns=COVERED_PATTERNS,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_version,
            azure_deployment=azure_deploy,
        )
        agent3 = ConstraintValidator(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_version,
            azure_deployment=azure_deploy,
        )
        agent4 = PolicyProjectionAgent(
            enriched_graph=None,
            validation_report=None,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_version,
            azure_deployment=azure_deploy,
        )
        agent5 = PolicyAuditor(
            fp_results={},
            enriched_graph=None,
            raw_b2p=self.b2p_policies,
        )

        # Register bus handlers (receiver = agent name)
        self.bus.register("agent1", lambda env: self._deliver_to_legacy(agent1, env))
        self.bus.register("exception_handling_agent", lambda env: self._deliver_to_legacy(exception_handling_agent, env))
        self.bus.register("agent4", lambda env: self._deliver_to_legacy(agent4, env))
        self.bus.register("agent5", lambda env: self._deliver_to_legacy(agent5, env))

        # Agent 3 and human are orchestrated explicitly for HITL
        self.bus.register("agent3", lambda env: self._handle_agent3(env, agent3, agent4, agent5))
        if self._enable_hitl:
            self.bus.register("human3", self._handle_human3)
        self.bus.register("pipeline", self._handle_pipeline)

        await self.bus.start()

        try:
            def _threadsafe_publish(m: AgentMessage, *, status: Optional[str] = None) -> None:
                async def _safe_publish() -> None:
                    try:
                        await self._publish_legacy(m, status=status)
                    except Exception:
                        import traceback as _tb

                        mt = (
                            m.msg_type.value
                            if hasattr(m.msg_type, "value")
                            else str(m.msg_type)
                        )
                        print(
                            f"[AsyncPipeline][ERROR] Async publish failed {m.sender}→{m.recipient} "
                            f"({mt}):"
                        )
                        _tb.print_exc()

                loop.call_soon_threadsafe(asyncio.create_task, _safe_publish())

            agent1.register_send_callback(lambda m: _threadsafe_publish(m, status=None))
            exception_handling_agent.register_send_callback(lambda m: _threadsafe_publish(m, status=None))
            agent3.register_send_callback_exception_handling(lambda m: _threadsafe_publish(m, status=None))

            # Route Agent 3 -> Agent 4/1/2 through bus
            agent3.register_send_callback_agent4(lambda m: _threadsafe_publish(m, status=None))
            agent4.register_send_callback(lambda m: _threadsafe_publish(m, status=None))
            agent5.register_send_callback(lambda m: _threadsafe_publish(m, status=None))

            kick = ACLEnvelope(
                performative=ACLPerformative.REQUEST,
                sender="pipeline",
                receiver="agent1",
                ontology="graph-structural",
                content={
                    "utterance": (
                        "Analyze this business process graph and produce an enriched structural model."
                    ),
                    "intent": "analyze-enriched-graph",
                    FIPA_EXPECTS_AGREE_KEY: True,
                },
                conversation_id=self._conversation_id,
            )
            await self.bus.publish(kick)

            # Wait for final result (infinite-loop safeguard)
            deadline = time.monotonic() + self._max_wall_s
            while self._final_result is None:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Async pipeline interrupted after {self._max_wall_s:.0f}s with no final result "
                        f"(I4/I5 profile — check Agent 4/5 logs)."
                    )
                await asyncio.sleep(0.05)

            return self._finalize_disk_exports(self._final_result)
        finally:
            await self.bus.stop()

    def _remember_fp_results(self, fp_results: Any) -> None:
        if fp_results:
            self._last_exportable_fp_results = fp_results

    def _finalize_disk_exports(self, result: PipelineAsyncResult) -> PipelineAsyncResult:
        """Last resort: ensure a disk export if the pipeline produced policies."""
        if self._odrl_export_paths:
            result.odrl_export_paths = dict(self._odrl_export_paths)
            return result
        fp = self._last_exportable_fp_results or self._partial_fp_results
        if fp:
            stage = _pick_export_stage(result.status, is_valid=bool(result.is_valid))
            path = _export_fp_results_to_disk(
                self._scenario_id,
                stage,
                fp,
                enriched_graph=self._enriched_graph,
            )
            if path:
                self._odrl_export_paths[stage] = path
        if self._odrl_export_paths:
            result.odrl_export_paths = dict(self._odrl_export_paths)
        elif fp:
            print(
                f"[AsyncPipeline][WARN] Pipeline finished with no disk export for "
                f"{self._scenario_id!r} — check the logs above."
            )
        return result

    def _try_export_terminal(
        self,
        *,
        status: str,
        is_valid: bool,
        fp_results: Any,
    ) -> None:
        if not fp_results:
            fp_results = self._last_exportable_fp_results or self._partial_fp_results
        if not fp_results:
            return
        self._remember_fp_results(fp_results)
        stage = _pick_export_stage(status, is_valid=is_valid)
        if stage in self._odrl_export_paths:
            return
        path = _export_fp_results_to_disk(
            self._scenario_id,
            stage,
            fp_results,
            enriched_graph=self._enriched_graph,
        )
        if path:
            self._odrl_export_paths[stage] = path

    async def _publish_legacy(self, msg: AgentMessage, *, status: Optional[str]) -> None:
        from communication.legacy_adapter import agent_message_to_acl

        env = agent_message_to_acl(
            msg,
            conversation_id=self._conversation_id,
            in_reply_to=None,
            status=status,
        )
        # Ensure conversation_id is set (adapter default may be "")
        if not env.conversation_id:
            env = ACLEnvelope(
                performative=env.performative,
                sender=env.sender,
                receiver=env.receiver,
                ontology=env.ontology,
                content=env.content,
                language=env.language,
                conversation_id=self._conversation_id,
                reply_with=env.reply_with,
                in_reply_to=env.in_reply_to,
                timestamp=env.timestamp,
            )
        try:
            await self.bus.publish(env)
        except Exception as exc:
            mt = (msg.msg_type.value if hasattr(msg.msg_type, "value") else str(msg.msg_type))
            print(
                f"[AsyncPipeline][ERROR] ACL publication rejected for {msg.sender}→{msg.recipient} "
                f"({mt}): {exc}"
            )
            raise

    async def _deliver_to_legacy(self, agent: Any, env: ACLEnvelope) -> None:
        from communication.legacy_adapter import acl_to_agent_message

        msg = acl_to_agent_message(env)
        # Capture partial template-only bundle when Agent 4 forwards to Agent 5.
        if (
            env.sender == "agent4"
            and env.receiver == "agent5"
            and (env.content or {}).get("status") == "partial_template_only"
            and "fp_results" in (env.content or {})
        ):
            self._partial_fp_results = env.content.get("fp_results")
            self._remember_fp_results(self._partial_fp_results)
            try:
                path = _export_fp_results_to_disk(
                    self._scenario_id,
                    "partial_template_only",
                    self._partial_fp_results,
                    enriched_graph=self._enriched_graph,
                )
                if path:
                    self._odrl_export_paths["partial_template_only"] = path
            except Exception as e:
                print(f"[AsyncPipeline][WARN] Partial export failed: {e}")
        await asyncio.to_thread(agent.receive, msg)

    async def _handle_agent3(
        self,
        env: ACLEnvelope,
        agent3: ConstraintValidator,
        agent4: PolicyProjectionAgent,
        agent5: PolicyAuditor,
    ) -> None:
        """
        Route unmapped proposals.

        - ``accept_all_unmapped`` (I5 batch): orchestrator acceptance, merge, projection.
        - ``enable_hitl`` (interactive I5): human queue + partial export on timeout.
        - Otherwise (**I4**): direct delivery to Agent 3 (semantic validation, no HITL).
        """
        from communication.legacy_adapter import acl_to_agent_message

        # Track enriched graph for later merges/exports
        if env.ontology in {"graph-structural", "unmapped-formulation", "validation", "policy-projection"}:
            if isinstance(env.content, dict) and "enriched_graph" in env.content:
                self._enriched_graph = env.content.get("enriched_graph")

        msg = acl_to_agent_message(env)

        # Human replies are routed to agent3: resume the HITL gate.
        if env.sender == "human3" and env.in_reply_to:
            await self.on_human_reply_to_agent3(env, agent4)
            return

        if msg.msg_type == MessageType.UNMAPPED_PROPOSALS:
            proposals = list(env.content.get("unmapped_proposals") or [])

            if not proposals:
                await asyncio.to_thread(agent3.receive, msg)
                return

            # I5 batch (auto-agree): all proposals accepted without HITL gate.
            if self._accept_all_unmapped:
                self._pending_unmapped_env = env
                self._unmapped_queue = []
                self._unmapped_accepted = []
                self._unmapped_pending_by_request = {}
                print(
                    f"[AsyncPipeline][{self._increment_profile}] {len(proposals)} unmapped proposal(s) "
                    f"accepted by default (no HITL)."
                )
                for idx, prop in enumerate(proposals, start=1):
                    self._log_unmapped_proposal(prop, idx, len(proposals))
                self._unmapped_accepted = list(proposals)
                await self._commit_accepted_unmapped_batch(agent4)
                return

            # Interactive I5: human validation one by one.
            if self._enable_hitl:
                self._pending_unmapped_env = env
                self._unmapped_queue = []
                self._unmapped_accepted = []
                self._unmapped_pending_by_request = {}
                self._unmapped_queue = list(proposals)
                await self._ask_next_unmapped()
                print(f"[AsyncPipeline] HITL gate opened — timeout={self.human_timeout_s}s")
                asyncio.create_task(self._timeout_partial_templates(agent4, agent5, base_env=env))
                return

            # I4: Agent 3 semantic validation (no HITL, no orchestrator accept-all).
            print(
                f"[AsyncPipeline][{self._increment_profile}] {len(proposals)} unmapped proposal(s) "
                "→ Agent 3 validation (semantic)."
            )
            await asyncio.to_thread(agent3.receive, msg)
            return

        if msg.msg_type == MessageType.REFORMULATED_PROPOSALS:
            # FIPA inform (not contract-net propose): skip HITL, deliver straight to Agent 3.
            await asyncio.to_thread(agent3.receive, msg)
            return

        # Default: deliver to legacy Agent 3 (semantic loop etc.)
        await asyncio.to_thread(agent3.receive, msg)

    async def _emit_templates_only_validation(
        self,
        enriched: Any,
        *,
        loop_turn: int = 0,
        status: str = "partial_template_only",
        log_prefix: str = "Generating template-only bundle",
    ) -> None:
        """Push an empty ``VALIDATION_DONE`` to agent 4 (no unmapped accepted)."""
        report = ValidationReport(results=[])
        msg = AgentMessage(
            sender="agent3",
            recipient="agent4",
            msg_type=MessageType.VALIDATION_DONE,
            payload={"validation_report": report, "enriched_graph": enriched},
            loop_turn=int(loop_turn or 0),
        )
        print(f"[AsyncPipeline] {log_prefix}.")
        await self._publish_legacy(msg, status=status)

    async def _timeout_partial_templates(
        self,
        agent4: PolicyProjectionAgent,
        agent5: PolicyAuditor,
        *,
        base_env: ACLEnvelope,
    ) -> None:
        """
        After human_timeout_s, if still waiting, proceed with templates-only generation:
        Agent 4 generates with empty ValidationReport (no accepted unmapped) and status partial_template_only,
        then Agent 5 audits that partial bundle.
        """
        try:
            await asyncio.sleep(self.human_timeout_s)
            if self._pending_unmapped_env is None:
                return

            enriched = base_env.content.get("enriched_graph")
            if enriched is None:
                print("[AsyncPipeline][WARN] Timeout partial: enriched_graph missing.")
                return

            await self._emit_templates_only_validation(
                enriched,
                loop_turn=int(base_env.content.get("loop_turn", 0) or 0),
                status="partial_template_only",
                log_prefix="HITL timeout reached — generating template-only partial bundle",
            )
        except Exception as e:
            print(f"[AsyncPipeline][ERROR] Timeout partial generation failed: {e}")

        # Try exporting the partial bundle once Agent 5 has received it later;
        # if we already captured it, export now.
        try:
            if self._partial_fp_results is not None:
                path = _export_fp_results_to_disk(
                    self._scenario_id,
                    "partial_template_only",
                    self._partial_fp_results,
                    enriched_graph=self._enriched_graph,
                )
                if path:
                    self._odrl_export_paths["partial_template_only"] = path
        except Exception as e:
            print(f"[AsyncPipeline][WARN] Partial export failed: {e}")

        # Note: partial fp_results is captured when we intercept POLICIES_READY to agent5.
        _ = agent4
        _ = agent5

    @staticmethod
    def _log_unmapped_proposal(proposal: dict[str, Any], index: int, total: int) -> None:
        from agents.human_agent import HumanAgent

        print(f"\n── Unmapped proposal {index}/{total} ──")
        for ln in HumanAgent._format_unmapped_proposal(proposal):
            print(ln)

    async def _handle_human3(self, env: ACLEnvelope) -> None:
        if self.human is None:
            return
        # HumanAgent returns a response envelope or None on timeout
        resp = await self.human.handle(env)
        if resp is None:
            return
        await self.bus.publish(resp)

    async def _handle_pipeline(self, env: ACLEnvelope) -> None:
        """
        Orchestrator inbox: structural delegation (Agent 1), relay to Agent 3,
        and terminal ODRL syntax audit outcome (forwarded by Agent 4).
        """
        # Agent 1 → pipeline: AGREE (delegation) or INFORM (GRAPH_READY relay to Agent 3)
        if env.sender == "agent1" and env.receiver == "pipeline":
            if env.performative == ACLPerformative.AGREE:
                print("[AsyncPipeline] Agent 1 agreed to run structural analysis.")
                return
            if env.performative == ACLPerformative.INFORM:
                c = env.content or {}
                if c.get("msg_type") != MessageType.GRAPH_READY.value:
                    return
                loop_turn = int(c.get("loop_turn", 0) or 0)
                payload = {k: v for k, v in c.items() if k not in ("msg_type", "loop_turn", "status")}
                relay = AgentMessage(
                    sender="pipeline",
                    recipient="agent3",
                    msg_type=MessageType.GRAPH_READY,
                    payload=payload,
                    loop_turn=loop_turn,
                )
                await self._publish_legacy(relay, status=None)
                if self._accept_all_unmapped:
                    print(
                        f"[AsyncPipeline][{self._increment_profile}] Enriched graph relayed to Agent 3 — "
                        "waiting for unmapped proposals (exception handling agent, LLM calls)…"
                    )
            return

        # Final audit outcome: Agent 4 forwards Agent 5's terminal INFORM to the orchestrator
        if env.sender == "agent4" and env.receiver == "pipeline":
            mt = str((env.content or {}).get("msg_type", ""))
            if mt not in (MessageType.ODRL_VALID.value, MessageType.ODRL_SYNTAX_ERROR.value):
                return
            inferred = "final_validated"
            if self._pending_unmapped_env is not None and self._partial_result is None:
                inferred = "partial_template_only"
            status = str((env.content or {}).get("status") or inferred)
            payload = env.content or {}
            report = payload.get("report")
            summary = payload.get("summary") or {}
            syntax_score = float(payload.get("syntax_score", 1.0) or 1.0)
            msg_type = str(payload.get("msg_type") or payload.get("signal") or "")
            is_valid = bool(payload.get("is_valid", False))

            out = PipelineAsyncResult(
                is_valid=is_valid,
                report=report,
                summary=summary,
                syntax_score=syntax_score,
                msg_type=msg_type,
                status=status,
                manifest=payload.get("manifest"),
                scenario_id=self._scenario_id,
                odrl_export_paths=dict(self._odrl_export_paths),
            )
            if status == "partial_template_only" and not self._auto_close_after_partial:
                self._partial_result = out
                print(
                    "[AsyncPipeline] Partial bundle ready (template-only). "
                    "Waiting for human decision to merge unmapped cases..."
                )
                return

            if mt == MessageType.ODRL_VALID.value and is_valid:
                fp_results = payload.get("fp_results")
                if fp_results:
                    self._remember_fp_results(fp_results)
                else:
                    print(
                        "[AsyncPipeline][WARN] ODRL_VALID without fp_results — "
                        "attempting export from orchestrator cache."
                    )
                self._try_export_terminal(
                    status=status,
                    is_valid=is_valid,
                    fp_results=fp_results,
                )
            elif payload.get("fp_results"):
                # Best-effort export even if final syntax validation failed (UI visualization).
                self._try_export_terminal(
                    status=status,
                    is_valid=is_valid,
                    fp_results=payload.get("fp_results"),
                )

            print(
                f"[AsyncPipeline] Pipeline finished (Agent 4 → pipeline): msg_type={msg_type!r}, "
                f"is_valid={is_valid}, status={status!r} — resuming orchestrator."
            )
            self._final_result = out
            return

        # Legacy: direct Agent 5 → pipeline (kept for compatibility)
        if env.sender == "agent5" and env.receiver == "pipeline":
            # If we're still waiting for the human gate, the first Agent 5 result is a partial bundle.
            inferred = "final_validated"
            if self._pending_unmapped_env is not None and self._partial_result is None:
                inferred = "partial_template_only"
            status = str(env.content.get("status") or inferred)
            payload = env.content
            report = payload.get("report")
            summary = payload.get("summary") or {}
            syntax_score = float(payload.get("syntax_score", 1.0) or 1.0)
            msg_type = str(payload.get("msg_type") or payload.get("signal") or "")
            is_valid = bool(payload.get("is_valid", False))

            out = PipelineAsyncResult(
                is_valid=is_valid,
                report=report,
                summary=summary,
                syntax_score=syntax_score,
                msg_type=msg_type,
                status=status,
                manifest=payload.get("manifest"),
                scenario_id=self._scenario_id,
                odrl_export_paths=dict(self._odrl_export_paths),
            )
            if status == "partial_template_only" and not self._auto_close_after_partial:
                self._partial_result = out
                print(
                    "[AsyncPipeline] Partial bundle ready (template-only). "
                    "Waiting for human decision to merge unmapped cases..."
                )
                return

            print(
                f"[AsyncPipeline] Pipeline finished (Agent 5 direct): msg_type={msg_type!r}, "
                f"is_valid={is_valid}, status={status!r} — resuming orchestrator."
            )
            self._try_export_terminal(
                status=status,
                is_valid=is_valid,
                fp_results=payload.get("fp_results"),
            )
            self._final_result = out

    async def on_human_reply_to_agent3(
        self, env: ACLEnvelope, agent4: PolicyProjectionAgent
    ) -> None:
        """
        Called when human replies AGREE/REFUSE to one unmapped proposal.
        """
        if self._pending_unmapped_env is None:
            return

        req_id = env.in_reply_to or ""
        proposal = self._unmapped_pending_by_request.pop(req_id, None)
        if proposal is None:
            print(
                "[AsyncPipeline][WARN] Human reply ignored: no pending proposal for "
                f"in_reply_to={req_id!r}. Known keys: {list(self._unmapped_pending_by_request.keys())!r}"
            )
            return

        decision = env.performative
        if decision == ACLPerformative.AGREE:
            self._unmapped_accepted.append(proposal)

        enriched = self._pending_unmapped_env.content.get("enriched_graph")
        if enriched is None:
            self._pending_unmapped_env = None
            self._unmapped_queue = []
            self._unmapped_accepted = []
            self._unmapped_pending_by_request = {}
            return

        # Continue prompting until all proposals are processed.
        if self._unmapped_queue:
            await self._ask_next_unmapped()
            return

        # Human finished validating all proposals.
        if not self._unmapped_accepted:
            propose_rw = self._pending_unmapped_env.reply_with
            loop_t = int(self._pending_unmapped_env.content.get("loop_turn", 0) or 0)
            rej = AgentMessage(
                sender="agent3",
                recipient="exception_handling_agent",
                msg_type=MessageType.REJECT_PROPOSAL_BATCH,
                payload={
                    "utterance": "Human operator rejected all proposed unmapped rules.",
                    "acl_in_reply_to": propose_rw,
                    "action": "human-gate-reject-all",
                    "reason": "No unmapped proposals were accepted in human review.",
                },
                loop_turn=loop_t,
            )
            await self._publish_legacy(rej, status=None)
            self._pending_unmapped_env = None
            self._unmapped_queue = []
            self._unmapped_accepted = []
            self._unmapped_pending_by_request = {}
            if self._partial_result is not None and self._final_result is None:
                self._try_export_terminal(
                    status=str(self._partial_result.status or "partial_template_only"),
                    is_valid=bool(self._partial_result.is_valid),
                    fp_results=self._partial_fp_results,
                )
                self._final_result = self._partial_result
                return
            if enriched is not None and self._final_result is None:
                await self._emit_templates_only_validation(
                    enriched,
                    loop_turn=loop_t,
                    status="partial_template_only",
                    log_prefix=(
                        "HITL: no proposal accepted — exporting deterministic "
                        "policies (templates) only"
                    ),
                )
            return

        await self._commit_accepted_unmapped_batch(agent4)

    async def _commit_accepted_unmapped_batch(self, agent4: PolicyProjectionAgent) -> None:
        """Merge accepted unmapped cases and run the final Agent 4 → 5 pass."""
        if self._pending_unmapped_env is None or not self._unmapped_accepted:
            return

        enriched = self._pending_unmapped_env.content.get("enriched_graph")
        if enriched is None:
            self._pending_unmapped_env = None
            self._unmapped_queue = []
            self._unmapped_accepted = []
            self._unmapped_pending_by_request = {}
            return

        propose_rw = self._pending_unmapped_env.reply_with
        loop_t = int(self._pending_unmapped_env.content.get("loop_turn", 0) or 0)
        acc = AgentMessage(
            sender="agent3",
            recipient="exception_handling_agent",
            msg_type=MessageType.ACCEPT_PROPOSAL_BATCH,
            payload={
                "utterance": (
                    "Unmapped proposals accepted; "
                    "proceeding to merge and fragment-policy projection."
                ),
                "acl_in_reply_to": propose_rw,
            },
            loop_turn=loop_t,
        )
        await self._publish_legacy(acc, status=None)

        n_acc = len(self._unmapped_accepted)
        print(
            f"[AsyncPipeline][{self._increment_profile}] Projecting merged batch "
            f"({n_acc} unmapped proposal(s)) — dedicated thread (LLM + templates)…"
        )
        merged, manifest, partial = await asyncio.to_thread(
            _build_merged_bundle_sync,
            enriched,
            list(self._unmapped_accepted),
            self._partial_fp_results,
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            azure_api_version=self.azure_api_version,
            azure_deployment=self.azure_deployment,
        )
        self._partial_fp_results = partial
        n_total = sum(len(fps.all_policies()) for fps in merged.values())
        print(
            f"[AsyncPipeline][{self._increment_profile}] Merged batch: {n_total} policy/policies "
            f"({manifest.get('count_added', 0)} unmapped FPd added) — "
            "sending FP_BUNDLE_READY → syntax/semantic audit."
        )
        bundle_msg = AgentMessage(
            sender="pipeline",
            recipient="agent4",
            msg_type=MessageType.FP_BUNDLE_READY,
            payload={
                "fp_results": merged,
                "enriched_graph": enriched,
                "manifest": manifest,
                "status": "final_merged",
                "utterance": (
                    "Merged template and accepted unmapped policies; "
                    "run syntax then semantic validation before any disk export."
                ),
            },
            loop_turn=loop_t,
        )
        # Direct delivery on the asyncio loop (not via agent4 bus or to_thread:
        # receive() only emits SYNTAX_AUDIT_REQUEST via call_soon_threadsafe).
        print(
            f"[AsyncPipeline][{self._increment_profile}] Delivering FP_BUNDLE_READY → agent4 "
            f"({n_total} policies, syntax then semantic audit)…"
        )
        agent4.receive(bundle_msg)
        await asyncio.sleep(0)  # let the _safe_publish task scheduled by send() run
        print(
            f"[AsyncPipeline][{self._increment_profile}] FP_BUNDLE_READY handled by agent 4 — "
            "pipeline continues via bus (A5 / A3)."
        )

        self._pending_unmapped_env = None
        self._unmapped_queue = []
        self._unmapped_accepted = []
        self._unmapped_pending_by_request = {}

    async def _ask_next_unmapped(self) -> None:
        if self._pending_unmapped_env is None:
            return
        if not self._unmapped_queue:
            return

        proposal = self._unmapped_queue.pop(0)
        done = len(self._unmapped_accepted)
        remaining_after = len(self._unmapped_queue)
        total = done + remaining_after + 1

        pt = proposal.get("pattern_type", "?")
        gw = proposal.get("gateway_name", "?")
        fid = proposal.get("fragment_id", "?")
        rt = proposal.get("odrl_rule_type", "?")
        conf = proposal.get("confidence", None)
        hint = (proposal.get("hint_text") or "").strip()
        just = (proposal.get("justification") or "").strip()

        summary_lines = [
            f"Unmapped proposal ({done + 1} of {total})",
            f"- pattern: {pt}",
            f"- fragment: {fid}",
            f"- gateway: {gw}",
            f"- rule kind: {rt}",
        ]
        if conf is not None:
            summary_lines.append(f"- confidence: {conf}")
        if hint:
            summary_lines.append("")
            summary_lines.append("summary from agent:")
            summary_lines.append(hint[:1200] + ("…" if len(hint) > 1200 else ""))
        if just:
            summary_lines.append("")
            summary_lines.append("rationale:")
            summary_lines.append(just[:1200] + ("…" if len(just) > 1200 else ""))

        activity_directory = [
            {"name": str(a.get("name")), "role": str(a.get("role") or "")}
            for a in self.bp_model.get("activities", [])
            if isinstance(a, dict) and a.get("name")
        ]

        req = ACLEnvelope(
            performative=ACLPerformative.REQUEST,
            sender="agent3",
            receiver="human3",
            ontology="human-gate",
            content={
                "title": "Review proposed rule",
                "summary": "\n".join(summary_lines),
                "proposal": proposal,
                "status": "pending_human_unmapped",
                "conversation_id": self._conversation_id,
                "accepted_so_far": done,
                "remaining_after": remaining_after,
                "hitl_context": {
                    "process_title": self._process_display_name,
                    "scenario_id": self._scenario_id,
                    "activity_directory": activity_directory,
                },
            },
            conversation_id=self._conversation_id,
            in_reply_to=self._pending_unmapped_env.reply_with,
        )
        self._unmapped_pending_by_request[req.reply_with] = proposal
        await self.bus.publish(req)


def _build_merged_bundle_sync(
    enriched: Any,
    accepted_proposals: list[Any],
    partial_fp_results: Optional[dict[str, Any]],
    *,
    api_key: Optional[str],
    azure_endpoint: Optional[str],
    azure_api_version: Optional[str],
    azure_deployment: Optional[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Template projection + unmapped FPd (LLM calls) off the asyncio loop.
    """
    report = ValidationReport(results=[])
    report.accepted_unmapped_proposals = list(accepted_proposals)

    projector = PolicyProjectionAgent(
        enriched_graph=enriched,
        validation_report=report,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        azure_deployment=azure_deployment,
    )
    print("[AsyncPipeline] Generating unmapped FPd (Agent 4, LLM)…")
    unmapped_only = projector.generate_unmapped_only(enriched, report)

    manifest = {
        "merge_strategy": "additive_unmapped_only",
        "added_policy_uids": _collect_policy_uids(unmapped_only),
        "count_added": sum(len(v) for v in _collect_policy_uids(unmapped_only).values()),
    }

    partial = partial_fp_results
    if partial is None:
        print("[AsyncPipeline] Deterministic template projection (Agent 4)…")
        templates_report = ValidationReport(results=[])
        templates_projector = PolicyProjectionAgent(
            enriched_graph=enriched,
            validation_report=templates_report,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            azure_deployment=azure_deployment,
        )
        partial = templates_projector.project(enriched, templates_report)

    merged = _merge_fp_results_additive(partial, unmapped_only)
    return merged, manifest, partial


def _collect_policy_uids(fp_results: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for frag_id, fps in (fp_results or {}).items():
        uids: list[str] = []
        for pol in fps.all_policies():
            uid = pol.get("uid")
            if uid:
                uids.append(uid)
        out[frag_id] = uids
    return out


def _merge_fp_results_additive(base: dict[str, Any], add: dict[str, Any]) -> dict[str, Any]:
    """
    Additive merge:
    - Keep all base policies unchanged.
    - Append added FPd policies (unmapped) into corresponding FragmentPolicySet.fpd_policies.
    """
    merged = base
    for frag_id, fps_add in (add or {}).items():
        if frag_id not in merged:
            merged[frag_id] = fps_add
            continue
        fps_base = merged[frag_id]
        for pol in fps_add.fpd_policies:
            fps_base.fpd_policies.append(pol)
    return merged


async def run_pipeline_async(
    *,
    bp_model: dict,
    fragments: list[dict],
    b2p_policies: list[dict],
    human_timeout_s: float = 120.0,
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_api_version: Optional[str] = None,
    azure_deployment: Optional[str] = None,
    human_decision_bridge: Optional[HumanDecisionBridge] = None,
    on_acl_message: Optional[PublishHook] = None,
    scenario_id: str = "scenario1",
    process_title: Optional[str] = None,
    accept_all_unmapped: bool = False,
    enable_hitl: bool = True,
    max_wall_s: Optional[float] = None,
    increment_profile: str = "I5",
) -> PipelineAsyncResult:
    human_agent: Optional[HumanAgent] = None
    if enable_hitl:
        human_agent = HumanAgent(
            timeout_s=human_timeout_s,
            decision_bridge=human_decision_bridge,
        )
    orch = AsyncPipelineOrchestrator(
        bp_model=bp_model,
        fragments=fragments,
        b2p_policies=b2p_policies,
        human_timeout_s=human_timeout_s,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        azure_deployment=azure_deployment,
        human_agent=human_agent,
        acl_trace=on_acl_message,
        scenario_id=scenario_id,
        process_title=process_title,
        accept_all_unmapped=accept_all_unmapped,
        enable_hitl=enable_hitl,
        max_wall_s=max_wall_s,
        increment_profile=increment_profile,
    )
    print(f"[AsyncPipeline] Incremental profile: {increment_profile.strip().upper()}")
    return await orch.run()


def _scenario_odrl_stage_dir(scenario_id: str, stage: str) -> Path:
    """
    ``{project}/output/{scenarioN}/odrl_fragment_policies/{stage}/``
    Export directory distinct from ``dataset/``; one subdirectory per fragment (flat files).
    """
    project_root = Path(__file__).resolve().parents[2]
    sid = (scenario_id or "scenario1").strip()
    if not re.fullmatch(r"scenario\d+", sid, re.IGNORECASE):
        sid = "scenario1"
    return (project_root / "output" / sid / "odrl_fragment_policies" / stage).resolve()

