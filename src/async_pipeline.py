from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

from agents.pipeline_registry import COVERED_PATTERNS
from agents.structural_analyzer import StructuralAnalyzer, AgentMessage, MessageType
from agents.unsupported_case_formulator import UnsupportedCaseFormulator
from agents.Agent_3.constraint_validator import ConstraintValidator, ValidationReport
from agents.policy_projection_agent import PolicyProjectionAgent
from agents.policy_auditor import PolicyAuditor, AuditReport
from agents.human_agent import HumanAgent, HumanDecisionBridge

from mas.acl import ACLEnvelope, ACLPerformative
from mas.bus import AsyncBus, PublishHook


@dataclass
class PipelineAsyncResult:
    is_valid: bool
    report: AuditReport
    summary: dict
    syntax_score: float
    msg_type: str
    status: str  # partial_template_only | final_merged | final_validated
    manifest: Optional[dict] = None


class AsyncPipelineOrchestrator:
    """
    Phase 1-5 implementation:
    - asyncio bus + async orchestration
    - HITL gate at Agent 3 for unsupported proposals (batch)
    - partial template-only continuation on timeout
    - merge additively with unsupported-generated FPd once human validates

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
    ):
        self.bp_model = bp_model
        self.fragments = fragments
        self.b2p_policies = b2p_policies
        self.human_timeout_s = human_timeout_s

        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self.azure_deployment = azure_deployment

        self.bus = AsyncBus(on_publish=acl_trace)
        self.human = human_agent or HumanAgent(timeout_s=human_timeout_s)

        self._conversation_id = os.urandom(8).hex()
        self._final_result: Optional[PipelineAsyncResult] = None
        self._partial_result: Optional[PipelineAsyncResult] = None
        self._partial_fp_results: Optional[dict[str, Any]] = None
        self._enriched_graph: Optional[Any] = None
        self._pending_unsupported_env: Optional[ACLEnvelope] = None
        self._unsupported_queue: list[dict] = []
        self._unsupported_accepted: list[dict] = []
        self._unsupported_pending_by_request: dict[str, dict] = {}
        self._auto_close_after_partial = os.environ.get("HITL_AUTOCLOSE_AFTER_PARTIAL", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }

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
        agent2 = UnsupportedCaseFormulator(
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
        self.bus.register("agent2", lambda env: self._deliver_to_legacy(agent2, env))
        self.bus.register("agent4", lambda env: self._deliver_to_legacy(agent4, env))
        self.bus.register("agent5", lambda env: self._deliver_to_legacy(agent5, env))

        # Agent 3 and human are orchestrated explicitly for HITL
        self.bus.register("agent3", lambda env: self._handle_agent3(env, agent3, agent4, agent5))
        self.bus.register("human3", self._handle_human3)
        self.bus.register("pipeline", self._handle_pipeline_final)

        await self.bus.start()

        try:
            # Kick off analysis (legacy call), but route first message via bus
            msg = AgentMessage(
                sender="pipeline",
                recipient="agent1",
                msg_type=MessageType.GRAPH_READY,
                payload={"_kickoff": True},
            )
            # Instead of sending kickoff to agent1.receive, just call analyze_and_send and intercept send
            def _threadsafe_publish(m: AgentMessage, *, status: Optional[str] = None) -> None:
                loop.call_soon_threadsafe(asyncio.create_task, self._publish_legacy(m, status=status))

            agent1.register_send_callback(lambda m: _threadsafe_publish(m, status=None))
            agent2.register_send_callback(lambda m: _threadsafe_publish(m, status=None))
            agent3.register_send_callback_agent1(lambda m: _threadsafe_publish(m, status=None))
            agent3.register_send_callback_agent2(lambda m: _threadsafe_publish(m, status=None))

            # Route Agent 3 -> Agent 4/1/2 through bus
            agent3.register_send_callback_agent4(lambda m: _threadsafe_publish(m, status=None))
            agent4.register_send_callback(lambda m: _threadsafe_publish(m, status=None))
            agent5.register_send_callback(lambda m: _threadsafe_publish(m, status=None))

            _ = msg
            # Start actual pipeline from Agent 1
            await asyncio.to_thread(agent1.analyze_and_send)

            # Wait for final result
            while self._final_result is None:
                await asyncio.sleep(0.05)

            return self._final_result
        finally:
            await self.bus.stop()

    async def _publish_legacy(self, msg: AgentMessage, *, status: Optional[str]) -> None:
        from mas.legacy_adapter import agent_message_to_acl

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
        await self.bus.publish(env)

    async def _deliver_to_legacy(self, agent: Any, env: ACLEnvelope) -> None:
        from mas.legacy_adapter import acl_to_agent_message

        msg = acl_to_agent_message(env)
        # Capture partial template-only bundle when Agent 4 forwards to Agent 5.
        if (
            env.sender == "agent4"
            and env.receiver == "agent5"
            and (env.content or {}).get("status") == "partial_template_only"
            and "fp_results" in (env.content or {})
        ):
            self._partial_fp_results = env.content.get("fp_results")
            try:
                out_dir = _default_output_dir("odrl_policies_partial_template_only")
                exporter = PolicyProjectionAgent(enriched_graph=self._enriched_graph, validation_report=None)
                exporter.export(self._partial_fp_results, output_dir=str(out_dir))
            except Exception as e:
                print(f"[AsyncPipeline][WARN] Export partial impossible : {e}")
        await asyncio.to_thread(agent.receive, msg)

    async def _handle_agent3(
        self,
        env: ACLEnvelope,
        agent3: ConstraintValidator,
        agent4: PolicyProjectionAgent,
        agent5: PolicyAuditor,
    ) -> None:
        """
        Intercept unsupported proposals path for HITL.

        - If env is UNSUPPORTED_PROPOSALS: pause for human approval.
        - Meanwhile, allow templates-only partial generation (timeout path).
        """
        from mas.legacy_adapter import acl_to_agent_message

        # Track enriched graph for later merges/exports
        if env.ontology in {"graph-structural", "unsupported-formulation", "validation", "policy-projection"}:
            if isinstance(env.content, dict) and "enriched_graph" in env.content:
                self._enriched_graph = env.content.get("enriched_graph")

        msg = acl_to_agent_message(env)

        # Human replies are routed to agent3: resume the HITL gate.
        if env.sender == "human3" and env.in_reply_to:
            await self.on_human_reply_to_agent3(env)
            return

        if msg.msg_type == MessageType.UNSUPPORTED_PROPOSALS:
            # Store pending and request human validation one-by-one.
            self._pending_unsupported_env = env
            self._unsupported_queue = list(env.content.get("unsupported_proposals") or [])
            self._unsupported_accepted = []
            self._unsupported_pending_by_request = {}

            if not self._unsupported_queue:
                self._pending_unsupported_env = None
                await asyncio.to_thread(agent3.receive, msg)
                return

            await self._ask_next_unsupported()

            # Schedule timeout: proceed with partial template-only
            print(f"[AsyncPipeline] HITL gate opened — timeout={self.human_timeout_s}s")
            asyncio.create_task(self._timeout_partial_templates(agent4, agent5, base_env=env))
            return

        # Default: deliver to legacy Agent 3 (semantic loop etc.)
        await asyncio.to_thread(agent3.receive, msg)

    async def _timeout_partial_templates(
        self,
        agent4: PolicyProjectionAgent,
        agent5: PolicyAuditor,
        *,
        base_env: ACLEnvelope,
    ) -> None:
        """
        After human_timeout_s, if still waiting, proceed with templates-only generation:
        Agent 4 generates with empty ValidationReport (no accepted unsupported) and status partial_template_only,
        then Agent 5 audits that partial bundle.
        """
        try:
            await asyncio.sleep(self.human_timeout_s)
            if self._pending_unsupported_env is None:
                return

            enriched = base_env.content.get("enriched_graph")
            if enriched is None:
                print("[AsyncPipeline][WARN] Timeout partial: enriched_graph missing.")
                return

            report = ValidationReport(results=[])
            # Generate templates-only by sending VALIDATION_DONE to Agent 4
            msg = AgentMessage(
                sender="agent3",
                recipient="agent4",
                msg_type=MessageType.VALIDATION_DONE,
                payload={"validation_report": report, "enriched_graph": enriched},
                loop_turn=int(base_env.content.get("loop_turn", 0) or 0),
            )
            print("[AsyncPipeline] HITL timeout reached — generating template-only partial bundle.")
            await self._publish_legacy(msg, status="partial_template_only")
        except Exception as e:
            print(f"[AsyncPipeline][ERROR] Timeout partial generation failed: {e}")

        # Try exporting the partial bundle once Agent 5 has received it later;
        # if we already captured it, export now.
        try:
            if self._partial_fp_results is not None:
                out_dir = _default_output_dir("odrl_policies_partial_template_only")
                exporter = PolicyProjectionAgent(enriched_graph=enriched, validation_report=None)
                exporter.export(self._partial_fp_results, output_dir=str(out_dir))
        except Exception as e:
            print(f"[AsyncPipeline][WARN] Export partial impossible : {e}")

        # Note: partial fp_results is captured when we intercept POLICIES_READY to agent5.
        _ = agent4
        _ = agent5

    async def _handle_human3(self, env: ACLEnvelope) -> None:
        # HumanAgent returns a response envelope or None on timeout
        resp = await self.human.handle(env)
        if resp is None:
            return
        await self.bus.publish(resp)

    async def _handle_pipeline_final(self, env: ACLEnvelope) -> None:
        """
        Receive final signals from Agent 5 and set orchestrator result.

        Also handles the 'final_merged' path: after human agrees, we create the
        merged bundle and ask Agent 5 to audit again.
        """
        # Final from Agent 5 to pipeline
        if env.sender == "agent5" and env.receiver == "pipeline":
            # If we're still waiting for the human gate, the first Agent 5 result is a partial bundle.
            inferred = "final_validated"
            if self._pending_unsupported_env is not None and self._partial_result is None:
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
            )
            if status == "partial_template_only" and not self._auto_close_after_partial:
                self._partial_result = out
                print(
                    "[AsyncPipeline] Partial bundle ready (template-only). "
                    "Waiting for human decision to merge unsupported cases..."
                )
                return

            self._final_result = out

    async def on_human_reply_to_agent3(self, env: ACLEnvelope) -> None:
        """
        Called when human replies AGREE/REFUSE to one unsupported proposal.
        """
        if self._pending_unsupported_env is None:
            return

        req_id = env.in_reply_to or ""
        proposal = self._unsupported_pending_by_request.pop(req_id, None)
        if proposal is None:
            print(
                "[AsyncPipeline][WARN] Réponse humaine ignorée : aucune proposition en attente pour "
                f"in_reply_to={req_id!r}. Clés connues : {list(self._unsupported_pending_by_request.keys())!r}"
            )
            return

        decision = env.performative
        if decision == ACLPerformative.AGREE:
            self._unsupported_accepted.append(proposal)

        enriched = self._pending_unsupported_env.content.get("enriched_graph")
        if enriched is None:
            self._pending_unsupported_env = None
            self._unsupported_queue = []
            self._unsupported_accepted = []
            self._unsupported_pending_by_request = {}
            return

        # Continue prompting until all proposals are processed.
        if self._unsupported_queue:
            await self._ask_next_unsupported()
            return

        # Human finished validating all proposals.
        if not self._unsupported_accepted:
            self._pending_unsupported_env = None
            self._unsupported_queue = []
            self._unsupported_accepted = []
            self._unsupported_pending_by_request = {}
            if self._partial_result is not None and self._final_result is None:
                self._final_result = self._partial_result
            return

        report = ValidationReport(results=[])
        report.accepted_unsupported_proposals = list(self._unsupported_accepted)

        # Generate only the unsupported FPd policies and merge additively with any partial bundle we have.
        projector = PolicyProjectionAgent(enriched_graph=enriched, validation_report=report)
        unsupported_only = projector.generate_unsupported_only(enriched, report)

        manifest = {
            "conversation_id": self._conversation_id,
            "merge_strategy": "additive_unsupported_only",
            "added_policy_uids": _collect_policy_uids(unsupported_only),
            "count_added": sum(len(v) for v in _collect_policy_uids(unsupported_only).values()),
        }

        # If partial exists, merge; else start from templates-only generation (no timeout path hit).
        if self._partial_fp_results is None:
            # generate templates-only immediately
            templates_report = ValidationReport(results=[])
            templates_projector = PolicyProjectionAgent(enriched_graph=enriched, validation_report=templates_report)
            self._partial_fp_results = templates_projector.project(enriched, templates_report)

        merged = _merge_fp_results_additive(self._partial_fp_results, unsupported_only)

        # Export final merged bundle (separate folder)
        try:
            out_dir = _default_output_dir("odrl_policies_final_merged")
            exporter = PolicyProjectionAgent(enriched_graph=enriched, validation_report=None)
            exporter.export(merged, output_dir=str(out_dir))
        except Exception as e:
            print(f"[AsyncPipeline][WARN] Export final merged impossible : {e}")

        # Ask Agent 5 to audit merged and emit final
        from mas.legacy_adapter import agent_message_to_acl
        msg = AgentMessage(
            sender="pipeline",
            recipient="agent5",
            msg_type=MessageType.POLICIES_READY,
            payload={
                "fp_results": merged,
                "enriched_graph": enriched,
                "manifest": manifest,
                "status": "final_merged",
            },
            loop_turn=int(self._pending_unsupported_env.content.get("loop_turn", 0) or 0),
        )
        env2 = agent_message_to_acl(msg, conversation_id=self._conversation_id, status="final_merged")
        env2 = ACLEnvelope(
            performative=env2.performative,
            sender=env2.sender,
            receiver=env2.receiver,
            ontology=env2.ontology,
            content=dict(env2.content, manifest=manifest, status="final_merged"),
            conversation_id=self._conversation_id,
            in_reply_to=env.reply_with,
        )
        await self.bus.publish(env2)

        self._pending_unsupported_env = None
        self._unsupported_queue = []
        self._unsupported_accepted = []
        self._unsupported_pending_by_request = {}

    async def _ask_next_unsupported(self) -> None:
        if self._pending_unsupported_env is None:
            return
        if not self._unsupported_queue:
            return

        proposal = self._unsupported_queue.pop(0)
        done = len(self._unsupported_accepted)
        remaining_after = len(self._unsupported_queue)
        total = done + remaining_after + 1

        pt = proposal.get("pattern_type", "?")
        gw = proposal.get("gateway_name", "?")
        fid = proposal.get("fragment_id", "?")
        rt = proposal.get("odrl_rule_type", "?")
        conf = proposal.get("confidence", None)
        hint = (proposal.get("hint_text") or "").strip()
        just = (proposal.get("justification") or "").strip()

        summary_lines = [
            f"Unsupported proposal ({done + 1} of {total})",
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

        req = ACLEnvelope(
            performative=ACLPerformative.QUERY_IF,
            sender="agent3",
            receiver="human3",
            ontology="human-gate",
            content={
                "title": "Review proposed rule",
                "summary": "\n".join(summary_lines),
                "proposal": proposal,
                "status": "pending_human_unsupported",
                "conversation_id": self._conversation_id,
                "accepted_so_far": done,
                "remaining_after": remaining_after,
            },
            conversation_id=self._conversation_id,
            in_reply_to=self._pending_unsupported_env.reply_with,
        )
        self._unsupported_pending_by_request[req.reply_with] = proposal
        await self.bus.publish(req)


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
    - Append added FPd policies (unsupported) into corresponding FragmentPolicySet.fpd_policies.
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
) -> PipelineAsyncResult:
    human_agent = (
        HumanAgent(timeout_s=human_timeout_s, decision_bridge=human_decision_bridge)
        if human_decision_bridge is not None
        else None
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
    )
    return await orch.run()


def _default_output_dir(folder_name: str) -> Path:
    root = Path(__file__).resolve().parent.parent
    return (root / folder_name).resolve()

