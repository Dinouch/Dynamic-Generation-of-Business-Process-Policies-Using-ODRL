"""
pipeline.py — Orchestration du pipeline multi-agent ODRL

Câble les 5 agents et expose run_pipeline() comme point d'entrée unique.

Flux nominal (refactor) :
    Agent 1 ──GRAPH_READY──► Agent 3
    Agent 3 ──GRAPH_READY──► Agent 2  (si patterns non couverts)
    Agent 2 ──UNHANDLED_PROPOSALS──► Agent 3
    Agent 3 ──VALIDATION_DONE──► Agent 4
    Agent 4 ──POLICIES_READY──► Agent 3  (validation sémantique)
    Agent 3 ──SEMANTIC_CORRECTION / SEMANTIC_VALIDATED──► Agent 4
    Agent 4 ──POLICIES_READY──► Agent 5
    Agent 5 ──SYNTAX_CORRECTION──► Agent 4
    Agent 5 ──ODRL_VALID / ODRL_SYNTAX_ERROR──► ResultCollector

Le pipeline est entièrement synchrone : chaque receive() appelle le suivant
en cascade.
"""

from dataclasses import dataclass
from typing import Optional

from agents.structural_analyzer import (
    StructuralAnalyzer,
    AgentMessage,
    MessageType,
)
from agents.pipeline_registry import COVERED_PATTERNS
from agents.unhandled_case_formulator import UnhandledCaseFormulator
from agents.constraint_validator import ConstraintValidator
from agents.policy_projection_agent import PolicyProjectionAgent
from agents.policy_auditor import PolicyAuditor, AuditReport


@dataclass
class PipelineResult:
    """
    Résultat final retourné par run_pipeline().

    Attributs
    ----------
    is_valid    : True si aucune issue CRITICAL dans l'audit final.
    report      : AuditReport complet produit par Agent 5.
    summary     : dict résumé (counts, scores, validité par layer).
    syntax_score: score syntaxique [0.0 – 1.0].
    msg_type    : MessageType.ODRL_VALID ou MessageType.ODRL_SYNTAX_ERROR.
    loop_turns  : nombre de tours de boucle syntaxique utilisés (Agent 5).
    semantic_loop_turns : tours de boucle sémantique Agent 3 ↔ Agent 4 (informatif).
    """

    is_valid: bool
    report: AuditReport
    summary: dict
    syntax_score: float
    msg_type: MessageType
    loop_turns: int = 0
    semantic_loop_turns: int = 0


class ResultCollector:
    """Endpoint final du pipeline (signaux Agent 5)."""

    def __init__(self):
        self._result: Optional[PipelineResult] = None

    def receive(self, msg: AgentMessage) -> None:
        """Reçoit le signal final d'Agent 5 et stocke le résultat."""
        print(f"[Collector] ◄ RECEIVE {msg}")
        self._result = PipelineResult(
            is_valid=msg.payload["is_valid"],
            report=msg.payload["report"],
            summary=msg.payload["summary"],
            syntax_score=msg.payload.get("syntax_score", 1.0),
            msg_type=msg.msg_type,
            loop_turns=msg.payload.get("loop_turns_used", 0),
            semantic_loop_turns=msg.payload.get("semantic_loop_turns", 0),
        )

    def get_result(self) -> PipelineResult:
        if self._result is None:
            raise RuntimeError(
                "Pipeline non terminé — aucun résultat reçu d'Agent 5."
            )
        return self._result


def run_pipeline(
    bp_model: dict,
    fragments: dict,
    b2p_policies: list[dict],
    context_mode: str = "local",
    *,
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_api_version: Optional[str] = None,
    azure_deployment: Optional[str] = None,
) -> PipelineResult:
    """
    Câble les 5 agents et lance le pipeline ODRL.

    Parameters
    ----------
    context_mode
        Conservé pour compatibilité ; l'Agent 2 n'utilise plus de mode local/global explicite.
    """
    _ = context_mode
    print("[Pipeline] Initialisation du pipeline multi-agent ODRL")

    agent1 = StructuralAnalyzer(
        bp_model,
        fragments,
        b2p_policies,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        azure_deployment=azure_deployment,
    )

    agent2 = UnhandledCaseFormulator(
        covered_patterns=COVERED_PATTERNS,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        azure_deployment=azure_deployment,
    )

    agent3 = ConstraintValidator(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        azure_deployment=azure_deployment,
    )

    agent4 = PolicyProjectionAgent(
        enriched_graph=None,
        validation_report=None,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        azure_deployment=azure_deployment,
    )

    agent5 = PolicyAuditor(
        fp_results={},
        enriched_graph=None,
        raw_b2p=b2p_policies,
    )

    collector = ResultCollector()

    def agent4_router(msg: AgentMessage) -> None:
        """Dispatch messages targeting Agent 4, or Agent 4's outbound routing."""
        if msg.recipient == "agent4":
            agent4.receive(msg)
        elif msg.recipient == "agent3":
            agent3.receive(msg)
        elif msg.recipient == "agent5":
            agent5.receive(msg)
        else:
            print(f"[Pipeline][WARN] Routage inconnu : {msg.recipient}")

    agent1.register_send_callback(agent3.receive)
    # Agent 2 → Agent 3 (UNHANDLED_PROPOSALS, etc.)
    agent2.register_send_callback(agent3.receive)
    agent3.register_send_callback_agent2(agent2.receive)
    agent3.register_send_callback_agent1(agent1.receive)
    agent3.register_send_callback_agent4(agent4_router)
    agent4.register_send_callback(agent4_router)

    def agent5_router(msg: AgentMessage) -> None:
        if msg.recipient == "agent4":
            agent4.receive(msg)
        else:
            collector.receive(msg)

    agent5.register_send_callback(agent5_router)

    print("[Pipeline] Câblage des 5 agents terminé — démarrage de l'analyse")

    agent1.analyze_and_send()

    result = collector.get_result()

    try:
        final_fp_results = getattr(agent5, "fp_results", None)
        if final_fp_results:
            import os

            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            output_dir = os.path.join(root_dir, "odrl_policies_pipeline")

            exporter = PolicyProjectionAgent(
                enriched_graph=agent5.enriched_graph,
                validation_report=None,
            )
            exported = exporter.export(final_fp_results, output_dir=output_dir)
            total_files = sum(len(v) for v in exported.values())
            print(
                f"[Pipeline] Export JSON-LD pipeline : {total_files} fichiers "
                f"→ '{output_dir}'"
            )
    except Exception as e:
        print(f"[Pipeline][WARN] Export JSON-LD pipeline impossible : {e}")

    status = "✅ VALID" if result.is_valid else "❌ INVALID"
    print(
        f"[Pipeline] Terminé — {status} | "
        f"syntax_score={result.syntax_score:.2f} | "
        f"syntax_loops={result.loop_turns} | "
        f"semantic_loops={result.semantic_loop_turns} | "
        f"signal={result.msg_type.value}"
    )

    return result
