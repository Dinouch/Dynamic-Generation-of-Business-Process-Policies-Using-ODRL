"""
pipeline.py — Orchestration du pipeline multi-agent ODRL

Câble les 5 agents et expose run_pipeline() comme point d'entrée unique.

Flux nominal :
    Agent 1 ──GRAPH_READY──► Agent 2 ──CANDIDATES_READY──► Agent 3
                                 ▲                              │
                                 │         REFORMULATE ◄────────┤ (boucle courte)
                                 │                              │
               Agent 1 ◄──STRUCTURAL_UPDATE──────────────────┘  (boucle structurelle)
                                                               │
                                                    VALIDATION_DONE
                                                               │
    Agent 5 ◄──POLICIES_READY──── Agent 4 ◄───────────────────┘
        │                             ▲
        └──SYNTAX_CORRECTION──────────┘  (boucle syntaxique)
        │
        └──ODRL_VALID / ODRL_SYNTAX_ERROR──► ResultCollector

Le pipeline est entièrement synchrone : chaque receive() appelle le suivant
en cascade. Quand agent1.analyze_and_send() retourne, tout est terminé.
"""

from dataclasses import dataclass
from typing import Optional

# Imports absolus depuis le package agents, car pipeline.py
# est chargé comme module top-level (pas comme package).
from agents.structural_analyzer import (
    StructuralAnalyzer,
    AgentMessage,
    MessageType,
)
from agents.implicit_dependency_detector import ImplicitDependencyDetector
from agents.constraint_validator import ConstraintValidator
from agents.policy_projection_agent import PolicyProjectionAgent
from agents.policy_auditor import PolicyAuditor, AuditReport


# ══════════════════════════════════════════════════════════════════════
#  Résultat du pipeline
# ══════════════════════════════════════════════════════════════════════

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
    """
    is_valid:    bool
    report:      AuditReport
    summary:     dict
    syntax_score: float
    msg_type:    MessageType   # ODRL_VALID ou ODRL_SYNTAX_ERROR
    loop_turns:  int = 0


# ══════════════════════════════════════════════════════════════════════
#  Collecteur de résultat
# ══════════════════════════════════════════════════════════════════════

class ResultCollector:
    """
    Endpoint final du pipeline.
    Agent 5 lui envoie ODRL_VALID ou ODRL_SYNTAX_ERROR via son callback.
    """

    def __init__(self):
        self._result: Optional[PipelineResult] = None

    def receive(self, msg: AgentMessage) -> None:
        """Reçoit le signal final d'Agent 5 et stocke le résultat."""
        print(f"[Collector] ◄ RECEIVE {msg}")
        self._result = PipelineResult(
            is_valid     = msg.payload["is_valid"],
            report       = msg.payload["report"],
            summary      = msg.payload["summary"],
            syntax_score = msg.payload.get("syntax_score", 1.0),
            msg_type     = msg.msg_type,
            loop_turns   = msg.payload.get("loop_turns_used", 0),
        )

    def get_result(self) -> PipelineResult:
        if self._result is None:
            raise RuntimeError(
                "Pipeline non terminé — aucun résultat reçu d'Agent 5. "
                "Vérifier que tous les callbacks sont correctement câblés."
            )
        return self._result


# ══════════════════════════════════════════════════════════════════════
#  Point d'entrée principal
# ══════════════════════════════════════════════════════════════════════

def run_pipeline(
    bp_model:     dict,
    fragments:    dict,
    b2p_policies: list[dict],
    context_mode: str = "global",
    *,
    api_key:           Optional[str] = None,
    azure_endpoint:    Optional[str] = None,
    azure_api_version: Optional[str] = None,
    azure_deployment:  Optional[str] = None,
) -> PipelineResult:
    """
    Câble les 5 agents et lance le pipeline ODRL.

    Retourne un PipelineResult quand Agent 5 émet son signal final
    (ODRL_VALID ou ODRL_SYNTAX_ERROR).

    Paramètres
    ----------
    bp_model          : modèle BPMN brut (dict parsé)
    fragments         : découpage en fragments du BPMN
    b2p_policies      : liste des policies B2P d'origine
    context_mode      : "global" ou "local" — mode de contexte pour Agent 2
    api_key           : clé OpenAI (optionnel si Azure)
    azure_endpoint    : endpoint Azure OpenAI (optionnel)
    azure_api_version : version API Azure (optionnel)
    azure_deployment  : nom du déploiement Azure (optionnel)
    """

    print("[Pipeline] Initialisation du pipeline multi-agent ODRL")

    # ── Instanciation des agents ─────────────────────────────────────

    agent1 = StructuralAnalyzer(bp_model, fragments, b2p_policies)

    agent2 = ImplicitDependencyDetector(
        context_mode      = context_mode,
        api_key           = api_key,
        azure_endpoint    = azure_endpoint,
        azure_api_version = azure_api_version,
        azure_deployment  = azure_deployment,
    )

    agent3 = ConstraintValidator(
        api_key           = api_key,
        azure_endpoint    = azure_endpoint,
        azure_api_version = azure_api_version,
        azure_deployment  = azure_deployment,
    )

    # Agent 4 et 5 reçoivent leurs données via les messages —
    # on les instancie avec des valeurs vides, elles seront remplacées.
    agent4 = PolicyProjectionAgent(
        enriched_graph    = None,
        validation_report = None,
    )

    agent5 = PolicyAuditor(
        fp_results     = {},
        enriched_graph = None,
        raw_b2p        = b2p_policies,
    )

    collector = ResultCollector()

    # ── Câblage des callbacks ────────────────────────────────────────

    # Chemin nominal : 1 → 2 → 3 → 4 → 5 → collecteur
    agent1.register_send_callback(agent2.receive)           # 1 → 2

    agent2.register_send_callback(agent3.receive)           # 2 → 3

    agent3.register_send_callback_agent2(agent2.receive)    # 3 → 2  (boucle courte)
    agent3.register_send_callback_agent1(agent1.receive)    # 3 → 1  (boucle structurelle)
    agent3.register_send_callback_agent4(agent4.receive)    # 3 → 4  (sortie normale)

    agent4.register_send_callback(agent5.receive)           # 4 → 5

    # Agent 5 envoie soit vers Agent 4 (SYNTAX_CORRECTION), soit vers le collecteur (signal final)
    def agent5_router(msg: AgentMessage) -> None:
        if msg.recipient == "agent4":
            agent4.receive(msg)
        else:
            collector.receive(msg)
    agent5.register_send_callback(agent5_router)

    print("[Pipeline] Câblage des 5 agents terminé — démarrage de l'analyse")

    # ── Démarrage ────────────────────────────────────────────────────
    # Le pipeline est synchrone : analyze_and_send() déclenche la cascade
    # complète de receive() en profondeur d'abord.
    # Quand cette ligne retourne, le collecteur a reçu son résultat.
    agent1.analyze_and_send()

    result = collector.get_result()

    # ── Export JSON-LD des policies finales du pipeline ───────────────
    # On utilise les fp_results actuellement en mémoire dans Agent 5,
    # qui incluent les éventuelles corrections appliquées via la boucle
    # syntaxique Agent 5 ↔ Agent 4.
    try:
        final_fp_results = getattr(agent5, "fp_results", None)
        if final_fp_results:
            # Export dédié au mode pipeline pour ne pas écraser l'export standalone
            import os
            root_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            output_dir = os.path.join(root_dir, "odrl_policies_pipeline")

            exporter = PolicyProjectionAgent(
                enriched_graph    = agent5.enriched_graph,
                validation_report = None,
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
        f"signal={result.msg_type.value}"
    )

    return result