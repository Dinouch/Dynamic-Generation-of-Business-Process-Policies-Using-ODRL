"""
constraint_validator.py — Agent 3 : Constraint Validator

Responsabilité unique :
    Valider les candidats proposés par l'Agent 2.
    Accepter, rejeter, ou demander reformulation.

Architecture à deux niveaux :
    Niveau 1 — Filtre déterministe (règles objectives)
        → Rejette immédiatement les violations structurelles claires :
            - Cycle invalide introduit par le candidat
            - Doublon d'une dépendance explicite déjà dans le graphe
            - Violation d'un gateway (ex: les deux branches d'un XOR activées)
            - Contradiction directe avec une B2P policy existante
            - Auto-référence (source == target)

    Niveau 2 — LLM juge (cas ambigus)
        → Pour les candidats qui passent le filtre déterministe,
          le LLM statue sur la cohérence sémantique :
            - La dépendance est-elle cohérente avec les B2P policies ?
            - Crée-t-elle une contradiction inter-fragments ?
            - La justification est-elle suffisamment solide ?
            - Le graphe est-il structurellement complet pour valider
              cette dépendance inter-fragment ? (→ STRUCTURAL_ERROR)

Sortie :
    ValidationResult par candidat :
        - ACCEPTED        : dépendance validée, intégrable au graphe
        - REJECTED        : dépendance rejetée avec motif
        - REFORMULATE     : dépendance à reformuler (retour à Agent 2)
        - STRUCTURAL_ERROR: graphe incomplet, arête manquante (retour à Agent 1)

Couche multi-agent :
    - receive()  : accepte CANDIDATES_READY depuis Agent 2
    - send()     : routing vers Agent 1, 2, ou 4 selon la décision
    - Boucle courte     (REFORMULATE → Agent 2, max géré par Agent 2)
    - Boucle structurelle (STRUCTURAL_UPDATE → Agent 1, max MAX_STRUCTURAL_LOOPS)
    - Sortie normale    (VALIDATION_DONE → Agent 4)
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from openai import OpenAI, AzureOpenAI

from .structural_analyzer import (
    AgentMessage,
    EnrichedGraph,
    MessageType,
    ReanalysisLimitError,
)
from .implicit_dependency_detector import CandidateDependency, ImplicitAnalysisResult


# ─────────────────────────────────────────────
#  Décision de validation
# ─────────────────────────────────────────────

class ValidationDecision(Enum):
    ACCEPTED         = "accepted"
    REJECTED         = "rejected"
    REFORMULATE      = "reformulate"
    STRUCTURAL_ERROR = "structural_error"   # graphe incomplet — renvoyer à Agent 1


class RejectionReason(Enum):
    """Motifs de rejet déterministes — traçables et reproductibles."""
    SELF_REFERENCE      = "self_reference"       # source == target
    DUPLICATE_EXPLICIT  = "duplicate_explicit"   # déjà dans le BPMN
    DUPLICATE_CANDIDATE = "duplicate_candidate"  # doublon entre fragments
    CREATES_CYCLE       = "creates_cycle"        # introduit un cycle invalide
    VIOLATES_GATEWAY    = "violates_gateway"     # contredit la logique XOR/AND
    B2P_CONTRADICTION   = "b2p_contradiction"    # contredit une B2P policy
    LLM_REJECTED        = "llm_rejected"         # rejeté par le LLM juge
    LOW_CONFIDENCE      = "low_confidence"       # confiance trop faible


# ─────────────────────────────────────────────
#  Structures de sortie
# ─────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Résultat de validation pour un candidat donné.
    Chaque décision est traçable — motif + niveau de décision.
    """
    candidate:          CandidateDependency
    decision:           ValidationDecision
    decision_level:     str                    # "deterministic" ou "llm"
    reason:             Optional[RejectionReason] = None
    explanation:        str = ""
    llm_raw_response:   Optional[str] = None   # Pour audit
    reformulation_hint: Optional[str] = None   # Si REFORMULATE ou STRUCTURAL_ERROR
    hint:               Optional[str] = None   # Alias explicite utilisé par le routing
    description:        str = ""               # Description longue pour STRUCTURAL_UPDATE

    @property
    def is_accepted(self) -> bool:
        return self.decision == ValidationDecision.ACCEPTED

    def __repr__(self):
        icon = {
            "accepted":        "✅",
            "rejected":        "❌",
            "reformulate":     "🔄",
            "structural_error": "🏗️",
        }
        d = self.decision.value
        return (
            f"{icon.get(d,'?')} [{d.upper():16s}] "
            f"{self.candidate.source_activity} → {self.candidate.target_activity} "
            f"[{self.decision_level}]"
        )


@dataclass
class ValidationReport:
    """
    Rapport complet de l'Agent 3 pour tous les fragments.
    """
    results:           list[ValidationResult]
    accepted:          list[ValidationResult] = field(default_factory=list)
    rejected:          list[ValidationResult] = field(default_factory=list)
    reformulate:       list[ValidationResult] = field(default_factory=list)
    structural_errors: list[ValidationResult] = field(default_factory=list)
    deterministic_rejections: int = 0
    llm_rejections:    int = 0
    llm_acceptances:   int = 0

    def __post_init__(self):
        self.accepted    = [r for r in self.results if r.is_accepted]
        self.rejected    = [r for r in self.results
                           if r.decision == ValidationDecision.REJECTED]
        self.reformulate = [r for r in self.results
                           if r.decision == ValidationDecision.REFORMULATE]
        self.structural_errors = [r for r in self.results
                                  if r.decision == ValidationDecision.STRUCTURAL_ERROR]
        self.deterministic_rejections = sum(
            1 for r in self.rejected if r.decision_level == "deterministic"
        )
        self.llm_rejections = sum(
            1 for r in self.rejected if r.decision_level == "llm"
        )
        self.llm_acceptances = sum(
            1 for r in self.accepted if r.decision_level == "llm"
        )

    def validated_candidates(self) -> list[CandidateDependency]:
        """Retourne les candidats acceptés — input de l'Agent 4."""
        return [r.candidate for r in self.accepted]


# ─────────────────────────────────────────────
#  Agent 3 — Constraint Validator
# ─────────────────────────────────────────────

class ConstraintValidator:
    """
    Agent 3 du pipeline multi-agent.

    Mode standalone :
        Instancier avec enriched_graph + analysis_results et appeler validate().

    Mode pipeline :
        Instancier sans enriched_graph ni analysis_results (None).
        Les données arrivent via receive(CANDIDATES_READY).
        Enregistrer les trois callbacks de routage avant le démarrage.

    Deux niveaux de décision :
        1. Filtre déterministe — règles objectives, sans LLM
        2. LLM juge           — raisonnement sémantique sur les cas ambigus,
                                 incluant la détection STRUCTURAL_ERROR
    """

    MODEL       = "gpt-4o"
    TEMPERATURE = 0.1   # Très bas — on veut des décisions stables et reproductibles

    AGENT_NAME           = "agent3"
    MAX_STRUCTURAL_LOOPS = 1   # max de renvois vers Agent 1
    MAX_REFORMULATE      = 3   # max de demandes REFORMULATE par candidat (aligné sur Agent 2)

    def __init__(
        self,
        enriched_graph:   Optional[EnrichedGraph] = None,
        analysis_results: Optional[dict[str, ImplicitAnalysisResult]] = None,
        api_key:          Optional[str] = None,
        min_confidence:   float = 0.70,
        *,
        azure_endpoint:    Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment:  Optional[str] = None,
    ):
        self.enriched_graph   = enriched_graph
        self.analysis_results = analysis_results or {}
        self.min_confidence   = min_confidence
        self._use_azure = False
        self._deployment: Optional[str] = None

        # ── Couche multi-agent ──────────────────────────────────────
        self._on_send_agent2:        Optional[Callable[[AgentMessage], None]] = None
        self._on_send_agent1:        Optional[Callable[[AgentMessage], None]] = None
        self._on_send_agent4:        Optional[Callable[[AgentMessage], None]] = None
        self._structural_loop_count: int = 0
        self._last_enriched_graph:   Optional[EnrichedGraph] = None
        # Compteur de REFORMULATE envoyés par candidat (fragment, source, target) — évite la boucle infinie
        self._reformulate_sent_count: dict[tuple[str, str, str], int] = {}

        key_azure = api_key or os.environ.get("AZURE_OPENAI_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = (azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")).rstrip("/")
        if key_azure and endpoint:
            self._use_azure = True
            self._deployment = (
                azure_deployment
                or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
                or os.environ.get("AZURE_OPENAI_MODEL")
                or "gpt-4o"
            )
            api_ver = azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            self.client = AzureOpenAI(
                api_key=key_azure,
                api_version=api_ver,
                azure_endpoint=endpoint,
            )
        else:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "Clé manquante. Définis OPENAI_API_KEY (OpenAI) ou "
                    "AZURE_OPENAI_KEY + AZURE_OPENAI_ENDPOINT (Azure)."
                )
            self.client = OpenAI(api_key=key)

        # Pré-construire les index seulement si on a un graphe (mode standalone)
        if self.enriched_graph is not None:
            self._explicit_connections = self._index_explicit_connections()
            self._all_candidates       = self._collect_all_candidates()
        else:
            self._explicit_connections = {}
            self._all_candidates       = []

    # ═════════════════════════════════════════
    #  COUCHE MULTI-AGENT
    # ═════════════════════════════════════════

    def register_send_callback_agent2(self, fn: Callable[[AgentMessage], None]) -> None:
        """Enregistre le callback vers Agent 2 (REFORMULATE)."""
        self._on_send_agent2 = fn

    def register_send_callback_agent1(self, fn: Callable[[AgentMessage], None]) -> None:
        """Enregistre le callback vers Agent 1 (STRUCTURAL_UPDATE)."""
        self._on_send_agent1 = fn

    def register_send_callback_agent4(self, fn: Callable[[AgentMessage], None]) -> None:
        """Enregistre le callback vers Agent 4 (VALIDATION_DONE)."""
        self._on_send_agent4 = fn

    def send(self, msg: AgentMessage) -> None:
        """
        Routing du message selon msg.recipient.
        Chaque destinataire a son propre callback enregistré.
        """
        print(f"[Agent 3] ► SEND {msg}")
        routes = {
            "agent2": self._on_send_agent2,
            "agent1": self._on_send_agent1,
            "agent4": self._on_send_agent4,
        }
        fn = routes.get(msg.recipient)
        if fn:
            fn(msg)
        else:
            print(f"[Agent 3][WARN] Destinataire '{msg.recipient}' sans callback — ignoré.")

    def receive(self, msg: AgentMessage) -> None:
        """
        Point d'entrée des messages entrants.

        Messages acceptés :
          - CANDIDATES_READY : résultats de détection depuis Agent 2

        Tout autre type est ignoré avec un warning.
        """
        print(f"[Agent 3] ◄ RECEIVE {msg}")
        if msg.msg_type == MessageType.CANDIDATES_READY:
            results        = msg.payload["results"]
            enriched_graph = msg.payload["enriched_graph"]
            self._last_enriched_graph = enriched_graph
            self._validate_and_route(results, enriched_graph, loop_turn=msg.loop_turn)
        else:
            print(f"[Agent 3][WARN] Message '{msg.msg_type.value}' non géré — ignoré.")

    # ─────────────────────────────────────────
    #  Routing interne
    # ─────────────────────────────────────────

    def _validate_and_route(
        self,
        results:        dict[str, ImplicitAnalysisResult],
        enriched_graph: EnrichedGraph,
        loop_turn:      int,
    ) -> None:
        """
        Valide tous les candidats puis route vers l'agent approprié.

        Ordre de priorité (STRUCTURAL_ERROR avant REFORMULATE) :
          1. Si STRUCTURAL_ERROR trouvé et limite non atteinte → Agent 1
          2. Si REFORMULATE trouvé → Agent 2  (un seul à la fois)
          3. Aucun cas bloquant → VALIDATION_DONE vers Agent 4

        Cette priorité est intentionnelle : une fois le graphe enrichi,
        la reformulation devient peut-être inutile.
        """
        report = self.validate(enriched_graph, results)

        # ── Priorité 1 : STRUCTURAL_ERROR ───────────────────────────
        for vr in report.structural_errors:
            if self._structural_loop_count < self.MAX_STRUCTURAL_LOOPS:
                self._structural_loop_count += 1

                # Extraire l'arête à ajouter depuis le hint du LLM juge
                # Le hint contient un dict JSON décrivant l'arête manquante
                edge_dict = self._extract_edge_from_hint(vr.hint or vr.reformulation_hint or "")
                implicit_edges = [edge_dict] if edge_dict else []

                fragment_id = vr.candidate.source_fragment

                print(
                    f"[Agent 3] STRUCTURAL_ERROR détecté pour fragment '{fragment_id}' "
                    f"(boucle structurelle #{self._structural_loop_count}) — renvoi à Agent 1"
                )

                self.send(AgentMessage(
                    sender    = self.AGENT_NAME,
                    recipient = "agent1",
                    msg_type  = MessageType.STRUCTURAL_UPDATE,
                    payload   = {
                        "reason":                vr.description or vr.explanation,
                        "affected_fragments":    [fragment_id],
                        "hint":                  vr.hint or vr.reformulation_hint,
                        "implicit_edges_to_add": implicit_edges,
                    },
                    loop_turn = loop_turn + 1,
                ))
                return   # Attendre la réponse enrichie d'Agent 1

            else:
                # Limite structurelle atteinte — continuer avec ce qu'on a
                print(
                    f"[Agent 3][WARN] MAX_STRUCTURAL_LOOPS ({self.MAX_STRUCTURAL_LOOPS}) atteint "
                    f"pour '{vr.candidate.source_activity}' → '{vr.candidate.target_activity}'. "
                    "Poursuite du pipeline avec le rapport tel quel."
                )
                break   # Sortir de la boucle for et continuer vers étape 2/3

        # ── Priorité 2 : REFORMULATE (limité pour éviter boucle infinie) ─────
        exhausted_reformulate: list[ValidationResult] = []
        for vr in report.reformulate:
            key = (
                vr.candidate.source_fragment,
                vr.candidate.source_activity,
                vr.candidate.target_activity,
            )
            if self._reformulate_sent_count.get(key, 0) >= self.MAX_REFORMULATE:
                exhausted_reformulate.append(vr)
                continue
            self._reformulate_sent_count[key] = self._reformulate_sent_count.get(key, 0) + 1
            fragment_id = vr.candidate.source_fragment
            print(
                f"[Agent 3] REFORMULATE pour fragment '{fragment_id}' : "
                f"'{vr.candidate.source_activity}' → '{vr.candidate.target_activity}' "
                f"(demande #{self._reformulate_sent_count[key]}/{self.MAX_REFORMULATE})"
            )
            self.send(AgentMessage(
                sender    = self.AGENT_NAME,
                recipient = "agent2",
                msg_type  = MessageType.REFORMULATE,
                payload   = {
                    "affected_fragment":  fragment_id,
                    "hint":               vr.reformulation_hint or vr.explanation,
                    "rejected_candidate": str(vr.candidate),
                },
                loop_turn = loop_turn,
            ))
            return   # Un seul REFORMULATE à la fois — attendre la réponse

        # Tous les REFORMULATE sont épuisés pour ce tour → rejeter ces candidats et continuer
        if exhausted_reformulate:
            print(
                f"[Agent 3] Limite MAX_REFORMULATE ({self.MAX_REFORMULATE}) atteinte pour "
                f"{len(exhausted_reformulate)} candidat(s) — abandon et poursuite vers Agent 4."
            )
            exhausted_set = {id(vr) for vr in exhausted_reformulate}
            new_results = []
            for r in report.results:
                if id(r) in exhausted_set:
                    new_results.append(ValidationResult(
                        candidate=r.candidate,
                        decision=ValidationDecision.REJECTED,
                        decision_level=r.decision_level,
                        reason=RejectionReason.LLM_REJECTED,
                        explanation="Abandon après limite de reformulation (MAX_REFORMULATE).",
                        description="Abandon après limite de reformulation (MAX_REFORMULATE).",
                    ))
                else:
                    new_results.append(r)
            report = ValidationReport(results=new_results)

        # ── Étape 3 : VALIDATION_DONE ───────────────────────────────
        accepted_count = len(report.accepted)
        print(
            f"[Agent 3] Validation terminée — {accepted_count} candidat(s) accepté(s). "
            "Émission VALIDATION_DONE vers Agent 4."
        )

        self.send(AgentMessage(
            sender    = self.AGENT_NAME,
            recipient = "agent4",
            msg_type  = MessageType.VALIDATION_DONE,
            payload   = {
                "validation_report": report,
                "enriched_graph":    enriched_graph,
            },
            loop_turn = loop_turn,
        ))

    def _extract_edge_from_hint(self, hint: str) -> Optional[dict]:
        """
        Tente d'extraire un dict d'arête depuis le hint du LLM juge.

        Le LLM juge est instruit de fournir dans son hint un objet JSON
        décrivant exactement l'arête manquante. Cette méthode le parse.
        Retourne None si le parsing échoue (fail-open : pas d'arête ajoutée).
        """
        if not hint:
            return None

        # Chercher un objet JSON dans la chaîne (peut être entouré de texte)
        try:
            # Cas 1 : le hint est directement du JSON valide
            return json.loads(hint)
        except (json.JSONDecodeError, ValueError):
            pass

        # Cas 2 : le hint contient un bloc JSON entre accolades
        try:
            start = hint.index("{")
            end   = hint.rindex("}") + 1
            return json.loads(hint[start:end])
        except (ValueError, json.JSONDecodeError):
            print(f"[Agent 3][WARN] Impossible d'extraire l'arête depuis le hint : {hint[:200]}")
            return None

    # ═════════════════════════════════════════
    #  LOGIQUE MÉTIER (inchangée)
    # ═════════════════════════════════════════

    # ─────────────────────────────────────────
    #  Point d'entrée standalone / pipeline
    # ─────────────────────────────────────────

    def validate(
        self,
        enriched_graph:   Optional[EnrichedGraph] = None,
        analysis_results: Optional[dict[str, ImplicitAnalysisResult]] = None,
    ) -> ValidationReport:
        """
        Valide tous les candidats de l'Agent 2.
        Retourne un ValidationReport complet.

        Peut être appelé :
          - En mode standalone : enriched_graph et analysis_results passés en argument
            ou stockés dans self depuis le constructeur.
          - Depuis _validate_and_route() : arguments explicites fournis par le message.
        """
        graph   = enriched_graph   or self.enriched_graph
        results = analysis_results or self.analysis_results

        if graph is None:
            raise ValueError("[Agent 3] Aucun enriched_graph disponible pour la validation.")

        # Reconstruire les index avec le graphe courant
        # (nécessaire en mode pipeline où le graphe peut changer entre les appels)
        self._explicit_connections = self._index_explicit_connections(graph)
        self._all_candidates       = self._collect_all_candidates(results)

        # Conserver une référence au graphe courant pour les méthodes internes
        self._current_graph = graph

        print("[Agent 3] Constraint Validator — démarrage de la validation")

        all_results = []

        for fragment_id, analysis in results.items():
            print(f"[Agent 3] Validation du fragment '{fragment_id}' "
                  f"({len(analysis.candidates)} candidats)...")

            for candidate in analysis.candidates:

                # ── Niveau 1 : filtre déterministe ──
                det_result = self._deterministic_filter(candidate)

                if det_result is not None:
                    all_results.append(det_result)
                    icon = "❌" if det_result.decision == ValidationDecision.REJECTED else "🔄"
                    print(f"  {icon} [{det_result.reason.value}] "
                          f"{candidate.source_activity} → {candidate.target_activity}")
                    continue

                # ── Niveau 2 : LLM juge ──
                llm_result = self._llm_judge(candidate, graph)
                all_results.append(llm_result)
                icon = {
                    "accepted":        "✅",
                    "rejected":        "❌",
                    "reformulate":     "🔄",
                    "structural_error": "🏗️",
                }.get(llm_result.decision.value, "?")
                print(f"  {icon} [LLM] {candidate.source_activity} → {candidate.target_activity} "
                      f"→ {llm_result.decision.value.upper()}")

        report = ValidationReport(results=all_results)

        print(f"\n[Agent 3] Validation terminée :")
        print(f"  ✅ Acceptés          : {len(report.accepted)}")
        print(f"  ❌ Rejetés           : {len(report.rejected)} "
              f"(dont {report.deterministic_rejections} déterministes, "
              f"{report.llm_rejections} LLM)")
        print(f"  🔄 Reformuler        : {len(report.reformulate)}")
        print(f"  🏗️  Erreurs struct.  : {len(report.structural_errors)}")

        return report

    # ─────────────────────────────────────────
    #  Niveau 1 — Filtre déterministe
    # ─────────────────────────────────────────

    def _deterministic_filter(
        self, candidate: CandidateDependency
    ) -> Optional[ValidationResult]:
        """
        Applique les règles déterministes dans l'ordre de criticité.
        Retourne un ValidationResult si rejeté/reformulé, None si le candidat passe.
        """

        # Règle 1 — Auto-référence
        if candidate.source_activity == candidate.target_activity:
            return ValidationResult(
                candidate=candidate,
                decision=ValidationDecision.REJECTED,
                decision_level="deterministic",
                reason=RejectionReason.SELF_REFERENCE,
                explanation=(
                    f"La source et la cible sont identiques : "
                    f"'{candidate.source_activity}'. Une activité ne peut pas "
                    f"dépendre d'elle-même."
                ),
            )

        # Règle 2 — Confiance trop faible
        if candidate.confidence < self.min_confidence:
            return ValidationResult(
                candidate=candidate,
                decision=ValidationDecision.REJECTED,
                decision_level="deterministic",
                reason=RejectionReason.LOW_CONFIDENCE,
                explanation=(
                    f"Confiance {candidate.confidence:.0%} inférieure au seuil "
                    f"minimum {self.min_confidence:.0%}. "
                    f"Dépendance trop incertaine pour être intégrée."
                ),
            )

        # Règle 3 — Doublon d'une connexion explicite du BPMN
        explicit_key = (candidate.source_activity, candidate.target_activity)
        if explicit_key in self._explicit_connections:
            return ValidationResult(
                candidate=candidate,
                decision=ValidationDecision.REJECTED,
                decision_level="deterministic",
                reason=RejectionReason.DUPLICATE_EXPLICIT,
                explanation=(
                    f"La connexion '{candidate.source_activity}' → "
                    f"'{candidate.target_activity}' existe déjà explicitement "
                    f"dans le BPMN (type: "
                    f"{self._explicit_connections[explicit_key]}). "
                    f"Pas de valeur ajoutée à la dupliquer."
                ),
            )

        # Règle 4 — Doublon entre fragments (même dep détectée par 2 agents)
        duplicate = self._find_duplicate_across_fragments(candidate)
        if duplicate:
            return ValidationResult(
                candidate=candidate,
                decision=ValidationDecision.REFORMULATE,
                decision_level="deterministic",
                reason=RejectionReason.DUPLICATE_CANDIDATE,
                explanation=(
                    f"Dépendance '{candidate.source_activity}' → "
                    f"'{candidate.target_activity}' ({candidate.dep_type.value}) "
                    f"déjà proposée par le fragment '{duplicate.source_fragment}'. "
                    f"Consolider les deux propositions."
                ),
                reformulation_hint=(
                    f"Fusionner avec le candidat du fragment "
                    f"'{duplicate.source_fragment}' en précisant "
                    f"quelle politique ODRL s'applique globalement."
                ),
            )

        # Règle 5 — Violation de gateway XOR
        gw_violation = self._check_gateway_violation(candidate)
        if gw_violation:
            return ValidationResult(
                candidate=candidate,
                decision=ValidationDecision.REJECTED,
                decision_level="deterministic",
                reason=RejectionReason.VIOLATES_GATEWAY,
                explanation=gw_violation,
            )

        # Passe le filtre déterministe → soumettre au LLM
        return None

    # ─────────────────────────────────────────
    #  Niveau 2 — LLM juge
    # ─────────────────────────────────────────

    def _llm_judge(
        self,
        candidate:      CandidateDependency,
        enriched_graph: Optional[EnrichedGraph] = None,
    ) -> ValidationResult:
        """
        Soumet le candidat au LLM pour une décision sémantique.
        Le LLM reçoit le contexte complet et statue sur la cohérence,
        y compris la détection d'erreurs structurelles (STRUCTURAL_ERROR).
        """
        graph  = enriched_graph or getattr(self, "_current_graph", None) or self.enriched_graph
        prompt = self._build_judge_prompt(candidate, graph)

        try:
            raw_response = self._call_llm(prompt)
            decision, explanation, hint = self._parse_judge_response(raw_response)

            return ValidationResult(
                candidate=candidate,
                decision=decision,
                decision_level="llm",
                reason=RejectionReason.LLM_REJECTED
                       if decision == ValidationDecision.REJECTED else None,
                explanation=explanation,
                description=explanation,
                llm_raw_response=raw_response,
                reformulation_hint=hint,
                hint=hint,
            )

        except Exception as e:
            # En cas d'erreur LLM → accepter prudemment avec avertissement
            print(f"[Agent 3][WARN] Erreur LLM pour candidat "
                  f"'{candidate.source_activity}' → '{candidate.target_activity}': {e}")
            return ValidationResult(
                candidate=candidate,
                decision=ValidationDecision.ACCEPTED,
                decision_level="llm",
                explanation=f"Accepté prudemment (erreur LLM : {e})",
                description=f"Accepté prudemment (erreur LLM : {e})",
            )

    def _build_judge_prompt(
        self,
        candidate:      CandidateDependency,
        enriched_graph: Optional[EnrichedGraph] = None,
    ) -> str:
        """
        Construit le prompt pour le LLM juge.

        Le LLM reçoit :
            - Le candidat à évaluer
            - Les connexions explicites du BPMN (pour contexte)
            - Les B2P policies applicables aux activités concernées
            - Les patterns structurels détectés par l'Agent 1
            - Les instructions pour détecter les STRUCTURAL_ERROR
              (dépendance inter-fragment sans ConnectionInfo correspondante)
        """
        graph = enriched_graph or getattr(self, "_current_graph", None) or self.enriched_graph

        # B2P policies des activités impliquées
        source_mapping = next(
            (m for m in graph.b2p_mappings.values()
             if m.activity_name == candidate.source_activity), None
        )
        target_mapping = next(
            (m for m in graph.b2p_mappings.values()
             if m.activity_name == candidate.target_activity), None
        )

        b2p_context = {
            "source_activity_policies": {
                "activity": candidate.source_activity,
                "policy_ids": source_mapping.b2p_policy_ids if source_mapping else [],
                "rule_types": source_mapping.rule_types if source_mapping else [],
            },
            "target_activity_policies": {
                "activity": candidate.target_activity,
                "policy_ids": target_mapping.b2p_policy_ids if target_mapping else [],
                "rule_types": target_mapping.rule_types if target_mapping else [],
            },
        }

        # Connexions explicites du BPMN autour des activités concernées
        relevant_connections = [
            {
                "from": c.from_activity,
                "to": c.to_activity,
                "type": c.connection_type,
                "condition": c.condition,
                "inter_fragment": c.is_inter,
            }
            for c in graph.connections
            if (c.from_activity in (candidate.source_activity, candidate.target_activity)
                or c.to_activity in (candidate.source_activity, candidate.target_activity))
        ]

        # Patterns structurels concernant ces activités
        relevant_patterns = [
            {
                "type": p.pattern_type,
                "gateway": p.gateway_name,
                "description": p.description,
            }
            for p in graph.patterns
            if (candidate.source_activity in str(p.involved_nodes)
                or candidate.target_activity in str(p.involved_nodes))
        ]

        # Inter-fragment : connexions existantes entre les deux fragments
        inter_fragment_connections = []
        if candidate.is_inter:
            inter_fragment_connections = [
                {
                    "from": c.from_activity,
                    "to": c.to_activity,
                    "from_fragment": c.from_fragment,
                    "to_fragment": c.to_fragment,
                    "type": c.connection_type,
                }
                for c in graph.connections
                if c.is_inter
                and c.from_fragment == candidate.source_fragment
                and c.to_fragment   == candidate.target_fragment
            ]

        prompt = f"""You are a strict validator of business process dependency policies.

Your role is to judge whether a proposed implicit dependency is:
- SEMANTICALLY COHERENT with the existing process structure
- NON-CONTRADICTORY with existing ODRL policies
- GENUINELY ADDING VALUE (not redundant, not harmful)
- STRUCTURALLY SUPPORTABLE by the existing graph connections

═══════════════════════════════════════════
CANDIDATE DEPENDENCY TO EVALUATE:
═══════════════════════════════════════════
Source activity  : {candidate.source_activity}
Target activity  : {candidate.target_activity}
Source fragment  : {candidate.source_fragment}
Target fragment  : {candidate.target_fragment}
Dependency type  : {candidate.dep_type.value}
Confidence       : {candidate.confidence:.0%}
Inter-fragment   : {candidate.is_inter}
Suggested rule   : {candidate.suggested_odrl_rule or 'not specified'}
Justification    : {candidate.justification}

═══════════════════════════════════════════
EXISTING BPMN CONNECTIONS (context):
═══════════════════════════════════════════
{json.dumps(relevant_connections, indent=2)}

═══════════════════════════════════════════
INTER-FRAGMENT CONNECTIONS (between source and target fragments):
═══════════════════════════════════════════
{json.dumps(inter_fragment_connections, indent=2) if candidate.is_inter else "N/A — intra-fragment dependency"}

═══════════════════════════════════════════
STRUCTURAL PATTERNS (context):
═══════════════════════════════════════════
{json.dumps(relevant_patterns, indent=2)}

═══════════════════════════════════════════
EXISTING B2P POLICIES (context):
═══════════════════════════════════════════
{json.dumps(b2p_context, indent=2)}

═══════════════════════════════════════════
YOUR DECISION — STRICT JSON:
═══════════════════════════════════════════

Respond ONLY with this exact JSON structure:

{{
  "decision": "accepted" | "rejected" | "reformulate" | "structural_error",
  "explanation": "1-2 sentences explaining your decision",
  "reformulation_hint": "only if decision is reformulate or structural_error — what should be changed or what edge is missing",
  "coherence_score": 0.0 to 1.0
}}

DECISION CRITERIA:
- "accepted"         : dependency is semantically valid, non-contradictory, adds value
- "rejected"         : dependency is semantically wrong, contradictory, or harmful
- "reformulate"      : dependency has merit but needs adjustment (scope, type, or wording)
- "structural_error" : dependency involves two activities from DIFFERENT fragments AND
                       no inter-fragment connection exists between them in the BPMN graph
                       (the graph is incomplete — not the candidate's fault)

If you respond with "structural_error", provide in "reformulation_hint" the exact edge
that should be added to the graph, as a JSON object with this structure:
{{
  "source_activity": "exact name of source activity",
  "target_activity": "exact name of target activity",
  "from_fragment":   "fragment_id of source",
  "to_fragment":     "fragment_id of target",
  "dep_type":        "temporal|data|role|compliance|conflict",
  "is_inter":        true
}}

Be strict. A dependency that is merely plausible but not clearly justified should be "reformulate".
A dependency that contradicts an XOR gateway or an existing prohibition should be "rejected".
A valid inter-fragment dependency with NO documented connection in the graph should be "structural_error".
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        model = self._deployment if self._use_azure else self.MODEL
        kwargs = {
            "model": model,
            "temperature": self.TEMPERATURE,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict business process policy validator. "
                        "You always respond with valid JSON. "
                        "You are conservative — when in doubt, you ask for reformulation. "
                        "You detect structural_error when an inter-fragment dependency "
                        "has no supporting connection in the BPMN graph."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        if not self._use_azure:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _parse_judge_response(
        self, raw_response: str
    ) -> tuple[ValidationDecision, str, Optional[str]]:
        """Parse la réponse du LLM juge, incluant le nouveau cas structural_error."""
        parsed = json.loads(raw_response)

        decision_str = parsed.get("decision", "reformulate").lower()
        explanation  = parsed.get("explanation", "")
        hint         = parsed.get("reformulation_hint")

        decision_map = {
            "accepted":        ValidationDecision.ACCEPTED,
            "rejected":        ValidationDecision.REJECTED,
            "reformulate":     ValidationDecision.REFORMULATE,
            "structural_error": ValidationDecision.STRUCTURAL_ERROR,
        }
        decision = decision_map.get(decision_str, ValidationDecision.REFORMULATE)

        return decision, explanation, hint

    # ─────────────────────────────────────────
    #  Helpers — index et vérifications
    # ─────────────────────────────────────────

    def _index_explicit_connections(
        self, enriched_graph: Optional[EnrichedGraph] = None
    ) -> dict[tuple, str]:
        """
        Construit un index (source, target) → type
        de toutes les connexions explicites du BPMN.
        """
        graph = enriched_graph or self.enriched_graph
        if graph is None:
            return {}
        index = {}
        for conn in graph.connections:
            key = (conn.from_activity, conn.to_activity)
            index[key] = conn.connection_type
        return index

    def _collect_all_candidates(
        self, analysis_results: Optional[dict] = None
    ) -> list[CandidateDependency]:
        """Collecte tous les candidats de tous les fragments."""
        results = analysis_results or self.analysis_results
        all_candidates = []
        for result in results.values():
            all_candidates.extend(result.candidates)
        return all_candidates

    def _find_duplicate_across_fragments(
        self, candidate: CandidateDependency
    ) -> Optional[CandidateDependency]:
        """
        Cherche si le même candidat (même source, cible, type)
        a été proposé par un autre fragment.
        """
        for other in self._all_candidates:
            if other is candidate:
                continue
            if (other.source_activity == candidate.source_activity
                    and other.target_activity == candidate.target_activity
                    and other.dep_type == candidate.dep_type
                    and other.source_fragment != candidate.source_fragment):
                return other
        return None

    def _check_gateway_violation(
        self, candidate: CandidateDependency
    ) -> Optional[str]:
        """
        Vérifie si le candidat viole la logique d'un gateway XOR.

        Violation XOR : deux activités sur des branches exclusives
        ne peuvent pas avoir de dépendance directe entre elles
        (elles ne s'exécutent jamais ensemble).
        """
        graph = getattr(self, "_current_graph", None) or self.enriched_graph
        if graph is None:
            return None

        for pattern in graph.patterns:
            if pattern.pattern_type != "fork_xor":
                continue

            g = graph.graph
            gw_node = g.get_node(pattern.gateway_id)
            if not gw_node:
                continue

            xor_branches = [
                g.get_node(e.target).name
                for e in g.out_edges(pattern.gateway_id)
                if g.get_node(e.target)
            ]

            src_in = candidate.source_activity in xor_branches
            tgt_in = candidate.target_activity in xor_branches

            if src_in and tgt_in:
                src_idx = xor_branches.index(candidate.source_activity) if candidate.source_activity in xor_branches else -1
                tgt_idx = xor_branches.index(candidate.target_activity) if candidate.target_activity in xor_branches else -1

                if src_idx != tgt_idx and src_idx != -1 and tgt_idx != -1:
                    return (
                        f"Violation XOR : '{candidate.source_activity}' et "
                        f"'{candidate.target_activity}' sont sur des branches "
                        f"exclusives du gateway '{pattern.gateway_name}'. "
                        f"Ces activités ne s'exécutent jamais simultanément — "
                        f"une dépendance entre elles est invalide."
                    )
        return None