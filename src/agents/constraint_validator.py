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

Sortie :
    ValidationResult par candidat :
        - ACCEPTED   : dépendance validée, intégrable au graphe
        - REJECTED   : dépendance rejetée avec motif
        - REFORMULATE: dépendance à reformuler (retour à Agent 2)
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from openai import OpenAI, AzureOpenAI

from .structural_analyzer import EnrichedGraph
from .implicit_dependency_detector import CandidateDependency, ImplicitAnalysisResult


# ─────────────────────────────────────────────
#  Décision de validation
# ─────────────────────────────────────────────

class ValidationDecision(Enum):
    ACCEPTED    = "accepted"
    REJECTED    = "rejected"
    REFORMULATE = "reformulate"


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
    candidate:         CandidateDependency
    decision:          ValidationDecision
    decision_level:    str                    # "deterministic" ou "llm"
    reason:            Optional[RejectionReason] = None
    explanation:       str = ""
    llm_raw_response:  Optional[str] = None   # Pour audit
    reformulation_hint: Optional[str] = None  # Si REFORMULATE

    @property
    def is_accepted(self) -> bool:
        return self.decision == ValidationDecision.ACCEPTED

    def __repr__(self):
        icon = {"accepted": "✅", "rejected": "❌", "reformulate": "🔄"}
        d = self.decision.value
        return (
            f"{icon.get(d,'?')} [{d.upper():12s}] "
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
    deterministic_rejections: int = 0
    llm_rejections:    int = 0
    llm_acceptances:   int = 0

    def __post_init__(self):
        self.accepted    = [r for r in self.results if r.is_accepted]
        self.rejected    = [r for r in self.results
                           if r.decision == ValidationDecision.REJECTED]
        self.reformulate = [r for r in self.results
                           if r.decision == ValidationDecision.REFORMULATE]
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

    Reçoit :
        - enriched_graph   : EnrichedGraph de l'Agent 1
        - analysis_results : dict[fragment_id → ImplicitAnalysisResult] de l'Agent 2

    Produit :
        - ValidationReport : décision sur chaque candidat

    Deux niveaux de décision :
        1. Filtre déterministe — règles objectives, sans LLM
        2. LLM juge           — raisonnement sémantique sur les cas ambigus
    """

    MODEL       = "gpt-4o"
    TEMPERATURE = 0.1   # Très bas — on veut des décisions stables et reproductibles

    def __init__(
        self,
        enriched_graph: EnrichedGraph,
        analysis_results: dict[str, ImplicitAnalysisResult],
        api_key: Optional[str] = None,
        min_confidence: float = 0.70,
        *,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
    ):
        self.enriched_graph   = enriched_graph
        self.analysis_results = analysis_results
        self.min_confidence   = min_confidence
        self._use_azure = False
        self._deployment: Optional[str] = None

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

        # Pré-construire les index pour les vérifications déterministes
        self._explicit_connections = self._index_explicit_connections()
        self._all_candidates       = self._collect_all_candidates()

    # ─────────────────────────────────────────
    #  Point d'entrée
    # ─────────────────────────────────────────

    def validate(self) -> ValidationReport:
        """
        Valide tous les candidats de l'Agent 2.
        Retourne un ValidationReport complet.
        """
        print("[Agent 3] Constraint Validator — démarrage de la validation")

        all_results = []

        for fragment_id, analysis in self.analysis_results.items():
            print(f"[Agent 3] Validation du fragment '{fragment_id}' "
                  f"({len(analysis.candidates)} candidats)...")

            for candidate in analysis.candidates:

                # ── Niveau 1 : filtre déterministe ──
                det_result = self._deterministic_filter(candidate)

                if det_result is not None:
                    # Rejeté ou reformulé par les règles
                    all_results.append(det_result)
                    icon = "❌" if det_result.decision == ValidationDecision.REJECTED else "🔄"
                    print(f"  {icon} [{det_result.reason.value}] "
                          f"{candidate.source_activity} → {candidate.target_activity}")
                    continue

                # ── Niveau 2 : LLM juge ──
                llm_result = self._llm_judge(candidate)
                all_results.append(llm_result)
                icon = "✅" if llm_result.is_accepted else (
                    "🔄" if llm_result.decision == ValidationDecision.REFORMULATE else "❌"
                )
                print(f"  {icon} [LLM] {candidate.source_activity} → {candidate.target_activity} "
                      f"→ {llm_result.decision.value.upper()}")

        report = ValidationReport(results=all_results)

        print(f"\n[Agent 3] Validation terminée :")
        print(f"  ✅ Acceptés    : {len(report.accepted)}")
        print(f"  ❌ Rejetés     : {len(report.rejected)} "
              f"(dont {report.deterministic_rejections} déterministes, "
              f"{report.llm_rejections} LLM)")
        print(f"  🔄 Reformuler  : {len(report.reformulate)}")

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

    def _llm_judge(self, candidate: CandidateDependency) -> ValidationResult:
        """
        Soumet le candidat au LLM pour une décision sémantique.
        Le LLM reçoit le contexte complet et statue sur la cohérence.
        """
        prompt = self._build_judge_prompt(candidate)

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
                llm_raw_response=raw_response,
                reformulation_hint=hint,
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
            )

    def _build_judge_prompt(self, candidate: CandidateDependency) -> str:
        """
        Construit le prompt pour le LLM juge.

        Le LLM reçoit :
            - Le candidat à évaluer
            - Les connexions explicites du BPMN (pour contexte)
            - Les B2P policies applicables aux activités concernées
            - Les patterns structurels détectés par l'Agent 1
        """

        # B2P policies des activités impliquées
        source_mapping = next(
            (m for m in self.enriched_graph.b2p_mappings.values()
             if m.activity_name == candidate.source_activity), None
        )
        target_mapping = next(
            (m for m in self.enriched_graph.b2p_mappings.values()
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
            for c in self.enriched_graph.connections
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
            for p in self.enriched_graph.patterns
            if (candidate.source_activity in str(p.involved_nodes)
                or candidate.target_activity in str(p.involved_nodes))
        ]

        prompt = f"""You are a strict validator of business process dependency policies.

Your role is to judge whether a proposed implicit dependency is:
- SEMANTICALLY COHERENT with the existing process structure
- NON-CONTRADICTORY with existing ODRL policies
- GENUINELY ADDING VALUE (not redundant, not harmful)

═══════════════════════════════════════════
CANDIDATE DEPENDENCY TO EVALUATE:
═══════════════════════════════════════════
Source activity  : {candidate.source_activity}
Target activity  : {candidate.target_activity}
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
  "decision": "accepted" | "rejected" | "reformulate",
  "explanation": "1-2 sentences explaining your decision",
  "reformulation_hint": "only if decision is reformulate — what should be changed",
  "coherence_score": 0.0 to 1.0
}}

DECISION CRITERIA:
- "accepted"    : dependency is semantically valid, non-contradictory, adds value
- "rejected"    : dependency is semantically wrong, contradictory, or harmful
- "reformulate" : dependency has merit but needs adjustment (scope, type, or wording)

Be strict. A dependency that is merely plausible but not clearly justified should be "reformulate".
A dependency that contradicts an XOR gateway or an existing prohibition should be "rejected".
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
                        "You are conservative — when in doubt, you ask for reformulation."
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
        """Parse la réponse du LLM juge."""
        parsed = json.loads(raw_response)

        decision_str = parsed.get("decision", "reformulate").lower()
        explanation  = parsed.get("explanation", "")
        hint         = parsed.get("reformulation_hint")

        decision_map = {
            "accepted":    ValidationDecision.ACCEPTED,
            "rejected":    ValidationDecision.REJECTED,
            "reformulate": ValidationDecision.REFORMULATE,
        }
        decision = decision_map.get(decision_str, ValidationDecision.REFORMULATE)

        return decision, explanation, hint

    # ─────────────────────────────────────────
    #  Helpers — index et vérifications
    # ─────────────────────────────────────────

    def _index_explicit_connections(self) -> dict[tuple, str]:
        """
        Construit un index (source, target) → type
        de toutes les connexions explicites du BPMN.
        """
        index = {}
        for conn in self.enriched_graph.connections:
            key = (conn.from_activity, conn.to_activity)
            index[key] = conn.connection_type
        return index

    def _collect_all_candidates(self) -> list[CandidateDependency]:
        """Collecte tous les candidats de tous les fragments."""
        all_candidates = []
        for result in self.analysis_results.values():
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
        # Trouver les patterns XOR qui impliquent ces activités
        for pattern in self.enriched_graph.patterns:
            if pattern.pattern_type != "fork_xor":
                continue

            # Récupérer les nœuds cibles de ce fork XOR
            graph = self.enriched_graph.graph
            gw_node = graph.get_node(pattern.gateway_id)
            if not gw_node:
                continue

            xor_branches = [
                graph.get_node(e.target).name
                for e in graph.out_edges(pattern.gateway_id)
                if graph.get_node(e.target)
            ]

            # Si source ET target sont dans des branches différentes du même XOR
            src_in = candidate.source_activity in xor_branches
            tgt_in = candidate.target_activity in xor_branches

            if src_in and tgt_in:
                # Vérifier qu'ils sont dans des branches différentes
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