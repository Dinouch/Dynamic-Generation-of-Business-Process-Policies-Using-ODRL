"""
implicit_dependency_detector.py — Agent 2 : Implicit Dependency Detector

Responsabilité unique :
    Détecter les dépendances implicites non représentées dans le BPMN,
    en utilisant le LLM comme raisonneur partiel.

Ce que cet agent NE fait PAS :
    - Il ne génère pas de policies ODRL
    - Il ne modifie pas le graphe
    - Il ne valide pas ses propres propositions

Ce qu'il produit :
    Une liste de CandidateDependency — chacune avec
    une source, une cible, un type, une confiance, et une justification textuelle.

Ces candidats seront soumis à l'Agent 3 (Constraint Validator) pour acceptation/rejet.

Design du prompt :
    Le LLM reçoit un contexte structuré (pas de texte brut)
    et doit répondre UNIQUEMENT en JSON valide.
    Aucune liberté de format — on contrôle la sortie.
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
    FragmentContext,
    MessageType,
    ReanalysisLimitError,
)


# ─────────────────────────────────────────────
#  Types de dépendances implicites détectables
# ─────────────────────────────────────────────

class ImplicitDepType(Enum):
    """
    Types de dépendances implicites que le LLM peut proposer.
    Chaque type a une sémantique ODRL différente → impact sur la génération de policies.
    """
    TEMPORAL = "temporal"      # A doit finir avant B (contrainte de temps)
    RESOURCE = "resource"     # A et B partagent une ressource
    DATA = "data"             # A produit une donnée que B consomme
    ROLE = "role"             # A et B requièrent le même rôle/acteur
    COMPLIANCE = "compliance" # A déclenche une obligation réglementaire sur B
    CONFLICT = "conflict"     # A et B sont mutuellement exclusifs (au-delà du XOR)


# ─────────────────────────────────────────────
#  Structure de sortie de l'agent
# ─────────────────────────────────────────────

@dataclass
class CandidateDependency:
    """
    Une dépendance implicite proposée par le LLM.

    'Candidate' car elle n'est pas encore validée —
    c'est l'Agent 3 qui décidera de l'accepter ou la rejeter.
    """
    source_activity: str
    target_activity: str
    dep_type: ImplicitDepType
    confidence: float              # Entre 0.0 et 1.0
    justification: str             # Raisonnement du LLM
    source_fragment: str
    target_fragment: str
    is_inter: bool                 # Inter-fragment ?
    suggested_odrl_rule: Optional[str] = None  # Suggestion de règle ODRL
    context_used: str = "local"    # "local" ou "global"

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.75

    def __repr__(self):
        inter = " [INTER]" if self.is_inter else ""
        return (
            f"Candidate({self.dep_type.value}: "
            f"{self.source_activity} → {self.target_activity}, "
            f"conf={self.confidence:.2f}{inter})"
        )


@dataclass
class ImplicitAnalysisResult:
    """
    Résultat complet de l'Agent 2 pour un fragment donné.
    """
    fragment_id: str
    context_type: str                        # "local" ou "global"
    candidates: list[CandidateDependency]
    llm_raw_response: str                    # Réponse brute du LLM (pour audit)
    prompt_used: str                         # Prompt envoyé (pour traçabilité)
    high_confidence: list[CandidateDependency] = field(default_factory=list)
    low_confidence: list[CandidateDependency] = field(default_factory=list)

    def __post_init__(self):
        self.high_confidence = [c for c in self.candidates if c.is_high_confidence]
        self.low_confidence = [c for c in self.candidates if not c.is_high_confidence]


# ─────────────────────────────────────────────
#  Agent 2 — Implicit Dependency Detector
# ─────────────────────────────────────────────

class ImplicitDependencyDetector:
    """
    Agent 2 du pipeline multi-agent.

    Reçoit :
        - enriched_graph : EnrichedGraph produit par l'Agent 1
        - context_mode   : "local" (CL) ou "global" (CG)
        - api_key        : clé OpenAI

    Produit :
        - dict[fragment_id → ImplicitAnalysisResult]

    Architecture du LLM call :
        1. Construction d'un prompt structuré depuis le contexte du fragment
        2. Appel OpenAI avec réponse forcée en JSON
        3. Parsing et validation de la réponse
        4. Construction des CandidateDependency

    Couche multi-agent :
        - receive()  : accepte GRAPH_READY et REFORMULATE
        - send()     : émet CANDIDATES_READY vers Agent 3
        - Boucle courte (max MAX_REFORMULATE tours) avec Agent 3
        - Boucle structurelle (reset du compteur) depuis Agent 1
    """

    MODEL = "gpt-4o"
    TEMPERATURE = 0.2

    AGENT_NAME      = "agent2"
    MAX_REFORMULATE = 3   # max de tours dans la boucle courte Agent2↔Agent3

    def __init__(
        self,
        context_mode: str = "local",
        api_key: Optional[str] = None,
        *,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        # Rétrocompatibilité : accepte enriched_graph en paramètre positionnel
        # pour le mode standalone (sans pipeline).
        enriched_graph: Optional[EnrichedGraph] = None,
    ):
        self.enriched_graph = enriched_graph   # peut rester None en mode pipeline
        self.context_mode = context_mode
        self._use_azure = False
        self._deployment: Optional[str] = None

        # ── Couche multi-agent ──────────────────────────────────────
        self._on_send:           Optional[Callable[[AgentMessage], None]] = None
        self._reformulate_count: int = 0
        self._last_graph:        Optional[EnrichedGraph] = None
        self._last_results:      Optional[dict] = None   # dict[frag_id → ImplicitAnalysisResult]

        # Azure OpenAI (prioritaire si endpoint + clé présents)
        key_azure = api_key or os.environ.get("AZURE_OPENAI_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
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
            # OpenAI (api.openai.com)
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "Clé manquante. Définis OPENAI_API_KEY (OpenAI) ou "
                    "AZURE_OPENAI_KEY + AZURE_OPENAI_ENDPOINT (Azure)."
                )
            self.client = OpenAI(api_key=key)

    # ═════════════════════════════════════════
    #  COUCHE MULTI-AGENT
    # ═════════════════════════════════════════

    def register_send_callback(self, fn: Callable[[AgentMessage], None]) -> None:
        """Enregistre le callback d'envoi (typiquement agent3.receive)."""
        self._on_send = fn

    def send(self, msg: AgentMessage) -> None:
        """Émet un message via le callback enregistré."""
        print(f"[Agent 2] ► SEND {msg}")
        if self._on_send:
            self._on_send(msg)
        else:
            print(f"[Agent 2][WARN] Aucun callback — message {msg.msg_type.value} non transmis.")

    def receive(self, msg: AgentMessage) -> None:
        """
        Point d'entrée des messages entrants.

        Messages acceptés :
          - GRAPH_READY    : nouveau graphe depuis Agent 1 (première analyse ou graphe enrichi)
          - REFORMULATE    : demande de reformulation depuis Agent 3 (même graphe, hint)

        Tout autre type est ignoré avec un warning.
        """
        print(f"[Agent 2] ◄ RECEIVE {msg}")
        if msg.msg_type == MessageType.GRAPH_READY:
            self._handle_graph_ready(msg)
        elif msg.msg_type == MessageType.REFORMULATE:
            self._handle_reformulate(msg)
        else:
            print(f"[Agent 2][WARN] Message '{msg.msg_type.value}' non géré — ignoré.")

    # ─────────────────────────────────────────
    #  Handlers internes
    # ─────────────────────────────────────────

    def _handle_graph_ready(self, msg: AgentMessage) -> None:
        """
        Traite un message GRAPH_READY depuis Agent 1.

        Deux cas distincts :
          - reanalysis=False : première analyse complète du graphe
          - reanalysis=True  : graphe enrichi par Agent 1 (arêtes implicites ajoutées)
            → reset du compteur de reformulation car c'est un nouveau problème,
              pas une reformulation du même.
        """
        enriched_graph = msg.payload["enriched_graph"]
        self._last_graph = enriched_graph

        if msg.payload.get("reanalysis"):
            # Graphe enrichi — reset du compteur de reformulation
            self._reformulate_count = 0
            affected = msg.payload.get("affected_fragments") or list(
                enriched_graph.fragment_contexts.keys()
            )
            print(f"[Agent 2] Graphe enrichi reçu — re-détection sur {affected}")
            results = self._detect_on_fragments(enriched_graph, affected)
        else:
            # Première analyse — tous les fragments
            results = self.detect(enriched_graph)

        self._last_results = results
        self._emit_candidates_ready(results, enriched_graph, loop_turn=msg.loop_turn)

    def _handle_reformulate(self, msg: AgentMessage) -> None:
        """
        Traite une demande de reformulation depuis Agent 3.

        REFORMULATE signifie que le graphe est INTACT — Agent 3 a rejeté un
        candidat parce que le raisonnement d'Agent 2 était mauvais. On relit
        le même _last_graph et on propose un candidat différent pour le fragment
        concerné, en intégrant le hint dans le prompt LLM.

        Si MAX_REFORMULATE est atteint, on émet les derniers résultats en l'état
        plutôt que de bloquer le pipeline (fail-open).
        """
        if self._reformulate_count >= self.MAX_REFORMULATE:
            print(
                f"[Agent 2][WARN] Limite MAX_REFORMULATE ({self.MAX_REFORMULATE}) atteinte. "
                "Émission des derniers résultats en l'état."
            )
            self._emit_candidates_ready(
                self._last_results or {},
                self._last_graph,
                loop_turn=msg.loop_turn,
            )
            return

        fragment_id        = msg.payload["affected_fragment"]
        hint               = msg.payload.get("hint", "")
        rejected_candidate = msg.payload.get("rejected_candidate", "")

        print(
            f"[Agent 2] Reformulation #{self._reformulate_count + 1} "
            f"pour fragment '{fragment_id}'"
        )

        # Re-détecter sur ce fragment uniquement avec le hint (appel LLM réel).
        # Le graphe est LE MÊME — seul le raisonnement change.
        new_result = self._detect_single_fragment_with_hint(
            fragment_id        = fragment_id,
            hint               = hint,
            rejected_candidate = rejected_candidate,
        )

        print(
            f"[Agent 2] Reformulation effectuée — fragment '{fragment_id}' : "
            f"{len(new_result.candidates)} candidat(s) proposé(s)"
        )

        # Mettre à jour uniquement ce fragment dans les résultats
        if self._last_results is None:
            self._last_results = {}
        self._last_results[fragment_id] = new_result
        self._reformulate_count += 1

        self._emit_candidates_ready(
            self._last_results,
            self._last_graph,
            loop_turn=msg.loop_turn + 1,
        )

    def _emit_candidates_ready(
        self,
        results: dict,
        enriched_graph: Optional[EnrichedGraph],
        loop_turn: int,
    ) -> None:
        """Émet CANDIDATES_READY vers Agent 3."""
        self.send(AgentMessage(
            sender    = self.AGENT_NAME,
            recipient = "agent3",
            msg_type  = MessageType.CANDIDATES_READY,
            payload   = {
                "results":        results,
                "enriched_graph": enriched_graph,
            },
            loop_turn = loop_turn,
        ))

    # ─────────────────────────────────────────
    #  Helpers multi-fragment
    # ─────────────────────────────────────────

    def _detect_on_fragments(
        self,
        enriched_graph: EnrichedGraph,
        fragment_ids: list[str],
    ) -> dict[str, ImplicitAnalysisResult]:
        """
        Re-détecte uniquement sur les fragments listés.
        Fusionne avec _last_results pour les fragments non affectés,
        de façon à conserver les résultats déjà calculés.
        """
        contexts = (
            enriched_graph.global_contexts
            if self.context_mode == "global"
            else enriched_graph.fragment_contexts
        )

        merged = dict(self._last_results) if self._last_results else {}

        for fragment_id in fragment_ids:
            if fragment_id not in contexts:
                print(f"[Agent 2][WARN] Fragment '{fragment_id}' introuvable dans le graphe — ignoré.")
                continue

            context = contexts[fragment_id]
            print(f"[Agent 2] Re-analyse du fragment '{fragment_id}'...")
            result = self._analyze_fragment(fragment_id, context)
            merged[fragment_id] = result

            print(
                f"[Agent 2] Fragment '{fragment_id}' : "
                f"{len(result.candidates)} candidats détectés "
                f"({len(result.high_confidence)} haute confiance)"
            )

        return merged

    def _detect_single_fragment_with_hint(
        self,
        fragment_id: str,
        hint: str,
        rejected_candidate: str,
    ) -> ImplicitAnalysisResult:
        """
        Re-analyse un unique fragment en injectant un hint de reformulation
        dans le prompt. Le graphe utilisé est self._last_graph (inchangé).

        Le hint vient d'Agent 3 — il explique pourquoi le candidat précédent
        a été rejeté et guide le LLM vers une alternative valide.
        """
        if self._last_graph is None:
            raise RuntimeError(
                "[Agent 2] _last_graph est None — impossible de reformuler sans graphe."
            )

        contexts = (
            self._last_graph.global_contexts
            if self.context_mode == "global"
            else self._last_graph.fragment_contexts
        )

        if fragment_id not in contexts:
            raise ValueError(
                f"[Agent 2] Fragment '{fragment_id}' introuvable dans _last_graph."
            )

        context = contexts[fragment_id]
        prompt  = self._build_prompt_with_hint(fragment_id, context, hint, rejected_candidate)
        raw_response = self._call_llm(prompt)
        candidates   = self._parse_response(raw_response, context)

        return ImplicitAnalysisResult(
            fragment_id      = fragment_id,
            context_type     = self.context_mode,
            candidates       = candidates,
            llm_raw_response = raw_response,
            prompt_used      = prompt,
        )

    # ═════════════════════════════════════════
    #  LOGIQUE MÉTIER (inchangée)
    # ═════════════════════════════════════════

    # ─────────────────────────────────────────
    #  Point d'entrée standalone
    # ─────────────────────────────────────────

    def detect(
        self,
        enriched_graph: Optional[EnrichedGraph] = None,
    ) -> dict[str, ImplicitAnalysisResult]:
        """
        Lance la détection de dépendances implicites pour tous les fragments.

        Peut être appelé en mode standalone (enriched_graph passé en argument
        ou stocké en self.enriched_graph) ou depuis le pipeline multi-agent
        via _handle_graph_ready().

        Retourne un dict : fragment_id → ImplicitAnalysisResult
        """
        graph = enriched_graph or self.enriched_graph
        if graph is None:
            raise ValueError(
                "[Agent 2] Aucun enriched_graph fourni. "
                "Passez-le en argument ou via le constructeur."
            )

        print(f"[Agent 2] Implicit Dependency Detector — mode contexte : {self.context_mode}")

        results = {}

        contexts = (
            graph.global_contexts
            if self.context_mode == "global"
            else graph.fragment_contexts
        )

        for fragment_id, context in contexts.items():
            print(f"[Agent 2] Analyse du fragment '{fragment_id}'...")
            result = self._analyze_fragment(fragment_id, context)
            results[fragment_id] = result

            print(
                f"[Agent 2] Fragment '{fragment_id}' : "
                f"{len(result.candidates)} candidats détectés "
                f"({len(result.high_confidence)} haute confiance)"
            )

        return results

    # ─────────────────────────────────────────
    #  Analyse d'un fragment
    # ─────────────────────────────────────────

    def _analyze_fragment(
        self,
        fragment_id: str,
        context: FragmentContext,
    ) -> ImplicitAnalysisResult:
        """
        Analyse un fragment donné et retourne ses dépendances implicites candidates.
        """
        prompt = self._build_prompt(fragment_id, context)
        raw_response = self._call_llm(prompt)
        candidates = self._parse_response(raw_response, context)

        return ImplicitAnalysisResult(
            fragment_id=fragment_id,
            context_type=self.context_mode,
            candidates=candidates,
            llm_raw_response=raw_response,
            prompt_used=prompt,
        )

    # ─────────────────────────────────────────
    #  Construction du prompt
    # ─────────────────────────────────────────

    def _build_prompt(self, fragment_id: str, context: FragmentContext) -> str:
        """
        Construit un prompt structuré pour le LLM.
        """
        activities_str = json.dumps(context.activities, indent=2)

        connections_str = json.dumps([
            {
                "from": c.from_activity,
                "to": c.to_activity,
                "type": c.connection_type,
                "gateway": c.gateway_name,
                "condition": c.condition,
                "inter_fragment": c.is_inter,
            }
            for c in context.connections
        ], indent=2)

        b2p_str = json.dumps([
            {
                "activity": m.activity_name,
                "policy_ids": m.b2p_policy_ids,
                "rule_types": m.rule_types,
            }
            for m in context.b2p_mappings
            if m.b2p_policy_ids
        ], indent=2)

        upstream_str = json.dumps([
            {"from_activity": u.from_activity, "from_fragment": u.from_fragment}
            for u in context.upstream_deps
        ], indent=2)

        downstream_str = json.dumps([
            {"to_activity": d.to_activity, "to_fragment": d.to_fragment}
            for d in context.downstream_deps
        ], indent=2)

        global_section = ""
        if context.is_global and context.global_graph_summary:
            global_section = f"""
GLOBAL PROCESS CONTEXT (full view):
- Total fragments in process: {len(context.global_graph_summary.get('fragments', []))}
- All fragments: {context.global_graph_summary.get('fragments', [])}
- Total activities: {context.global_graph_summary.get('activities', 0)}
- Inter-fragment edges in full process: {context.global_graph_summary.get('inter_edges', 0)}
- Has cycles: {context.global_graph_summary.get('has_cycle', False)}

All inter-fragment connections in process:
{json.dumps([
    {
        "from": e.from_activity,
        "to": e.to_activity,
        "from_fragment": e.from_fragment,
        "to_fragment": e.to_fragment,
    }
    for e in (context.all_inter_edges or [])
], indent=2)}
"""

        prompt = f"""You are an expert in Business Process Management and ODRL policy generation.

Your task is to identify IMPLICIT dependencies between activities in a business process fragment.

IMPLICIT dependencies are dependencies that are NOT represented by BPMN flows or gateways,
but that exist due to business semantics, regulatory requirements, resource sharing, or data flow.

═══════════════════════════════════════════
FRAGMENT: {fragment_id}
═══════════════════════════════════════════

ACTIVITIES IN THIS FRAGMENT:
{activities_str}

EXPLICIT CONNECTIONS (already in BPMN — do NOT repeat these):
{connections_str}

UPSTREAM DEPENDENCIES (what feeds into this fragment):
{upstream_str}

DOWNSTREAM DEPENDENCIES (what this fragment feeds into):
{downstream_str}

EXISTING B2P POLICIES (ODRL rules already defined for activities):
{b2p_str}
{global_section}

═══════════════════════════════════════════
DEPENDENCY TYPES YOU CAN PROPOSE:
═══════════════════════════════════════════

- "temporal"   : Activity A must complete before B starts (time constraint)
- "resource"   : Activities A and B share the same resource (cannot run simultaneously)
- "data"       : Activity A produces data that B implicitly needs
- "role"       : Activities A and B require the same person/role
- "compliance" : Activity A triggers a regulatory obligation on B
- "conflict"   : Activities A and B are mutually exclusive beyond XOR logic

═══════════════════════════════════════════
ODRL RULE TYPES FOR YOUR SUGGESTIONS:
═══════════════════════════════════════════

- "permission"  : B is allowed to execute if A's condition is met
- "prohibition" : B is forbidden to execute if A's condition is not met
- "obligation"  : B MUST execute after A completes

═══════════════════════════════════════════
YOUR RESPONSE — STRICT JSON FORMAT:
═══════════════════════════════════════════

Respond ONLY with a valid JSON object with a key "dependencies" whose value is an array.
No explanation, no markdown, no preamble.
Each element of the array must follow this exact schema:

{{
  "dependencies": [
    {{
      "source_activity": "exact activity name from the list above",
      "target_activity": "exact activity name from the list above OR from adjacent fragments",
      "dep_type": "temporal|resource|data|role|compliance|conflict",
      "confidence": 0.0 to 1.0,
      "justification": "1-2 sentence business justification for this dependency",
      "suggested_odrl_rule": "permission|prohibition|obligation",
      "is_inter_fragment": true or false
    }}
  ]
}}

RULES:
- Only propose dependencies that are NOT already explicit in the BPMN connections above
- Confidence >= 0.75 means you are quite sure this dependency exists
- Confidence < 0.75 means it is a hypothesis worth investigating
- If no implicit dependencies exist, return {{ "dependencies": [] }}
- Maximum 5 candidates per fragment — prioritize the most meaningful ones
- Use EXACT activity names as they appear in the lists above
"""
        return prompt

    def _build_prompt_with_hint(
        self,
        fragment_id: str,
        context: FragmentContext,
        hint: str,
        rejected_candidate: str,
    ) -> str:
        """
        Variante du prompt de détection pour la boucle de reformulation.

        Injecte en tête du prompt le hint fourni par Agent 3 et le candidat
        rejeté, de façon à orienter le LLM vers une alternative valide sans
        répéter l'erreur précédente.

        Le reste du prompt (structure du fragment, connexions, etc.) est
        identique à _build_prompt() — seul le contexte de reformulation s'ajoute.
        """
        base_prompt = self._build_prompt(fragment_id, context)

        reformulation_header = f"""
═══════════════════════════════════════════
REFORMULATION HINT FROM VALIDATOR:
═══════════════════════════════════════════

{hint}

The following candidate was rejected and must NOT be reproduced:
{rejected_candidate}

Generate a different alternative. Do not propose the same source/target/dep_type combination.

═══════════════════════════════════════════

"""
        # Insérer le header juste avant la section du fragment
        insert_marker = f"═══════════════════════════════════════════\nFRAGMENT: {fragment_id}"
        return base_prompt.replace(insert_marker, reformulation_header + insert_marker, 1)

    # ─────────────────────────────────────────
    #  Appel LLM
    # ─────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """
        Appelle l'API (OpenAI ou Azure OpenAI) et retourne la réponse brute.
        """
        model = self._deployment if self._use_azure else self.MODEL
        kwargs = {
            "model": model,
            "temperature": self.TEMPERATURE,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise business process analyst. "
                        "You always respond with valid JSON only. "
                        "You never add explanations outside the JSON structure."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }
        if not self._use_azure:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    # ─────────────────────────────────────────
    #  Parsing de la réponse
    # ─────────────────────────────────────────

    def _parse_response(
        self,
        raw_response: str,
        context: FragmentContext,
    ) -> list[CandidateDependency]:
        """
        Parse la réponse JSON du LLM en CandidateDependency.
        """
        candidates = []

        try:
            parsed = json.loads(raw_response)

            if isinstance(parsed, dict):
                items = (
                    parsed.get("dependencies")
                    or parsed.get("candidates")
                    or parsed.get("implicit_dependencies")
                    or []
                )
            elif isinstance(parsed, list):
                items = parsed
            else:
                print(f"[Agent 2][WARN] Format de réponse inattendu : {type(parsed)}")
                return []

        except json.JSONDecodeError as e:
            print(f"[Agent 2][ERROR] Impossible de parser la réponse JSON : {e}")
            print(f"[Agent 2][ERROR] Réponse brute : {raw_response[:200]}")
            return []

        known_activities = set(context.activities)
        for dep in context.upstream_deps + context.downstream_deps:
            known_activities.add(dep.from_activity)
            known_activities.add(dep.to_activity)

        for item in items:
            try:
                candidate = self._validate_and_build_candidate(
                    item, context, known_activities
                )
                if candidate:
                    candidates.append(candidate)
            except Exception as e:
                print(f"[Agent 2][WARN] Candidat ignoré (erreur) : {e} — {item}")

        return candidates

    def _validate_and_build_candidate(
        self,
        item: dict,
        context: FragmentContext,
        known_activities: set,
    ) -> Optional[CandidateDependency]:
        """
        Valide un item JSON et construit un CandidateDependency.
        Retourne None si l'item est invalide.
        """
        source = item.get("source_activity", "").strip()
        target = item.get("target_activity", "").strip()
        dep_type_str = item.get("dep_type", "").strip()
        confidence = float(item.get("confidence", 0.0))
        justification = item.get("justification", "").strip()
        suggested_rule = item.get("suggested_odrl_rule", "").strip() or None
        is_inter = bool(item.get("is_inter_fragment", False))

        if not source or not target:
            print(f"[Agent 2][WARN] Source ou cible manquante : {item}")
            return None

        if source not in known_activities:
            print(f"[Agent 2][WARN] Activité source inconnue : '{source}' — ignorée")
            return None

        try:
            dep_type = ImplicitDepType(dep_type_str)
        except ValueError:
            print(f"[Agent 2][WARN] Type de dépendance inconnu : '{dep_type_str}' — ignoré")
            return None

        confidence = max(0.0, min(1.0, confidence))

        source_fragment = context.fragment_id
        target_fragment = context.fragment_id
        if is_inter:
            for dep in context.downstream_deps:
                if dep.to_activity == target:
                    target_fragment = dep.to_fragment
                    break
            for dep in context.upstream_deps:
                if dep.from_activity == source:
                    source_fragment = dep.from_fragment
                    break

        return CandidateDependency(
            source_activity=source,
            target_activity=target,
            dep_type=dep_type,
            confidence=confidence,
            justification=justification,
            source_fragment=source_fragment,
            target_fragment=target_fragment,
            is_inter=is_inter,
            suggested_odrl_rule=suggested_rule,
            context_used=self.context_mode,
        )