"""
structural_analyzer.py — Agent 1 : Structural Analyzer

Responsabilité unique :
    Lire le BPMN brut → construire le graphe formel G_exp → détecter les patterns structurels.

Cet agent NE génère PAS de policies.
Il prépare la vérité structurelle pour tous les agents suivants.

Sortie : EnrichedGraph contenant
    - Le graphe formel G (BPMNGraph)
    - Les patterns détectés (forks, joins, loops, sync points)
    - Le mapping B2P → activités
    - Les connexions typées entre activités
    - Le contexte global CG et local CL pour chaque fragment

─────────────────────────────────────────────────────────────────
  COUCHE MULTI-AGENT
─────────────────────────────────────────────────────────────────
  Cet agent expose également l'infrastructure de messagerie
  partagée par tous les agents du pipeline :

      MessageType, AgentMessage, ReanalysisLimitError

  Ces classes sont importées par tous les autres agents :
      from .structural_analyzer import AgentMessage, MessageType, ReanalysisLimitError

  Méthodes multi-agent de StructuralAnalyzer :
      register_send_callback(fn)  — connecte Agent 1 → Agent 2
      send(msg)                   — émet un AgentMessage
      receive(msg)                — accepte STRUCTURAL_UPDATE de Agent 3
      analyze_and_send()          — analyze() + émet GRAPH_READY

  Boucle structurelle (Agent 3 → Agent 1 → Agent 2) :
      Quand Agent 3 détecte un STRUCTURAL_ERROR (graphe incomplet),
      il envoie STRUCTURAL_UPDATE avec des arêtes implicites à ajouter.
      Agent 1 les injecte chirurgicalement dans _last_enriched_graph
      via _add_implicit_connections() — SANS re-exécuter analyze().

      Pourquoi pas de re-analyse complète :
          Tout le pipeline repose sur les noms d'activités comme
          identifiants stables (_name_to_id, _fragment_of, ConnectionInfo).
          Re-exécuter analyze() depuis le BPMN brut redonnerait exactement
          le même graphe sans les nouvelles arêtes. Et modifier les fragments
          avant désynchroniserait les mappings déjà en mémoire dans les autres
          agents. La seule opération sûre est d'ajouter des ConnectionInfo
          implicites au graphe existant.
"""

import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

# Import fragmenter à la demande dans analyze() pour éviter dépendance circulaire
# et garder l'agent utilisable sans fragmenter si fragments est fourni.

from graph import (
    BPMNGraph,
    ActivityNode,
    GatewayNode,
    EventNode,
    Edge,
    NodeType,
    GatewayType,
    EdgeType,
    DependencyType,
)


# ─────────────────────────────────────────────
#  Infrastructure de messagerie multi-agent
#  (importée par tous les agents du pipeline)
# ─────────────────────────────────────────────

class MessageType(Enum):
    """
    Types de messages échangés entre agents.

    Flux complet du pipeline :
        Agent 1 ──GRAPH_READY──────────► Agent 2
        Agent 2 ──CANDIDATES_READY─────► Agent 3
        Agent 3 ──REFORMULATE──────────► Agent 2   (boucle courte)
        Agent 3 ──STRUCTURAL_UPDATE────► Agent 1   (boucle structurelle)
        Agent 3 ──VALIDATION_DONE──────► Agent 4
        Agent 4 ──POLICIES_READY───────► Agent 5
        Agent 5 ──SYNTAX_CORRECTION────► Agent 4   (boucle syntaxique)
        Agent 5 ──ODRL_VALID───────────► pipeline
        Agent 5 ──ODRL_SYNTAX_ERROR────► pipeline
    """
    # Agent 1 → Agent 2
    GRAPH_READY        = "graph_ready"
    # Agent 3 → Agent 1  (boucle structurelle)
    STRUCTURAL_UPDATE  = "structural_update"
    # Agent 2 → Agent 3
    CANDIDATES_READY   = "candidates_ready"
    # Agent 3 → Agent 2  (boucle courte)
    REFORMULATE        = "reformulate"
    # Agent 3 → Agent 4
    VALIDATION_DONE    = "validation_done"
    # Agent 4 → Agent 5
    POLICIES_READY     = "policies_ready"
    # Agent 5 → Agent 4  (boucle syntaxique)
    SYNTAX_CORRECTION  = "syntax_correction"
    # Agent 5 → pipeline (signal final)
    ODRL_VALID         = "odrl_valid"
    ODRL_SYNTAX_ERROR  = "odrl_syntax_error"


@dataclass
class AgentMessage:
    """
    Message échangé entre deux agents du pipeline.

    Champs :
        sender    — AGENT_NAME de l'émetteur  (ex: "agent1")
        recipient — AGENT_NAME du destinataire (ex: "agent2")
        msg_type  — type de message (MessageType)
        payload   — données spécifiques au message
        timestamp — ISO généré automatiquement à la création
        loop_turn — tour de boucle courant :
                      · 0 pour le premier message d'un cycle
                      · incrémenté par l'agent qui répond dans une boucle
                      · Exception : Agent 1 conserve le loop_turn reçu
                        lors d'un STRUCTURAL_UPDATE (Agent 3 avait déjà incrémenté)
    """
    sender:    str
    recipient: str
    msg_type:  MessageType
    payload:   dict
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )
    loop_turn: int = 0

    def __repr__(self) -> str:
        return (
            f"AgentMessage({self.msg_type.value} | "
            f"{self.sender} → {self.recipient} | "
            f"turn={self.loop_turn} | {self.timestamp})"
        )


class ReanalysisLimitError(Exception):
    """Levée quand Agent 1 dépasse MAX_REANALYSIS."""
    pass


# ─────────────────────────────────────────────
#  Structures de sortie de l'agent
# ─────────────────────────────────────────────


@dataclass
class StructuralPattern:
    """
    Un pattern structurel détecté dans le graphe.

    Exemples : fork-XOR, join-AND, synchronization point, loop.
    """

    pattern_type: str  # "fork_xor" | "fork_and" | "fork_or" | "join_and" | "join_xor" | "sync" | "loop"
    gateway_id: str  # ID de la gateway concernée
    gateway_name: str
    involved_nodes: list[str]  # IDs des nœuds impliqués
    fragment_id: Optional[str]  # Fragment concerné (None si inter-fragment)
    description: str  # Explication humaine du pattern


@dataclass
class ActivityB2PMapping:
    """
    Mapping entre une activité et ses B2P policies applicables.
    """

    activity_id: str
    activity_name: str
    fragment_id: str
    b2p_policy_ids: list[str]  # UIDs des policies B2P qui ciblent cette activité
    rule_types: list[str]  # ["permission", "prohibition", "obligation"]


@dataclass
class ConnectionInfo:
    """
    Information structurée sur une connexion entre deux activités.
    """

    from_activity: str  # Nom de l'activité source
    to_activity: str  # Nom de l'activité cible
    connection_type: str  # "sequence" | "xor" | "and" | "or" | "message"
    gateway_name: Optional[str]  # Nom de la gateway si présente
    condition: Optional[str]  # Condition de branche
    is_inter: bool  # Inter-fragment ?
    from_fragment: str
    to_fragment: str


@dataclass
class FragmentContext:
    """
    Contexte d'un fragment — utilisé par les agents suivants.

    Deux variantes :
        Global (CG) : vue complète du processus
        Local  (CL) : vue limitée au voisinage immédiat
    """

    fragment_id: str
    activities: list[str]  # Noms des activités du fragment
    gateways: list[str]  # Noms des gateways du fragment
    b2p_mappings: list[ActivityB2PMapping]
    connections: list[ConnectionInfo]  # Connexions internes
    upstream_deps: list[ConnectionInfo]  # Dépendances entrantes (inter)
    downstream_deps: list[ConnectionInfo]  # Dépendances sortantes (inter)

    # Contexte global (CG) — rempli si global_context=True
    global_graph_summary: Optional[dict] = None
    all_inter_edges: Optional[list[ConnectionInfo]] = None

    @property
    def is_global(self) -> bool:
        return self.global_graph_summary is not None


@dataclass
class EnrichedGraph:
    """
    Sortie finale de l'Agent 1.

    C'est ce que les agents suivants consomment.
    Contient le graphe formel + toutes les analyses structurelles.
    """

    graph: BPMNGraph
    patterns: list[StructuralPattern]
    b2p_mappings: dict[str, ActivityB2PMapping]  # activity_id → mapping
    connections: list[ConnectionInfo]
    fragment_contexts: dict[str, FragmentContext]  # fragment_id → contexte local
    global_contexts: dict[str, FragmentContext]  # fragment_id → contexte global
    raw_bpmn: dict  # BPMN original (référence)
    raw_b2p: list[dict]  # B2P policies originales


# ─────────────────────────────────────────────
#  Agent 1 — Structural Analyzer
# ─────────────────────────────────────────────


class StructuralAnalyzer:
    """
    Agent 1 du pipeline multi-agent.

    Reçoit :
        - bp_model   : dict BPMN (activities, gateways, flows)
        - fragments  : list de fragments (output du Fragmenter), ou None pour les calculer
        - b2p_policies : list de policies ODRL du BP global
        - fragmentation_strategy : si fragments=None, stratégie pour EnhancedFragmenter
                                  ('gateway', 'activity', 'connected', 'hierarchical')

    Si fragments est None, l'agent utilise EnhancedFragmenter pour calculer les fragments
    à partir du bp_model. Les ids sont normalisés en "f1", "f2", ... pour cohérence.

    Produit :
        - EnrichedGraph : graphe formel enrichi, prêt pour les agents suivants

    Cet agent est DÉTERMINISTE — pas de LLM, pas d'aléatoire.
    Chaque opération est traceable et justifiable.

    ── Couche multi-agent ──────────────────────────────────────────────
    En mode connecté (pipeline complet) :
        analyzer.register_send_callback(agent2.receive)
        analyzer.analyze_and_send()   # démarre le pipeline

    En mode standalone (tests unitaires) :
        enriched_graph = analyzer.analyze()   # aucun callback nécessaire
    """

    AGENT_NAME     = "agent1"
    MAX_REANALYSIS = 1   # sécurité : max 1 enrichissement structurel par session

    def __init__(
        self,
        bp_model: dict,
        fragments: Optional[list[dict]] = None,
        b2p_policies: Optional[list[dict]] = None,
        fragmentation_strategy: str = "gateway",
    ):
        self.bp_model = bp_model
        self.fragments = fragments
        self.b2p_policies = b2p_policies or []
        self.fragmentation_strategy = fragmentation_strategy
        self.graph = BPMNGraph()

        # Index interne : nom → id (pour la résolution des flows)
        self._name_to_id: dict[str, str] = {}
        self._fragment_of: dict[str, str] = {}  # node_name → fragment_id

        # ── Attributs multi-agent ──────────────────────────────────
        self._on_send:            Optional[Callable[[AgentMessage], None]] = None
        self._reanalysis_count:   int = 0
        self._last_enriched_graph: Optional[EnrichedGraph] = None

    # ─────────────────────────────────────────
    #  Couche multi-agent
    # ─────────────────────────────────────────

    def register_send_callback(self, fn: Callable[[AgentMessage], None]) -> None:
        """
        Enregistre le callback vers Agent 2.
        En mode pipeline : analyzer.register_send_callback(agent2.receive)
        """
        self._on_send = fn
        print(f"[Agent 1] Callback send enregistré → {fn}")

    def send(self, msg: AgentMessage) -> None:
        """Émet un AgentMessage vers le destinataire enregistré."""
        print(f"[Agent 1] ► SEND {msg}")
        if self._on_send:
            self._on_send(msg)
        else:
            print(f"[Agent 1][WARN] Aucun callback — message '{msg.msg_type.value}' non transmis.")

    def receive(self, msg: AgentMessage) -> None:
        """
        Point d'entrée pour les messages entrants.
        Accepte uniquement STRUCTURAL_UPDATE (de Agent 3).
        Ignore tout autre type avec warning — jamais d'exception.
        """
        print(f"[Agent 1] ◄ RECEIVE {msg}")
        if msg.msg_type == MessageType.STRUCTURAL_UPDATE:
            self._handle_structural_update(msg)
        else:
            print(f"[Agent 1][WARN] Message '{msg.msg_type.value}' non géré — ignoré.")

    def analyze_and_send(self) -> None:
        """
        Version connectée de analyze() pour le mode pipeline.
        Lance l'analyse complète puis émet GRAPH_READY vers Agent 2.

        En mode standalone (tests), utiliser analyze() directement.
        """
        enriched_graph = self.analyze()
        self._last_enriched_graph = enriched_graph
        self.send(AgentMessage(
            sender    = self.AGENT_NAME,
            recipient = "agent2",
            msg_type  = MessageType.GRAPH_READY,
            payload   = {
                "enriched_graph":     enriched_graph,
                "fragment_ids":       list(enriched_graph.fragment_contexts.keys()),
                "reanalysis":         False,
                "affected_fragments": None,
            },
        ))

    def _handle_structural_update(self, msg: AgentMessage) -> None:
        """
        Traite un STRUCTURAL_UPDATE d'Agent 3.

        Enrichissement CHIRURGICAL — sans re-exécuter analyze().
        Injecte les arêtes implicites dans _last_enriched_graph via
        _add_implicit_connections(), puis réémet GRAPH_READY.

        Lève ReanalysisLimitError si MAX_REANALYSIS est dépassé.
        """
        if self._reanalysis_count >= self.MAX_REANALYSIS:
            raise ReanalysisLimitError(
                f"[Agent 1] MAX_REANALYSIS ({self.MAX_REANALYSIS}) atteint. "
                "Arrêt de la boucle structurelle."
            )
        if self._last_enriched_graph is None:
            raise RuntimeError(
                "[Agent 1] Aucun graphe en mémoire — "
                "analyze_and_send() doit être appelé avant receive()."
            )

        edges_to_add   = msg.payload.get("implicit_edges_to_add", [])
        affected_frags = msg.payload.get("affected_fragments", [])

        print(
            f"[Agent 1] Enrichissement structurel demandé par Agent 3 "
            f"({len(edges_to_add)} arête(s) à ajouter, "
            f"fragments affectés : {affected_frags})"
        )
        if msg.payload.get("reason"):
            print(f"[Agent 1]   Raison : {msg.payload['reason']}")

        new_connections = self._add_implicit_connections(edges_to_add)
        self._reanalysis_count += 1

        print(
            f"[Agent 1] Enrichissement #{self._reanalysis_count} / {self.MAX_REANALYSIS} max — "
            f"{len(new_connections)} connexion(s) ajoutée(s)"
        )

        # Conserver le loop_turn reçu — Agent 3 avait déjà incrémenté
        self.send(AgentMessage(
            sender    = self.AGENT_NAME,
            recipient = "agent2",
            msg_type  = MessageType.GRAPH_READY,
            payload   = {
                "enriched_graph":     self._last_enriched_graph,
                "fragment_ids":       list(self._last_enriched_graph.fragment_contexts.keys()),
                "reanalysis":         True,
                "affected_fragments": affected_frags,
                "new_connections":    new_connections,
            },
            loop_turn = msg.loop_turn,  # pas d'incrémentation ici
        ))

    def _add_implicit_connections(self, edges: list[dict]) -> list[ConnectionInfo]:
        """
        Injecte des ConnectionInfo implicites dans le graphe existant.

        Opération chirurgicale — ne touche pas aux nœuds, patterns,
        ni à la structure des fragments. Utilise _name_to_id (déjà
        construit lors de analyze(), stable en mémoire) pour résoudre
        les noms sans risque de désynchronisation.

        Chaque edge dict attendu :
        {
            "source_activity": str,      # nom exact de l'activité source
            "target_activity": str,      # nom exact de l'activité cible
            "from_fragment":   str,      # fragment_id de la source
            "to_fragment":     str,      # fragment_id de la cible
            "dep_type":        str,      # "temporal"|"data"|"role"|"compliance"|"conflict"
            "is_inter":        bool,     # True si inter-fragment
            "condition":       str|None, # condition optionnelle
        }

        Met à jour upstream_deps / downstream_deps des FragmentContext
        concernés dans _last_enriched_graph.

        Retourne la liste des ConnectionInfo effectivement ajoutées.
        """
        new_connections: list[ConnectionInfo] = []

        for edge in edges:
            src_name  = edge.get("source_activity", "")
            tgt_name  = edge.get("target_activity", "")
            from_frag = edge.get("from_fragment", "")
            to_frag   = edge.get("to_fragment", "")
            is_inter  = edge.get("is_inter", from_frag != to_frag)

            # Vérifier que les activités existent dans le graphe via _name_to_id
            if src_name not in self._name_to_id:
                print(
                    f"[Agent 1][WARN] Activité source '{src_name}' "
                    "inconnue dans _name_to_id — arête ignorée."
                )
                continue
            if tgt_name not in self._name_to_id:
                print(
                    f"[Agent 1][WARN] Activité cible '{tgt_name}' "
                    "inconnue dans _name_to_id — arête ignorée."
                )
                continue

            # Vérifier les fragments
            if from_frag and from_frag not in self._last_enriched_graph.fragment_contexts:
                print(
                    f"[Agent 1][WARN] Fragment source '{from_frag}' "
                    "inconnu — arête ajoutée sans mise à jour du contexte."
                )
            if to_frag and to_frag not in self._last_enriched_graph.fragment_contexts:
                print(
                    f"[Agent 1][WARN] Fragment cible '{to_frag}' "
                    "inconnu — arête ajoutée sans mise à jour du contexte."
                )

            conn = ConnectionInfo(
                from_activity   = src_name,
                to_activity     = tgt_name,
                connection_type = edge.get("dep_type", "implicit"),
                gateway_name    = None,   # implicite — pas de gateway BPMN
                condition       = edge.get("condition"),
                is_inter        = is_inter,
                from_fragment   = from_frag,
                to_fragment     = to_frag,
            )

            # Injecter dans la liste globale des connexions
            self._last_enriched_graph.connections.append(conn)
            new_connections.append(conn)

            # Mettre à jour les contextes de fragments (upstream/downstream)
            if from_frag in self._last_enriched_graph.fragment_contexts:
                ctx = self._last_enriched_graph.fragment_contexts[from_frag]
                if is_inter and conn not in ctx.downstream_deps:
                    ctx.downstream_deps.append(conn)
                # Mettre à jour aussi le contexte global si présent
                if from_frag in self._last_enriched_graph.global_contexts:
                    gctx = self._last_enriched_graph.global_contexts[from_frag]
                    if is_inter and conn not in gctx.downstream_deps:
                        gctx.downstream_deps.append(conn)
                    if gctx.all_inter_edges is not None and conn not in gctx.all_inter_edges:
                        gctx.all_inter_edges.append(conn)

            if to_frag in self._last_enriched_graph.fragment_contexts:
                ctx = self._last_enriched_graph.fragment_contexts[to_frag]
                if is_inter and conn not in ctx.upstream_deps:
                    ctx.upstream_deps.append(conn)
                # Mettre à jour aussi le contexte global si présent
                if to_frag in self._last_enriched_graph.global_contexts:
                    gctx = self._last_enriched_graph.global_contexts[to_frag]
                    if is_inter and conn not in gctx.upstream_deps:
                        gctx.upstream_deps.append(conn)
                    if gctx.all_inter_edges is not None and conn not in gctx.all_inter_edges:
                        gctx.all_inter_edges.append(conn)

            print(
                f"[Agent 1] ✓ Arête implicite ajoutée : "
                f"'{src_name}' → '{tgt_name}' "
                f"({'inter' if is_inter else 'intra'}, "
                f"type={edge.get('dep_type', 'implicit')})"
            )

        return new_connections

    # ─────────────────────────────────────────
    #  Point d'entrée principal
    # ─────────────────────────────────────────

    def analyze(self) -> EnrichedGraph:
        """
        Lance l'analyse complète.

        Étapes :
            0. Si fragments non fourni : calcul via EnhancedFragmenter
            1. Construire le graphe formel G depuis le BPMN
            2. Mapper les B2P policies aux activités
            3. Détecter les patterns structurels
            4. Construire les ConnectionInfo
            5. Construire les contextes locaux CL et globaux CG
        """
        print("[Agent 1] Structural Analyzer — démarrage de l'analyse")

        # Étape 0 — Calcul des fragments si non fournis
        if self.fragments is None:
            self.fragments = self._compute_fragments()
            print(f"[Agent 1] Fragments calculés par EnhancedFragmenter : {len(self.fragments)}")

        # Étape 1 — Construction du graphe formel
        self._build_graph()
        print(f"[Agent 1] Graphe construit : {self.graph}")

        # Étape 2 — Mapping B2P → activités
        b2p_mappings = self._map_b2p_to_activities()
        print(f"[Agent 1] B2P mappings : {len(b2p_mappings)} activités mappées")

        # Étape 3 — Détection des patterns
        patterns = self._detect_patterns()
        print(f"[Agent 1] Patterns détectés : {len(patterns)}")
        for p in patterns:
            print(f"    -> [{p.pattern_type}] {p.gateway_name} : {p.description}")

        # Étape 4 — Construction des ConnectionInfo
        connections = self._build_connections()
        print(f"[Agent 1] Connexions analysées : {len(connections)}")

        # Étape 5 — Contextes locaux et globaux par fragment
        local_contexts = self._build_local_contexts(b2p_mappings, connections)
        global_contexts = self._build_global_contexts(b2p_mappings, connections)
        print(f"[Agent 1] Contextes construits pour {len(local_contexts)} fragments")

        return EnrichedGraph(
            graph=self.graph,
            patterns=patterns,
            b2p_mappings=b2p_mappings,
            connections=connections,
            fragment_contexts=local_contexts,
            global_contexts=global_contexts,
            raw_bpmn=self.bp_model,
            raw_b2p=self.b2p_policies,
        )

    def _compute_fragments(self) -> list[dict]:
        """
        Calcule les fragments via EnhancedFragmenter.
        Normalise les ids en "f1", "f2", ... pour cohérence avec les tests.
        """
        from enhanced_fragmenter import EnhancedFragmenter

        fragmenter = EnhancedFragmenter(self.bp_model)
        raw_fragments = fragmenter.fragment_process(strategy=self.fragmentation_strategy)
        # Normaliser les ids (EnhancedFragmenter renvoie 0, 1, 2...)
        normalized = []
        for i, frag in enumerate(raw_fragments):
            f = dict(frag)
            f["id"] = f"f{i + 1}"
            normalized.append(f)
        return normalized

    # ─────────────────────────────────────────
    #  Étape 1 — Construction du graphe formel
    # ─────────────────────────────────────────

    def _build_graph(self) -> None:
        """Transforme le BPMN brut en graphe formel G = (V, E)."""

        # Construire le mapping nom → fragment_id
        for i, frag in enumerate(self.fragments):
            frag_id = frag.get("id", f"f{i}")
            frag["id"] = frag_id
            for act_name in frag.get("activities", []):
                self._fragment_of[act_name] = frag_id

        # ── Nœuds Activity ──
        for act in self.bp_model.get("activities", []):
            node_id = f"act_{act['name'].replace(' ', '_')}"
            frag_id = self._fragment_of.get(act["name"])

            node = ActivityNode(
                id=node_id,
                name=act["name"],
                node_type=NodeType.ACTIVITY,
                fragment_id=frag_id,
                role=act.get("role"),
                is_start=act.get("start", False),
                is_end=act.get("end", False),
                metadata={k: v for k, v in act.items() if k not in ("name", "role", "start", "end")},
            )
            self.graph.add_node(node)
            self._name_to_id[act["name"]] = node_id

        # ── Nœuds Gateway ──
        for gw in self.bp_model.get("gateways", []):
            node_id = f"gw_{gw['name'].replace(' ', '_')}"

            gw_type_str = gw.get("type", "XOR").upper()
            gw_type = {
                "XOR": GatewayType.XOR,
                "AND": GatewayType.AND,
                "OR": GatewayType.OR,
            }.get(gw_type_str, GatewayType.XOR)

            node = GatewayNode(
                id=node_id,
                name=gw["name"],
                node_type=NodeType.GATEWAY,
                gateway_type=gw_type,
                fragment_id=None,
            )
            self.graph.add_node(node)
            self._name_to_id[gw["name"]] = node_id

        # ── Arêtes (flows) ──
        for i, flow in enumerate(self.bp_model.get("flows", [])):
            from_name = flow["from"]
            to_name = flow["to"]
            flow_type = flow.get("type", "sequence").lower()

            from_id = self._name_to_id.get(from_name)
            to_id = self._name_to_id.get(to_name)

            if not from_id or not to_id:
                print(f"[Agent 1][WARN] Flow ignoré : '{from_name}' → '{to_name}' (nœud introuvable)")
                continue

            # Type d'arête
            edge_type = EdgeType.MESSAGE if flow_type == "message" else EdgeType.SEQUENCE

            # Type de dépendance selon la gateway
            dep_type = DependencyType.CONTROL
            if flow_type == "message":
                dep_type = DependencyType.MESSAGE

            # Gateway associée
            gw_id = None
            gw_name = flow.get("gateway")
            if gw_name and gw_name in self._name_to_id:
                gw_id = self._name_to_id[gw_name]

            edge = Edge(
                id=f"e{i}_{from_name.replace(' ', '_')}_{to_name.replace(' ', '_')}",
                source=from_id,
                target=to_id,
                edge_type=edge_type,
                dependency_type=dep_type,
                gateway_id=gw_id,
                condition=flow.get("condition"),
            )
            self.graph.add_edge(edge)

        # Assigner les gateways aux fragments selon leurs activités voisines
        self._assign_gateways_to_fragments()

    def _assign_gateways_to_fragments(self) -> None:
        """
        Assigne chaque gateway au fragment de son activité source majoritaire.
        Une gateway sans fragment assigné est considérée comme inter-fragment.
        """
        for node in self.graph.all_nodes():
            if not isinstance(node, GatewayNode):
                continue
            if node.fragment_id:
                continue

            # Chercher le fragment des prédécesseurs
            pred_fragments = [p.fragment_id for p in self.graph.predecessors(node.id) if p.fragment_id]
            if pred_fragments:
                node.fragment_id = pred_fragments[0]

    # ─────────────────────────────────────────
    #  Étape 2 — Mapping B2P → activités
    # ─────────────────────────────────────────

    def _map_b2p_to_activities(self) -> dict[str, ActivityB2PMapping]:
        """
        Pour chaque activité, trouve les B2P policies qui la ciblent.

        Stratégie de matching :
            - Exact match sur le nom de l'activité dans l'URI target
            - Partial match (nom normalisé)
        """
        mappings: dict[str, ActivityB2PMapping] = {}

        for node in self.graph.all_nodes():
            if not isinstance(node, ActivityNode):
                continue

            matched_policy_ids: list[str] = []
            matched_rule_types: list[str] = []

            for policy in self.b2p_policies:
                policy_uid = policy.get("uid", "")

                for rule_type in ("permission", "prohibition", "obligation"):
                    for rule in policy.get(rule_type, []):
                        target = rule.get("target", "")
                        # Matching : le nom de l'activité apparaît dans l'URI target
                        if self._name_matches_target(node.name, target):
                            if policy_uid not in matched_policy_ids:
                                matched_policy_ids.append(policy_uid)
                            if rule_type not in matched_rule_types:
                                matched_rule_types.append(rule_type)

            mappings[node.id] = ActivityB2PMapping(
                activity_id=node.id,
                activity_name=node.name,
                fragment_id=node.fragment_id or "unknown",
                b2p_policy_ids=matched_policy_ids,
                rule_types=matched_rule_types,
            )

        return mappings

    def _name_matches_target(self, activity_name: str, target_uri: str) -> bool:
        """
        Vérifie si l'activité est ciblée par l'URI ODRL.

        Normalise les deux chaînes pour la comparaison.
        """
        normalized_name = activity_name.lower().replace(" ", "_").replace("-", "_")
        normalized_target = target_uri.lower().replace("-", "_").replace("/", "_")
        return normalized_name in normalized_target

    # ─────────────────────────────────────────
    #  Étape 3 — Détection des patterns
    # ─────────────────────────────────────────

    def _detect_patterns(self) -> list[StructuralPattern]:
        """
        Détecte les patterns structurels BPMN dans le graphe.

        Patterns détectés :
            - fork_xor  : gateway XOR avec plusieurs sorties
            - fork_and  : gateway AND avec plusieurs sorties
            - fork_or   : gateway OR avec plusieurs sorties
            - join_xor  : gateway XOR avec plusieurs entrées
            - join_and  : gateway AND avec plusieurs entrées
            - sync      : point de synchronisation (join_and critique)
            - loop      : cycle détecté
        """
        patterns: list[StructuralPattern] = []

        for node in self.graph.all_nodes():
            if not isinstance(node, GatewayNode):
                continue

            out_count = len(self.graph.out_edges(node.id))
            in_count = len(self.graph.in_edges(node.id))
            gw_type = node.gateway_type

            # Fork patterns (plusieurs sorties)
            if out_count > 1:
                involved = [e.target for e in self.graph.out_edges(node.id)]
                pattern_type = f"fork_{gw_type.value.lower()}" if gw_type else "fork_unknown"

                description = self._describe_fork(gw_type, node.name, involved)
                patterns.append(
                    StructuralPattern(
                        pattern_type=pattern_type,
                        gateway_id=node.id,
                        gateway_name=node.name,
                        involved_nodes=[node.id] + involved,
                        fragment_id=node.fragment_id,
                        description=description,
                    )
                )

            # Join patterns (plusieurs entrées)
            if in_count > 1:
                involved = [e.source for e in self.graph.in_edges(node.id)]
                pattern_type = f"join_{gw_type.value.lower()}" if gw_type else "join_unknown"

                description = self._describe_join(gw_type, node.name, involved)
                patterns.append(
                    StructuralPattern(
                        pattern_type=pattern_type,
                        gateway_id=node.id,
                        gateway_name=node.name,
                        involved_nodes=involved + [node.id],
                        fragment_id=node.fragment_id,
                        description=description,
                    )
                )

                # Sync point : join AND = synchronisation obligatoire
                if gw_type == GatewayType.AND:
                    patterns.append(
                        StructuralPattern(
                            pattern_type="sync",
                            gateway_id=node.id,
                            gateway_name=node.name,
                            involved_nodes=involved + [node.id],
                            fragment_id=node.fragment_id,
                            description=(
                                f"Point de synchronisation obligatoire en '{node.name}'. "
                                f"Toutes les branches parallèles doivent se terminer "
                                f"avant de continuer."
                            ),
                        )
                    )

        # Détection de loop
        if self.graph.has_cycle():
            patterns.append(
                StructuralPattern(
                    pattern_type="loop",
                    gateway_id="",
                    gateway_name="cycle_detected",
                    involved_nodes=[],
                    fragment_id=None,
                    description="Un cycle a été détecté dans le graphe. Vérifier les boucles BPMN.",
                )
            )

        return patterns

    def _describe_fork(self, gw_type: Optional[GatewayType], gw_name: str, targets: list[str]) -> str:
        if gw_type == GatewayType.XOR:
            return (
                f"Fork exclusif en '{gw_name}' : une seule branche sera activée parmi {len(targets)}. "
                f"Les policies doivent implémenter enable(ruleA XOR ruleB)."
            )
        if gw_type == GatewayType.AND:
            return (
                f"Fork parallèle en '{gw_name}' : toutes les {len(targets)} branches s'activent simultanément. "
                f"Les policies doivent implémenter enable(ruleA AND ruleB)."
            )
        if gw_type == GatewayType.OR:
            return (
                f"Fork inclusif en '{gw_name}' : une ou plusieurs branches parmi {len(targets)}. "
                f"Les policies doivent implémenter enable(ruleA OR ruleB)."
            )
        return f"Fork inconnu en '{gw_name}'."

    def _describe_join(self, gw_type: Optional[GatewayType], gw_name: str, sources: list[str]) -> str:
        if gw_type == GatewayType.AND:
            return (
                f"Join synchronisant en '{gw_name}' : attend la complétion de toutes les {len(sources)} branches. "
                f"Risque de deadlock si une policy bloque une branche."
            )
        if gw_type == GatewayType.XOR:
            return (
                f"Join alternatif en '{gw_name}' : la première branche complétée continue. "
                f"Les autres branches sont ignorées."
            )
        return f"Join inconnu en '{gw_name}'."

    # ─────────────────────────────────────────
    #  Étape 4 — ConnectionInfo
    # ─────────────────────────────────────────

    def _build_connections(self) -> list[ConnectionInfo]:
        """
        Construit la liste de toutes les connexions typées entre activités.

        - Connexions directes : arêtes activity → activity.
        - Connexions via gateway : chemins activity → gateway → activity
          (pour capturer les dépendances inter-fragments quand la gateway
          est entre deux fragments, ex. f1 --[GW]--> f2).
        """
        connections: list[ConnectionInfo] = []
        seen: set[tuple[str, str]] = set()  # (from_activity, to_activity) pour déduplication

        # 1) Arêtes directes activity → activity
        for edge in self.graph.all_edges():
            src_node = self.graph.get_node(edge.source)
            tgt_node = self.graph.get_node(edge.target)

            if not src_node or not tgt_node:
                continue

            if not isinstance(src_node, ActivityNode) or not isinstance(tgt_node, ActivityNode):
                continue

            key = (src_node.name, tgt_node.name)
            if key in seen:
                continue
            seen.add(key)

            conn_type = self._resolve_connection_type(edge)
            gw_name = None
            if edge.gateway_id:
                gw_node = self.graph.get_node(edge.gateway_id)
                gw_name = gw_node.name if gw_node else None

            connections.append(
                ConnectionInfo(
                    from_activity=src_node.name,
                    to_activity=tgt_node.name,
                    connection_type=conn_type,
                    gateway_name=gw_name,
                    condition=edge.condition,
                    is_inter=edge.is_inter,
                    from_fragment=src_node.fragment_id or "unknown",
                    to_fragment=tgt_node.fragment_id or "unknown",
                )
            )

        # 2) Connexions activity → gateway → activity (dépendances inter-fragments)
        for node in self.graph.all_nodes():
            if not isinstance(node, GatewayNode):
                continue
            gateway = node

            pred_activities = [
                n for n in self.graph.predecessors(gateway.id)
                if isinstance(n, ActivityNode)
            ]
            for out_edge in self.graph.out_edges(gateway.id):
                tgt = self.graph.get_node(out_edge.target)
                if not tgt or not isinstance(tgt, ActivityNode):
                    continue
                succ_activity = tgt
                for pred_activity in pred_activities:
                    key = (pred_activity.name, succ_activity.name)
                    if key in seen:
                        continue
                    seen.add(key)

                    is_inter = (
                        pred_activity.fragment_id != succ_activity.fragment_id
                        and pred_activity.fragment_id
                        and succ_activity.fragment_id
                    )
                    conn_type = (
                        gateway.gateway_type.value.lower()
                        if gateway.gateway_type
                        else "sequence"
                    )
                    connections.append(
                        ConnectionInfo(
                            from_activity=pred_activity.name,
                            to_activity=succ_activity.name,
                            connection_type=conn_type,
                            gateway_name=gateway.name,
                            condition=out_edge.condition,
                            is_inter=is_inter,
                            from_fragment=pred_activity.fragment_id or "unknown",
                            to_fragment=succ_activity.fragment_id or "unknown",
                        )
                    )

        return connections

    def _resolve_connection_type(self, edge: Edge) -> str:
        """Détermine le type de connexion sémantique depuis une arête."""
        if edge.edge_type == EdgeType.MESSAGE:
            return "message"

        if edge.gateway_id:
            gw_node = self.graph.get_node(edge.gateway_id)
            if gw_node and isinstance(gw_node, GatewayNode):
                return gw_node.gateway_type.value.lower() if gw_node.gateway_type else "sequence"

        return "sequence"

    # ─────────────────────────────────────────
    #  Étape 5 — Contextes CL et CG
    # ─────────────────────────────────────────

    def _build_local_contexts(
        self,
        b2p_mappings: dict[str, ActivityB2PMapping],
        all_connections: list[ConnectionInfo],
    ) -> dict[str, FragmentContext]:
        """
        Construit le contexte LOCAL CL(Fi) pour chaque fragment.

        CL(Fi) =
            - activités et gateways de Fi
            - B2P mappings des activités de Fi
            - connexions internes à Fi
            - dépendances avec les fragments voisins immédiats uniquement
        """
        contexts: dict[str, FragmentContext] = {}

        for i, frag in enumerate(self.fragments):
            frag_id = frag.get("id", f"f{i}")

            activities = frag.get("activities", [])
            gateways = []
            gw_field = frag.get("gateways", [])
            if gw_field:
                # Peut être une liste de noms ou d'objets gateway
                if isinstance(gw_field[0], str):
                    gateways = list(gw_field)
                else:
                    gateways = [gw.get("name", "") for gw in gw_field]

            # B2P mappings des activités de ce fragment
            frag_mappings = [
                m for m in b2p_mappings.values() if m.fragment_id == frag_id
            ]

            # Connexions internes
            internal_conns = [
                c for c in all_connections if c.from_fragment == frag_id and c.to_fragment == frag_id
            ]

            # Dépendances entrantes (upstream)
            upstream = [c for c in all_connections if c.to_fragment == frag_id and c.is_inter]

            # Dépendances sortantes (downstream)
            downstream = [c for c in all_connections if c.from_fragment == frag_id and c.is_inter]

            contexts[frag_id] = FragmentContext(
                fragment_id=frag_id,
                activities=activities,
                gateways=gateways,
                b2p_mappings=frag_mappings,
                connections=internal_conns,
                upstream_deps=upstream,
                downstream_deps=downstream,
                global_graph_summary=None,  # Local → pas de vue globale
                all_inter_edges=None,
            )

        return contexts

    def _build_global_contexts(
        self,
        b2p_mappings: dict[str, ActivityB2PMapping],
        all_connections: list[ConnectionInfo],
    ) -> dict[str, FragmentContext]:
        """
        Construit le contexte GLOBAL CG(Fi) pour chaque fragment.

        CG(Fi) = tout ce que CL(Fi) contient +
            - résumé complet du graphe global
            - toutes les arêtes inter-fragments
            - ordering global
        """
        local_contexts = self._build_local_contexts(b2p_mappings, all_connections)
        global_summary = self.graph.summary()
        all_inter = [c for c in all_connections if c.is_inter]

        global_contexts: dict[str, FragmentContext] = {}
        for frag_id, local_ctx in local_contexts.items():
            global_contexts[frag_id] = FragmentContext(
                fragment_id=local_ctx.fragment_id,
                activities=local_ctx.activities,
                gateways=local_ctx.gateways,
                b2p_mappings=local_ctx.b2p_mappings,
                connections=local_ctx.connections,
                upstream_deps=local_ctx.upstream_deps,
                downstream_deps=local_ctx.downstream_deps,
                global_graph_summary=global_summary,  # Vue globale
                all_inter_edges=all_inter,  # Toutes les dépendances inter
            )

        return global_contexts