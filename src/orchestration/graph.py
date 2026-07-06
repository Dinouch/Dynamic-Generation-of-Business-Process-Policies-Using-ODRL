"""
graph.py — Layer 0: Central formal representation

This is the system's source of truth. All agents manipulate
objects from this module, never raw text.

Structure:
    G = (V, E)
    - V: typed nodes (Activity, Gateway, Event)
    - E: typed edges (sequence, message, control)

Each edge carries a dependency type:
    - CONTROL   : control sequencing (XOR, AND, OR)
    - MESSAGE   : inter-fragment communication
    - RESOURCE  : resource sharing
    - TEMPORAL  : time constraint
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────
#  Enums — Formal system vocabulary
# ─────────────────────────────────────────────


class NodeType(Enum):
    ACTIVITY = "activity"
    GATEWAY = "gateway"
    EVENT = "event"


class GatewayType(Enum):
    XOR = "XOR"  # Exclusive → single branch
    AND = "AND"  # Parallel → all branches
    OR = "OR"    # Inclusive → one or more branches
    EVENT_BASED_EXCLUSIVE = "EVENT_BASED_EXCLUSIVE"  # Event-based gateway (exclusive)
    EVENT_BASED_PARALLEL = "EVENT_BASED_PARALLEL"    # Event-based gateway (parallel)
    COMPLEX = "COMPLEX"                              # Complex gateway
    DEFAULT = "DEFAULT"                              # Default flow (marker, rarely as node type)
    CONDITIONAL = "CONDITIONAL"                      # Conditional sequence (marker)


class EdgeType(Enum):
    SEQUENCE = "sequence"   # Sequential control flow
    MESSAGE = "message"     # Inter-fragment communication
    RESOURCE = "resource"   # Resource sharing
    TEMPORAL = "temporal"   # Time constraint


class DependencyType(Enum):
    """Semantic dependency type on an edge."""
    CONTROL = "control"     # Flow control (gateway)
    DATA = "data"           # Data flow
    MESSAGE = "message"     # Inter-fragment message
    RESOURCE = "resource"   # Shared resource
    TEMPORAL = "temporal"   # Time constraint


# ─────────────────────────────────────────────
#  Graph nodes
# ─────────────────────────────────────────────


@dataclass
class Node:
    """Generic node in the formal BPMN graph."""
    id: str
    name: str
    node_type: NodeType
    fragment_id: Optional[str] = None          # Fragment this node belongs to
    metadata: dict = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __repr__(self):
        return f"Node({self.node_type.value}:{self.name})"


@dataclass
class ActivityNode(Node):
    """Activity node — corresponds to an ODRL asset."""
    role: Optional[str] = None   # BPMN role/lane
    is_start: bool = False
    is_end: bool = False

    def __post_init__(self):
        self.node_type = NodeType.ACTIVITY


@dataclass
class GatewayNode(Node):
    """Gateway node — determines the dependency type."""
    gateway_type: Optional[GatewayType] = None

    def __post_init__(self):
        self.node_type = NodeType.GATEWAY


@dataclass
class EventNode(Node):
    """Event node (start, end, intermediate)."""
    event_kind: str = "intermediate"   # start | end | intermediate

    def __post_init__(self):
        self.node_type = NodeType.EVENT


# ─────────────────────────────────────────────
#  Graph edges
# ─────────────────────────────────────────────


@dataclass
class Edge:
    """
    Typed edge between two nodes.

    Carries:
        - edge_type       : flow nature (sequence, message…)
        - dependency_type : dependency semantics (control, data…)
        - gateway         : associated gateway if present
        - condition       : branch condition (XOR/OR)
        - is_inter        : True if the edge crosses two different fragments
    """
    id: str
    source: str                        # source node id
    target: str                        # target node id
    edge_type: EdgeType = EdgeType.SEQUENCE
    dependency_type: DependencyType = DependencyType.CONTROL
    gateway_id: Optional[str] = None      # Gateway that generates this edge
    condition: Optional[str] = None       # Branch condition (XOR/OR)
    is_inter: bool = False                # Inter-fragment?
    metadata: dict = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        arrow = "--msg-->" if self.edge_type == EdgeType.MESSAGE else "-->"
        inter = " [INTER]" if self.is_inter else ""
        return f"Edge({self.source} {arrow} {self.target}{inter})"


# ─────────────────────────────────────────────
#  Central formal graph  G = (V, E)
# ─────────────────────────────────────────────


class BPMNGraph:
    """
    Central formal graph of the system.

    This is the single source of truth.
    Agents read this graph — they never read raw BPMN.

    Usage:
        g = BPMNGraph()
        g.add_node(ActivityNode(id="a1", name="Check completeness", ...))
        g.add_edge(Edge(id="e1", source="a1", target="a2", ...))

        successors   = g.successors("a1")
        predecessors = g.predecessors("a2")
        sub          = g.subgraph(fragment_id="f1")
    """

    def __init__(self):
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}

        # Adjacency index for fast lookups
        self._out_edges: dict[str, list[str]] = {}   # node_id → [edge_id]
        self._in_edges: dict[str, list[str]] = {}    # node_id → [edge_id]

    # ── Add nodes / edges ──────────────────

    def add_node(self, node: Node) -> None:
        if node.id in self._nodes:
            raise ValueError(f"Node '{node.id}' already exists in the graph.")
        self._nodes[node.id] = node
        self._out_edges[node.id] = []
        self._in_edges[node.id] = []

    def add_edge(self, edge: Edge) -> None:
        if edge.source not in self._nodes:
            raise ValueError(f"Source node '{edge.source}' not found.")
        if edge.target not in self._nodes:
            raise ValueError(f"Target node '{edge.target}' not found.")
        if edge.id in self._edges:
            raise ValueError(f"Edge '{edge.id}' already exists.")

        # Automatically mark as inter-fragment when fragments differ
        src_frag = self._nodes[edge.source].fragment_id
        tgt_frag = self._nodes[edge.target].fragment_id
        if src_frag and tgt_frag and src_frag != tgt_frag:
            edge.is_inter = True

        self._edges[edge.id] = edge
        self._out_edges[edge.source].append(edge.id)
        self._in_edges[edge.target].append(edge.id)

    # ── Graph queries ────────────────────

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)


    def get_edge(self, edge_id: str) -> Optional[Edge]:
        return self._edges.get(edge_id)


    def successors(self, node_id: str) -> list[Node]:
        """Return direct successor nodes."""
        return [
            self._nodes[self._edges[eid].target]
            for eid in self._out_edges.get(node_id, [])
        ]

    def predecessors(self, node_id: str) -> list[Node]:
        """Return direct predecessor nodes."""
        return [
            self._nodes[self._edges[eid].source]
            for eid in self._in_edges.get(node_id, [])
        ]

    def out_edges(self, node_id: str) -> list[Edge]:
        """Return outgoing edges from a node."""
        return [self._edges[eid] for eid in self._out_edges.get(node_id, [])]

    def in_edges(self, node_id: str) -> list[Edge]:
        """Return incoming edges to a node."""
        return [self._edges[eid] for eid in self._in_edges.get(node_id, [])]

    def subgraph(self, fragment_id: str) -> "BPMNGraph":
        """
        Return the subgraph induced by a given fragment.
        Also includes inter-fragment edges (is_inter=True)
        whose source belongs to this fragment.
        """
        sub = BPMNGraph()

        # Fragment nodes
        for node in self._nodes.values():
            if node.fragment_id == fragment_id:
                sub.add_node(node)

        # Internal edges + outgoing inter edges
        sub_node_ids = set(sub._nodes.keys())
        for edge in self._edges.values():
            if edge.source in sub_node_ids:
                # Internal edge
                if edge.target in sub_node_ids:
                    sub.add_edge(edge)
                # Outgoing inter-fragment edge: include without target node
                elif edge.is_inter:
                    sub._edges[edge.id] = edge
                    sub._out_edges[edge.source].append(edge.id)

        return sub

    def inter_edges(self) -> list[Edge]:
        """Return all inter-fragment edges."""
        return [e for e in self._edges.values() if e.is_inter]

    def nodes_of_fragment(self, fragment_id: str) -> list[Node]:
        """Return all nodes belonging to a fragment."""
        return [n for n in self._nodes.values() if n.fragment_id == fragment_id]

    def all_nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def all_edges(self) -> list[Edge]:
        return list(self._edges.values())

    # ── Structural pattern detection ────────

    def detect_fork(self, node_id: str) -> Optional[GatewayType]:
        """
        If node_id is a gateway with multiple outgoing edges,
        return its type (fork AND/OR/XOR). Otherwise None.
        """
        node = self._nodes.get(node_id)
        if node and isinstance(node, GatewayNode):
            if len(self._out_edges.get(node_id, [])) > 1:
                return node.gateway_type
        return None

    def detect_join(self, node_id: str) -> Optional[GatewayType]:
        """
        If node_id is a gateway with multiple incoming edges,
        return its type (join AND/OR/XOR). Otherwise None.
        """
        node = self._nodes.get(node_id)
        if node and isinstance(node, GatewayNode):
            if len(self._in_edges.get(node_id, [])) > 1:
                return node.gateway_type
        return None

    def has_cycle(self) -> bool:
        """Detect a cycle in the graph (DFS)."""
        visited = set()
        rec_stack = set()

        def dfs(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)
            for successor in self.successors(node_id):
                if successor.id not in visited:
                    if dfs(successor.id):
                        return True
                elif successor.id in rec_stack:
                    return True
            rec_stack.discard(node_id)
            return False

        for nid in self._nodes:
            if nid not in visited:
                if dfs(nid):
                    return True
        return False

    def _cycle_involved_node_ids(self) -> set[str]:
        """Graph nodes on at least one directed cycle (back edges)."""
        if not self.has_cycle():
            return set()
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {nid: WHITE for nid in self._nodes}
        involved_nodes: set[str] = set()
        stack_path: list[str] = []

        def dfs(v: str) -> None:
            color[v] = GRAY
            stack_path.append(v)
            for succ in self.successors(v):
                w = succ.id
                if w not in self._nodes:
                    continue
                cw = color.get(w, WHITE)
                if cw == WHITE:
                    dfs(w)
                elif cw == GRAY:
                    i = stack_path.index(w)
                    involved_nodes.update(stack_path[i:])
            stack_path.pop()
            color[v] = BLACK

        for nid in self._nodes:
            if color[nid] == WHITE:
                dfs(nid)
        return involved_nodes

    def cycle_involved_fragment_ids(self) -> list[str]:
        """
        Return distinct fragment identifiers where at least one node
        belongs to a directed cycle.

        Used for the ``loop`` pattern (outside COVERED_PATTERNS): an unmapped FPd
        is duplicated in ``fps.fpd_policies`` of each listed fragment.
        """
        involved_nodes = self._cycle_involved_node_ids()
        frags: set[str] = set()
        for nid in involved_nodes:
            n = self._nodes.get(nid)
            if n and n.fragment_id:
                frags.add(n.fragment_id)
        return sorted(frags)

    def cycle_involved_activity_names(self) -> list[str]:
        """
        Activity names (``ActivityNode``) whose node belongs to a cycle.
        Used to document the ``loop`` FPd (readable target / label).
        """
        names: set[str] = set()
        for nid in self._cycle_involved_node_ids():
            n = self._nodes.get(nid)
            if isinstance(n, ActivityNode) and n.name:
                names.add(n.name)
        return sorted(names, key=lambda s: s.lower())

    # ── Representation ────────────────────────────

    def summary(self) -> dict:
        """Structural graph summary."""
        activities = [n for n in self._nodes.values() if isinstance(n, ActivityNode)]
        gateways = [n for n in self._nodes.values() if isinstance(n, GatewayNode)]
        events = [n for n in self._nodes.values() if isinstance(n, EventNode)]
        inter = self.inter_edges()

        fragments = set(
            n.fragment_id for n in self._nodes.values() if n.fragment_id
        )

        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "activities": len(activities),
            "gateways": len(gateways),
            "events": len(events),
            "inter_edges": len(inter),
            "fragments": sorted(fragments),
            "has_cycle": self.has_cycle(),
        }

    def __repr__(self):
        s = self.summary()
        return (
            f"BPMNGraph("
            f"nodes={s['nodes']}, edges={s['edges']}, "
            f"fragments={s['fragments']}, inter_edges={s['inter_edges']})"
        )


# ─────────────────────────────────────────────
#  Build from internal JSON model
# ─────────────────────────────────────────────

def build_graph_from_bp_model(bp_model: dict) -> BPMNGraph:
    """
    Build a BPMNGraph from the internal JSON model
    produced by BPMNParser (activities / gateways / flows).

    The expected model looks like:
        {
          "activities": [
             {"id": "...", "name": "...", "type": "task" | "event", "start": bool?, "end": bool?},
             ...
          ],
          "gateways": [
             {"id": "...", "name": "...", "type": "XOR" | "AND" | "OR" | ...},
             ...
          ],
          "flows": [
             {"id": "...", "from": <id or name>, "to": <id or name>, "type": "sequence" | "message", "gateway": <name?>},
             ...
          ]
        }

    The function is robust to flows referencing
    either BPMN IDs or names (after convert_ids_to_names).
    """
    graph = BPMNGraph()

    id_to_node_id: dict[str, str] = {}
    name_to_node_id: dict[str, str] = {}

    # 1) Create activity / event nodes
    for act in bp_model.get("activities", []):
        act_id = act.get("id") or act.get("name")
        act_name = act.get("name") or act_id
        act_type = (act.get("type") or "").lower()

        if act_type == "event":
            # Distinguish start / end / intermediate
            if act.get("start"):
                event_kind = "start"
            elif act.get("end"):
                event_kind = "end"
            else:
                event_kind = "intermediate"

            node = EventNode(
                id=act_id,
                name=act_name,
                node_type=NodeType.EVENT,
                event_kind=event_kind,
            )
        else:
            # By default, treat as an activity (task)
            node = ActivityNode(
                id=act_id,
                name=act_name,
                node_type=NodeType.ACTIVITY,
                role=act.get("role"),
                is_start=bool(act.get("start", False)),
                is_end=bool(act.get("end", False)),
            )

        graph.add_node(node)
        id_to_node_id[act_id] = act_id
        name_to_node_id[act_name] = act_id

    # 2) Create gateway nodes
    gw_type_map = {
        "XOR": GatewayType.XOR,
        "AND": GatewayType.AND,
        "OR": GatewayType.OR,
        "EVENT_BASED_EXCLUSIVE": GatewayType.EVENT_BASED_EXCLUSIVE,
        "EVENT_BASED_PARALLEL": GatewayType.EVENT_BASED_PARALLEL,
        "COMPLEX": GatewayType.COMPLEX,
    }

    for gw in bp_model.get("gateways", []):
        gw_id = gw.get("id") or gw.get("name")
        gw_name = gw.get("name") or gw_id
        gw_type_str = str(gw.get("type", "")).upper().replace("-", "_")
        gw_type = gw_type_map.get(gw_type_str)

        node = GatewayNode(
            id=gw_id,
            name=gw_name,
            node_type=NodeType.GATEWAY,
            gateway_type=gw_type,
        )
        graph.add_node(node)
        id_to_node_id[gw_id] = gw_id
        name_to_node_id[gw_name] = gw_id

    # 3) Create edges from flows
    def resolve_node_id(ref: str) -> Optional[str]:
        # ref may be a BPMN id or name (after convert_ids_to_names)
        if ref in id_to_node_id:
            return id_to_node_id[ref]
        if ref in name_to_node_id:
            return name_to_node_id[ref]
        return None

    for flow in bp_model.get("flows", []):
        raw_from = flow.get("from")
        raw_to = flow.get("to")
        src_id = resolve_node_id(raw_from) if raw_from is not None else None
        tgt_id = resolve_node_id(raw_to) if raw_to is not None else None

        # If endpoints cannot be resolved, skip this edge
        if not src_id or not tgt_id:
            continue

        flow_type = str(flow.get("type", "sequence")).lower()
        if flow_type == "message":
            edge_type = EdgeType.MESSAGE
        elif flow_type == "resource":
            edge_type = EdgeType.RESOURCE
        elif flow_type == "temporal":
            edge_type = EdgeType.TEMPORAL
        else:
            edge_type = EdgeType.SEQUENCE

        edge = Edge(
            id=flow.get("id") or f"{src_id}->{tgt_id}",
            source=src_id,
            target=tgt_id,
            edge_type=edge_type,
            dependency_type=DependencyType.CONTROL,
            gateway_id=flow.get("gateway"),
        )

        graph.add_edge(edge)

    return graph


