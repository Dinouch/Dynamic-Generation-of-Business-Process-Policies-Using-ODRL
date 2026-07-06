"""
structural_analyzer.py — Agent 1: Structural Analyzer

Single responsibility:
    Read raw BPMN → build formal graph G_exp → detect structural patterns.

This agent does NOT generate policies.
It prepares the structural truth for all downstream agents.

Output: EnrichedGraph containing
    - The formal graph G (BPMNGraph)
    - Detected patterns (forks, joins, loops, sync points)
    - B2P → activity mapping
    - Typed connections between activities
    - Global CG and local CL context for each fragment

─────────────────────────────────────────────────────────────────
  MULTI-AGENT LAYER
─────────────────────────────────────────────────────────────────
  This agent also exposes the messaging infrastructure
  shared by all pipeline agents:

      MessageType, AgentMessage

  These classes are imported by all other agents:
      from .structural_analyzer import AgentMessage, MessageType

  StructuralAnalyzer multi-agent methods:
      register_send_callback(fn)  — connect output (bus / Agent 3)
      send(msg)                   — emit an AgentMessage
      receive(msg)                — accept ANALYZE_GRAPH_TASK (async orchestrator)
      analyze_and_send()          — analyze() + GRAPH_READY (agent3) + CFP (exception handling) if needed

  The input BPMN graph is not modified during the pipeline: structural issues
  reported by Agent 3 result in rejection / report, not mutation of the analyzed model.
"""

import datetime
import re
import uuid
from urllib.parse import urlparse
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .pipeline_registry import BPMN_SEMANTICS, COVERED_PATTERNS

# Lazy fragmentation import in analyze() to avoid circular dependency
# and keep the agent usable when fragments are supplied as input.

from orchestration.graph import (
    BPMNGraph,
    ActivityNode,
    GatewayNode,
    Edge,
    NodeType,
    GatewayType,
    EdgeType,
    DependencyType,
)

from baseline.semantic_fragmenter import normalize_fragment_slug


# ─────────────────────────────────────────────
#  Multi-agent messaging infrastructure
#  (imported by all pipeline agents)
# ─────────────────────────────────────────────

class MessageType(Enum):
    """
    Message types exchanged between agents.

    Main flow:
        Agent 1 ──GRAPH_READY──────────► Agent 3
        Agent 1 ──CFP_UNMAPPED──────► Exception handling agent   (if uncovered patterns, after GRAPH_READY)
        Exception handling agent ──UNMAPPED_PROPOSALS──► Agent 3   (ACL PROPOSE)
        Exception handling agent ──REFORMULATED_PROPOSALS─► Agent 3   (ACL INFORM after REFORMULATE+agree)
        Agent 3 ──REFORMULATE──────────► Exception handling agent   (short loop)
        Agent 3 ──VALIDATION_DONE──────► Agent 4
        Agent 4 ──SYNTAX_AUDIT_REQUEST──► Agent 5   (syntax + global coherence)
        Agent 5 ──ODRL_SYNTAX_FAILURE──► Agent 4 → correction loop A4↔A5
        Agent 5 ──ODRL_VALID (pass 1)──► Agent 4 → POLICIES_READY → Agent 3
        Agent 3 ──SEMANTIC_VALIDATION_FAILURE──► Agent 4  (ACL FAILURE, then)
        Agent 3 ──SEMANTIC_CORRECTION──► Agent 4
        Agent 3 ──SEMANTIC_VALIDATED───► Agent 4 → SYNTAX_AUDIT_REQUEST → Agent 5 (pass 2)
        Agent 5 ──ODRL_VALID / ODRL_SYNTAX_ERROR──► Agent 4 → pipeline
    """
    GRAPH_READY        = "graph_ready"
    UNMAPPED_PROPOSALS = "unmapped_proposals"
    REFORMULATED_PROPOSALS = "reformulated_proposals"
    REFORMULATE        = "reformulate"
    VALIDATION_DONE    = "validation_done"
    POLICIES_READY     = "policies_ready"
    SEMANTIC_CORRECTION = "semantic_correction"
    SEMANTIC_VALIDATION_FAILURE = "semantic_validation_failure"
    SEMANTIC_VALIDATED = "semantic_validated"
    SYNTAX_CORRECTION  = "syntax_correction"
    ODRL_VALID         = "odrl_valid"
    ODRL_SYNTAX_ERROR  = "odrl_syntax_error"
    # FIPA-style extensions (ACL performatives mapped in ``legacy_adapter``)
    ANALYZE_GRAPH_TASK = "analyze_graph_task"
    CFP_UNMAPPED = "cfp_unmapped"
    ACCEPT_PROPOSAL_BATCH = "accept_proposal_batch"
    REJECT_PROPOSAL_BATCH = "reject_proposal_batch"
    DELEGATION_AGREE = "delegation_agree"
    DELEGATION_REFUSE = "delegation_refuse"
    SYNTAX_AUDIT_REQUEST = "syntax_audit_request"
    ODRL_SYNTAX_FAILURE = "odrl_syntax_failure"
    # FP batch already projected (merge templates + unmapped) — syntax/semantic audit without regeneration
    FP_BUNDLE_READY = "fp_bundle_ready"


@dataclass
class AgentMessage:
    """
    Message exchanged between two pipeline agents.

    Fields:
        sender    — AGENT_NAME of sender  (e.g. "agent1")
        recipient — AGENT_NAME of recipient (e.g. "exception_handling_agent")
        msg_type  — message type (MessageType)
        payload   — message-specific data
        timestamp — ISO generated automatically at creation
        loop_turn — current loop turn:
                      · 0 for the first message in a cycle
                      · incremented by the agent responding in a loop
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


# ─────────────────────────────────────────────
#  Agent output structures
# ─────────────────────────────────────────────


@dataclass
class StructuralPattern:
    """
    A structural pattern detected in the graph.

    Examples: fork-XOR, join-AND, synchronization point, loop.
    """

    pattern_type: str  # "fork_xor" | "fork_and" | "fork_or" | "join_and" | "join_xor" | "sync" | "loop"
    gateway_id: str  # ID of the gateway involved
    gateway_name: str
    involved_nodes: list[str]  # IDs of involved nodes
    fragment_id: Optional[str]  # Fragment involved (None if inter-fragment)
    description: str  # Human-readable pattern explanation
    involved_fragment_ids: list[str] = field(default_factory=list)
    involved_activity_names: list[str] = field(default_factory=list)


@dataclass
class UnmappedPattern:
    """
    BPMN structural pattern not covered by deterministic Agent 4 templates.
    Routed to the exception handling agent for LLM formulation.
    """

    pattern_type: str
    gateway_id: str
    gateway_name: str
    fragment_id: str
    description: str
    bpmn_semantic: str
    involved_fragment_ids: list[str] = field(default_factory=list)
    involved_activity_names: list[str] = field(default_factory=list)


@dataclass
class SemanticHint:
    """
    Semantic correction hint from Agent 3 for Agent 4 to patch generated ODRL.
    """

    policy_uid: str
    field_path: str
    issue: str
    suggested_fix: str
    odrl_template_key: str
    confidence: float


@dataclass
class ActivityB2PMapping:
    """
    Mapping between an activity and its applicable B2P policies.
    """

    activity_id: str
    activity_name: str
    fragment_id: str
    b2p_policy_ids: list[str]  # UIDs of B2P policies targeting this activity
    rule_types: list[str]  # ["permission", "prohibition", "obligation"]


@dataclass
class ConnectionInfo:
    """
    Structured information about a connection between two activities.
    """

    from_activity: str  # Source activity name
    to_activity: str  # Target activity name
    connection_type: str  # "sequence" | "xor" | "and" | "or" | "message"
    gateway_name: Optional[str]  # Gateway name if present
    condition: Optional[str]  # Branch condition
    is_inter: bool  # Inter-fragment ?
    from_fragment: str
    to_fragment: str


@dataclass
class FragmentContext:
    """
    Fragment context — used by downstream agents.

    Two variants:
        Global (CG): full process view
        Local  (CL): view limited to immediate neighborhood
    """

    fragment_id: str
    activities: list[str]  # Activity slugs (see fragments.json)
    gateways: list[str]  # Gateway names (slugs if from fragmenter)
    b2p_mappings: list[ActivityB2PMapping]
    connections: list[ConnectionInfo]  # Internal connections
    upstream_deps: list[ConnectionInfo]  # Incoming dependencies (inter)
    downstream_deps: list[ConnectionInfo]  # Outgoing dependencies (inter)

    # Global context (CG) — filled if global_context=True
    global_graph_summary: Optional[dict] = None
    all_inter_edges: Optional[list[ConnectionInfo]] = None

    @property
    def is_global(self) -> bool:
        return self.global_graph_summary is not None


@dataclass
class EnrichedGraph:
    """
    Final output of Agent 1.

    This is what downstream agents consume.
    Contains the formal graph + all structural analyses.
    """

    graph: BPMNGraph
    patterns: list[StructuralPattern]
    b2p_mappings: dict[str, ActivityB2PMapping]  # activity_id → mapping
    connections: list[ConnectionInfo]
    fragment_contexts: dict[str, FragmentContext]  # fragment_id → local context
    global_contexts: dict[str, FragmentContext]  # fragment_id → global context
    raw_bpmn: dict  # Original BPMN (reference)
    raw_b2p: list[dict]  # Original B2P policies
    unmapped_patterns: list[UnmappedPattern] = field(default_factory=list)
    # Reserved — B2P ambiguity LLM moved to Agent 3; always None.
    structural_llm_report: Optional[dict] = None


# ─────────────────────────────────────────────
#  Agent 1 — Structural Analyzer
# ─────────────────────────────────────────────


class StructuralAnalyzer:
    """
    Agent 1 of the multi-agent pipeline.

    Receives:
        - bp_model   : BPMN dict (activities, gateways, flows)
        - fragments  : fragment list (same format as ``fragments.json``), or None to compute them
        - b2p_policies : list of ODRL policies from the global BP
        - fragmentation_strategy : kept for API compatibility; auto fragmentation is LLM-driven

    If fragments is None, the agent uses LLM fragmentation (baseline) for fragments
    from bp_model. IDs are normalized to "f1", "f2", ... for consistency.

    Produces:
        - EnrichedGraph: enriched formal graph, ready for downstream agents

    Fully DETERMINISTIC analysis (graph, patterns, connections, heuristic B2P mapping).

    ── Multi-agent layer ──────────────────────────────────────────────
    In connected mode (full pipeline):
        analyzer.register_send_callback(agent3.receive)
        analyzer.analyze_and_send()   # starts the pipeline

    In standalone mode (unit tests):
        enriched_graph = analyzer.analyze()   # no callback required
    """

    AGENT_NAME = "agent1"

    def __init__(
        self,
        bp_model: dict,
        fragments: Optional[list[dict]] = None,
        b2p_policies: Optional[list[dict]] = None,
        fragmentation_strategy: str = "gateway",
        api_key: Optional[str] = None,
        *,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
    ):
        """
        Initialize the structural analyzer.

        Parameters
        ----------
        api_key, azure_* :
            Ignored — retained for backward compatibility; B2P ambiguity LLM runs in Agent 3.
        """
        _ = (api_key, azure_endpoint, azure_api_version, azure_deployment)
        self.bp_model = bp_model
        self.fragments = fragments
        self.b2p_policies = b2p_policies or []
        self.fragmentation_strategy = fragmentation_strategy
        self.graph = BPMNGraph()

        # Internal index: name → id (for flow resolution)
        self._name_to_id: dict[str, str] = {}
        self._fragment_of: dict[str, str] = {}  # activity slug (see fragments.json) → fragment_id

        # ── Multi-agent attributes ──────────────────────────────────
        self._on_send: Optional[Callable[[AgentMessage], None]] = None
        self._last_enriched_graph: Optional[EnrichedGraph] = None

    # ─────────────────────────────────────────
    #  Multi-agent layer
    # ─────────────────────────────────────────

    def register_send_callback(self, fn: Callable[[AgentMessage], None]) -> None:
        """
        Register callback to Agent 3.
        In pipeline mode: analyzer.register_send_callback(agent3.receive)
        """
        self._on_send = fn
        print(f"[Agent 1] Send callback registered → {fn}")

    def send(self, msg: AgentMessage) -> None:
        """Emit an AgentMessage to the registered recipient."""
        print(f"[Agent 1] ► SEND {msg}")
        if self._on_send:
            self._on_send(msg)
        else:
            print(f"[Agent 1][WARN] No callback — message '{msg.msg_type.value}' not sent.")

    def receive(self, msg: AgentMessage) -> None:
        """
        Entry point for incoming messages (async orchestrator).
        Accepts ``ANALYZE_GRAPH_TASK``; ignores the rest with a warning.
        """
        print(f"[Agent 1] ◄ RECEIVE {msg}")
        if msg.msg_type == MessageType.ANALYZE_GRAPH_TASK:
            self._handle_analyze_graph_task(msg)
        else:
            print(f"[Agent 1][WARN] Message '{msg.msg_type.value}' not handled — ignored.")

    def _handle_analyze_graph_task(self, msg: AgentMessage) -> None:
        """
        FIPA-style structural delegation from the pipeline orchestrator:
        AGREE to the REQUEST, then INFORM with ``GRAPH_READY`` payload.
        """
        request_id = msg.payload.get("request_message_id")
        self.send(
            AgentMessage(
                sender=self.AGENT_NAME,
                recipient="pipeline",
                msg_type=MessageType.DELEGATION_AGREE,
                payload={
                    "utterance": "I will analyze the BPMN graph and produce the enriched structural model.",
                    "_acl_ontology": "graph-structural",
                    "acl_in_reply_to": request_id,
                },
            )
        )
        enriched_graph = self.analyze()
        self._last_enriched_graph = enriched_graph
        self.send(
            AgentMessage(
                sender=self.AGENT_NAME,
                recipient="pipeline",
                msg_type=MessageType.GRAPH_READY,
                payload={
                    "enriched_graph": enriched_graph,
                    "fragment_ids": list(enriched_graph.fragment_contexts.keys()),
                    "reanalysis": False,
                    "affected_fragments": None,
                    "structural_llm_report": getattr(
                        enriched_graph, "structural_llm_report", None
                    ),
                    "utterance": "Structural analysis complete; enriched graph is ready.",
                },
                loop_turn=msg.loop_turn,
            )
        )
        self._emit_cfp_if_unmapped(enriched_graph, loop_turn=msg.loop_turn)

    def _emit_cfp_if_unmapped(self, enriched_graph: EnrichedGraph, *, loop_turn: int) -> None:
        """
        Call for proposals (CFP) to the exception handling agent if BPMN patterns are not covered by templates.
        Deterministic — no LLM: agent 1 already filled ``unmapped_patterns`` during analysis.
        """
        unmapped = list(getattr(enriched_graph, "unmapped_patterns", []) or [])
        if not unmapped:
            return
        print(
            f"[Agent 1] {len(unmapped)} uncovered pattern(s) — CFP to exception handling agent "
            "(formulation unmapped)"
        )
        cfp_call_id = uuid.uuid4().hex
        self.send(
            AgentMessage(
                sender=self.AGENT_NAME,
                recipient="exception_handling_agent",
                msg_type=MessageType.CFP_UNMAPPED,
                payload={
                    "enriched_graph": enriched_graph,
                    "utterance": (
                        "Call for proposals: the following structural patterns are not covered by "
                        "deterministic templates — propose ODRL fragment-policy hints with "
                        "conditions and justification."
                    ),
                    "unmapped_pattern_count": len(unmapped),
                    "cfp_call_id": cfp_call_id,
                    "_acl_reply_with": cfp_call_id,
                },
                loop_turn=loop_turn,
            )
        )

    def analyze_and_send(self) -> None:
        """
        Connected version of analyze() for pipeline mode.
        Runs full analysis then emits GRAPH_READY to Agent 3.

        In standalone mode (tests), use analyze() directly.
        """
        enriched_graph = self.analyze()
        self._last_enriched_graph = enriched_graph
        self.send(AgentMessage(
            sender    = self.AGENT_NAME,
            recipient = "agent3",
            msg_type  = MessageType.GRAPH_READY,
            payload   = {
                "enriched_graph":         enriched_graph,
                "fragment_ids":           list(enriched_graph.fragment_contexts.keys()),
                "reanalysis":             False,
                "affected_fragments":     None,
                "structural_llm_report":  getattr(
                    enriched_graph, "structural_llm_report", None
                ),
            },
        ))
        self._emit_cfp_if_unmapped(enriched_graph, loop_turn=0)

    # ─────────────────────────────────────────
    #  Main entry point
    # ─────────────────────────────────────────

    def analyze(self) -> EnrichedGraph:
        """
        Run full analysis.

        Steps:
            0. If fragments not supplied: compute via LLM fragmentation
            1. Build formal graph G from BPMN
            2. Map B2P policies to activities
            3. Detect structural patterns
            4. Build ConnectionInfo
            5. Build local CL and global CG contexts
        """
        print("[Agent 1] Structural Analyzer — starting analysis")

        # Step 0 — Compute fragments if not supplied
        if self.fragments is None:
            self.fragments = self._compute_fragments()
            print(f"[Agent 1] Fragments computed (LLM): {len(self.fragments)}")

        # Step 1 — Build formal graph
        self._build_graph()
        print(f"[Agent 1] Graph built: {self.graph}")

        # Step 2 — B2P → activity mapping
        b2p_mappings = self._map_b2p_to_activities()
        print(f"[Agent 1] B2P mappings: {len(b2p_mappings)} activities mapped")

        # Step 3 — Pattern detection (+ unmapped vs pipeline_registry)
        patterns, unmapped_patterns = self._detect_patterns()
        print(f"[Agent 1] Patterns detected: {len(patterns)}")
        for p in patterns:
            print(f"    -> [{p.pattern_type}] {p.gateway_name} : {p.description}")
        print(f"[Agent 1] Uncovered patterns (unmapped): {len(unmapped_patterns)}")

        # Step 4 — Build ConnectionInfo
        connections = self._build_connections()
        print(f"[Agent 1] Connections analyzed: {len(connections)}")

        # Step 5 — Local and global contexts per fragment
        local_contexts = self._build_local_contexts(b2p_mappings, connections)
        global_contexts = self._build_global_contexts(b2p_mappings, connections)
        print(f"[Agent 1] Contexts built for {len(local_contexts)} fragments")

        return EnrichedGraph(
            graph=self.graph,
            patterns=patterns,
            b2p_mappings=b2p_mappings,
            connections=connections,
            fragment_contexts=local_contexts,
            global_contexts=global_contexts,
            raw_bpmn=self.bp_model,
            raw_b2p=self.b2p_policies,
            unmapped_patterns=unmapped_patterns,
            structural_llm_report=None,
        )

    def _compute_fragments(self) -> list[dict]:
        """
        Compute fragments via ``SemanticFragmenter`` (LLM + BPMN gateways).
        Normalize ids to "f1", "f2", ... for test consistency.
        """
        from baseline.semantic_fragmenter import SemanticFragmenter

        fragmenter = SemanticFragmenter(self.bp_model)
        raw_fragments = fragmenter.fragment_process(strategy=self.fragmentation_strategy)
        # Normalize ids (SemanticFragmenter.fragment_process returns 0, 1, 2...)
        normalized = []
        for i, frag in enumerate(raw_fragments):
            f = dict(frag)
            f["id"] = f"f{i + 1}"
            normalized.append(f)
        return normalized

    # ─────────────────────────────────────────
    #  Step 1 — Build formal graph
    # ─────────────────────────────────────────

    def _build_graph(self) -> None:
        """Transform raw BPMN into formal graph G = (V, E)."""

        # Build name → fragment_id mapping
        for i, frag in enumerate(self.fragments):
            frag_id = frag.get("id", f"f{i}")
            frag["id"] = frag_id
            for act_name in frag.get("activities", []):
                self._fragment_of[normalize_fragment_slug(act_name)] = frag_id

        # ── Activity nodes ──
        for act in self.bp_model.get("activities", []):
            node_id = f"act_{act['name'].replace(' ', '_')}"
            frag_id = self._fragment_of.get(normalize_fragment_slug(act["name"]))

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

        # ── Gateway nodes ──
        gw_type_map = {
            "XOR": GatewayType.XOR,
            "AND": GatewayType.AND,
            "OR": GatewayType.OR,
            "EVENT_BASED_EXCLUSIVE": GatewayType.EVENT_BASED_EXCLUSIVE,
            "EVENT_BASED_PARALLEL": GatewayType.EVENT_BASED_PARALLEL,
            "COMPLEX": GatewayType.COMPLEX,
        }
        for gw in self.bp_model.get("gateways", []):
            node_id = f"gw_{gw['name'].replace(' ', '_')}"

            gw_type_str = str(gw.get("type", "XOR")).upper().replace("-", "_")
            gw_type = gw_type_map.get(gw_type_str, GatewayType.XOR)

            node = GatewayNode(
                id=node_id,
                name=gw["name"],
                node_type=NodeType.GATEWAY,
                gateway_type=gw_type,
                fragment_id=None,
            )
            self.graph.add_node(node)
            self._name_to_id[gw["name"]] = node_id

        # ── Edges (flows) ──
        for i, flow in enumerate(self.bp_model.get("flows", [])):
            from_name = flow["from"]
            to_name = flow["to"]
            flow_type = flow.get("type", "sequence").lower()

            from_id = self._name_to_id.get(from_name)
            to_id = self._name_to_id.get(to_name)

            if not from_id or not to_id:
                print(f"[Agent 1][WARN] Flow ignored: '{from_name}' → '{to_name}' (node not found)")
                continue

            # Edge type
            edge_type = EdgeType.MESSAGE if flow_type == "message" else EdgeType.SEQUENCE

            # Dependency type based on gateway
            dep_type = DependencyType.CONTROL
            if flow_type == "message":
                dep_type = DependencyType.MESSAGE

            # Associated gateway
            gw_id = None
            gw_name = flow.get("gateway")
            if gw_name and gw_name in self._name_to_id:
                gw_id = self._name_to_id[gw_name]

            cond = flow.get("condition") or flow.get("conditionExpression")
            edge_meta: dict = {}
            if flow.get("isDefault") or flow.get("default"):
                edge_meta["is_default"] = True
            if flow.get("conditionExpression"):
                edge_meta["condition_expression"] = flow.get("conditionExpression")

            edge = Edge(
                id=f"e{i}_{from_name.replace(' ', '_')}_{to_name.replace(' ', '_')}",
                source=from_id,
                target=to_id,
                edge_type=edge_type,
                dependency_type=dep_type,
                gateway_id=gw_id,
                condition=cond,
                metadata=edge_meta,
            )
            self.graph.add_edge(edge)

        # Assign gateways to fragments based on neighboring activities
        self._assign_gateways_to_fragments()

    def _assign_gateways_to_fragments(self) -> None:
        """
        Assign each gateway to the fragment of its majority source activity.
        A gateway without an assigned fragment is considered inter-fragment.
        """
        for node in self.graph.all_nodes():
            if not isinstance(node, GatewayNode):
                continue
            if node.fragment_id:
                continue

            # Find predecessor fragment
            pred_fragments = [p.fragment_id for p in self.graph.predecessors(node.id) if p.fragment_id]
            if pred_fragments:
                node.fragment_id = pred_fragments[0]

    # ─────────────────────────────────────────
    #  Step 2 — B2P → activity mapping
    # ─────────────────────────────────────────

    def _map_b2p_to_activities(self) -> dict[str, ActivityB2PMapping]:
        """
        For each activity, find B2P policies that target it.

        Matching strategy:
            - Exact match on activity name in target URI
            - Partial match (normalized name)
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
                        # Matching: activity name appears in target URI
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
        Check whether the activity is targeted by the ODRL URI (heuristic).

        - Historical case: normalized activity name appears in normalized URI
          (suited when URI is long or repeats the full label).
        - Often the URI is shorter (slug): test whether the last path segment is
          contained **in** the activity name (longer BPMN label).
        - Multiple BPMN words: if at least half of significant tokens
          (length ≥ 3) appear in the target, accept (compact URI with
          a few common keywords).

        Pure **synonyms** (different vocabulary with no shared text) are not
        detected here; they remain for the LLM (orphan targets).
        """
        normalized_name = activity_name.lower().replace(" ", "_").replace("-", "_")
        normalized_target = target_uri.lower().replace("-", "_").replace("/", "_")
        if normalized_name in normalized_target:
            return True

        # Last path segment (often short slug) ⊂ long activity name
        try:
            path = (urlparse(target_uri).path or "").rstrip("/")
            if path:
                last_seg = path.split("/")[-1].lower().replace("-", "_")
                if len(last_seg) >= 3 and last_seg in normalized_name:
                    return True
        except Exception:
            pass

        # Multiple BPMN words; URI contains only part of the words
        tokens = re.split(r"[\s_\-]+", activity_name.lower())
        tokens = [t for t in tokens if len(t) >= 3]
        if len(tokens) >= 2:
            hits = sum(1 for t in tokens if t in normalized_target)
            if hits >= max(2, (len(tokens) + 1) // 2):
                return True

        return False

    # ─────────────────────────────────────────
    #  Step 3 — Pattern detection
    # ─────────────────────────────────────────

    def _append_pattern_with_registry(
        self,
        patterns: list[StructuralPattern],
        unmapped: list[UnmappedPattern],
        pattern: StructuralPattern,
    ) -> None:
        """
        Append a structural pattern and, if its type is not in COVERED_PATTERNS,
        record an UnmappedPattern for the exception handling agent downstream.
        """
        patterns.append(pattern)
        if pattern.pattern_type not in COVERED_PATTERNS:
            sem = BPMN_SEMANTICS.get(
                pattern.pattern_type,
                "BPMN structural pattern without deterministic ODRL template coverage.",
            )
            unmapped.append(
                UnmappedPattern(
                    pattern_type=pattern.pattern_type,
                    gateway_id=pattern.gateway_id,
                    gateway_name=pattern.gateway_name,
                    fragment_id=pattern.fragment_id or "unknown",
                    description=pattern.description,
                    bpmn_semantic=sem,
                    involved_fragment_ids=list(pattern.involved_fragment_ids or []),
                    involved_activity_names=list(pattern.involved_activity_names or []),
                )
            )

    def _fork_pattern_type(self, gw: GatewayNode) -> str:
        """Map gateway fork node to canonical pattern_type string."""
        gt = gw.gateway_type
        if gt == GatewayType.EVENT_BASED_EXCLUSIVE:
            return "fork_event_based_exclusive"
        if gt == GatewayType.EVENT_BASED_PARALLEL:
            return "fork_event_based_parallel"
        if gt == GatewayType.COMPLEX:
            return "fork_complex"
        if gt:
            return f"fork_{gt.value.lower()}"
        return "fork_unknown"

    def _join_pattern_type(self, gw: GatewayNode) -> str:
        """Map gateway join node to canonical pattern_type string."""
        gt = gw.gateway_type
        if gt == GatewayType.OR:
            return "join_or"
        if gt:
            return f"join_{gt.value.lower()}"
        return "join_unknown"

    def _detect_patterns(self) -> tuple[list[StructuralPattern], list[UnmappedPattern]]:
        """
        Detect BPMN structural patterns on the formal graph and raw BPMN hints.

        Returns
        -------
        patterns
            All detected StructuralPattern records.
        unmapped
            Subset not covered by ``pipeline_registry.COVERED_PATTERNS`` (for the exception handling agent).
        """
        patterns: list[StructuralPattern] = []
        unmapped: list[UnmappedPattern] = []

        for node in self.graph.all_nodes():
            if not isinstance(node, GatewayNode):
                continue

            out_count = len(self.graph.out_edges(node.id))
            in_count = len(self.graph.in_edges(node.id))
            gw_type = node.gateway_type

            if out_count > 1:
                involved = [e.target for e in self.graph.out_edges(node.id)]
                pattern_type = self._fork_pattern_type(node)
                description = self._describe_fork(gw_type, node.name, involved)
                self._append_pattern_with_registry(
                    patterns,
                    unmapped,
                    StructuralPattern(
                        pattern_type=pattern_type,
                        gateway_id=node.id,
                        gateway_name=node.name,
                        involved_nodes=[node.id] + involved,
                        fragment_id=node.fragment_id,
                        description=description,
                    ),
                )

            if in_count > 1:
                involved = [e.source for e in self.graph.in_edges(node.id)]
                pattern_type = self._join_pattern_type(node)
                description = self._describe_join(gw_type, node.name, involved)
                self._append_pattern_with_registry(
                    patterns,
                    unmapped,
                    StructuralPattern(
                        pattern_type=pattern_type,
                        gateway_id=node.id,
                        gateway_name=node.name,
                        involved_nodes=involved + [node.id],
                        fragment_id=node.fragment_id,
                        description=description,
                    ),
                )

                if gw_type == GatewayType.AND:
                    self._append_pattern_with_registry(
                        patterns,
                        unmapped,
                        StructuralPattern(
                            pattern_type="sync",
                            gateway_id=node.id,
                            gateway_name=node.name,
                            involved_nodes=involved + [node.id],
                            fragment_id=node.fragment_id,
                            description=(
                                f"Mandatory synchronization point at '{node.name}'. "
                                f"All parallel branches must complete "
                                f"before continuing."
                            ),
                        ),
                    )

        if self.graph.has_cycle():
            cfids = self.graph.cycle_involved_fragment_ids()
            if not cfids:
                cfids = sorted(
                    {
                        n.fragment_id
                        for n in self.graph.all_nodes()
                        if getattr(n, "fragment_id", None)
                    }
                )
            cacts = self.graph.cycle_involved_activity_names()
            primary = cfids[0] if cfids else None
            loop_desc = "A cycle was detected in the graph. Check BPMN loops."
            if cacts:
                loop_desc = (
                    f"Control cycle on activities: {', '.join(cacts)}. "
                    + loop_desc
                )
            self._append_pattern_with_registry(
                patterns,
                unmapped,
                StructuralPattern(
                    pattern_type="loop",
                    gateway_id="",
                    gateway_name="cycle_detected",
                    involved_nodes=[],
                    fragment_id=primary,
                    description=loop_desc,
                    involved_fragment_ids=cfids,
                    involved_activity_names=cacts,
                ),
            )

        self._detect_flow_edge_patterns(patterns, unmapped)
        self._detect_activity_shape_patterns(patterns, unmapped)

        return patterns, unmapped

    def _detect_flow_edge_patterns(
        self,
        patterns: list[StructuralPattern],
        unmapped: list[UnmappedPattern],
    ) -> None:
        """
        Detect default and conditional sequence flows from edge metadata and topology.

        Inputs
        ------
        patterns, unmapped
            Lists updated in place.
        """
        for edge in self.graph.all_edges():
            src = self.graph.get_node(edge.source)
            tgt = self.graph.get_node(edge.target)
            if not isinstance(src, ActivityNode) or not isinstance(tgt, ActivityNode):
                continue
            frag = src.fragment_id or "unknown"
            if edge.metadata.get("is_default"):
                self._append_pattern_with_registry(
                    patterns,
                    unmapped,
                    StructuralPattern(
                        pattern_type="default_flow",
                        gateway_id=edge.id,
                        gateway_name=f"{src.name}→{tgt.name}",
                        involved_nodes=[src.id, tgt.id],
                        fragment_id=frag,
                        description=(
                            f"Default sequence flow from '{src.name}' to '{tgt.name}' "
                            f"(fallback when no other condition matches)."
                        ),
                    ),
                )
            if (
                edge.gateway_id is None
                and edge.condition
                and edge.edge_type == EdgeType.SEQUENCE
            ):
                self._append_pattern_with_registry(
                    patterns,
                    unmapped,
                    StructuralPattern(
                        pattern_type="conditional_flow",
                        gateway_id=edge.id,
                        gateway_name=f"{src.name}→{tgt.name}",
                        involved_nodes=[src.id, tgt.id],
                        fragment_id=frag,
                        description=(
                            f"Conditional sequence flow from '{src.name}' to '{tgt.name}' "
                            f"without an intermediate splitting gateway."
                        ),
                    ),
                )

    def _detect_activity_shape_patterns(
        self,
        patterns: list[StructuralPattern],
        unmapped: list[UnmappedPattern],
    ) -> None:
        """
        Detect advanced activity / subprocess constructs from the raw BPMN dict.

        Inputs
        ------
        patterns, unmapped
            Lists updated in place.
        """
        for act in self.bp_model.get("activities", []):
            name = act.get("name")
            if not name or name not in self._name_to_id:
                continue
            node_id = self._name_to_id[name]
            frag = self._fragment_of.get(normalize_fragment_slug(name), "unknown")
            meta = act.get("metadata") or {}

            def _add(ptype: str, desc: str) -> None:
                self._append_pattern_with_registry(
                    patterns,
                    unmapped,
                    StructuralPattern(
                        pattern_type=ptype,
                        gateway_id=node_id,
                        gateway_name=name,
                        involved_nodes=[node_id],
                        fragment_id=frag,
                        description=desc,
                    ),
                )

            if act.get("event_subprocess") or meta.get("event_subprocess"):
                _add(
                    "event_subprocess",
                    f"Activity '{name}' is modeled as or contains an event subprocess.",
                )
            if act.get("compensation") or meta.get("compensation"):
                _add(
                    "compensation",
                    f"Activity '{name}' has compensation boundary or handler semantics.",
                )
            lc = act.get("loopCharacteristics") or meta.get("loopCharacteristics")
            if isinstance(lc, dict):
                mi = lc.get("multiInstance") or lc.get("multiInstanceCharacteristics")
                if isinstance(mi, dict):
                    if mi.get("isSequential") is True:
                        _add(
                            "multi_instance_sequential",
                            f"Activity '{name}' uses sequential multi-instance execution.",
                        )
                    else:
                        _add(
                            "multi_instance_parallel",
                            f"Activity '{name}' uses parallel multi-instance execution.",
                        )
                else:
                    _add(
                        "loop_activity",
                        f"Activity '{name}' declares standard loop characteristics.",
                    )
            elif act.get("loop") or meta.get("loop"):
                _add("loop_activity", f"Activity '{name}' is marked as looping.")
            if act.get("adHocSubprocess") or meta.get("ad_hoc_subprocess"):
                _add(
                    "ad_hoc_subprocess",
                    f"Activity '{name}' is an ad-hoc subprocess.",
                )
            if act.get("callActivity") or meta.get("call_activity"):
                _add(
                    "call_activity",
                    f"Activity '{name}' is a call activity referencing another process.",
                )

    def _describe_fork(self, gw_type: Optional[GatewayType], gw_name: str, targets: list[str]) -> str:
        if gw_type == GatewayType.XOR:
            return (
                f"Exclusive fork at '{gw_name}': only one branch will activate among {len(targets)}. "
                f"Policies must implement enable(ruleA XOR ruleB)."
            )
        if gw_type == GatewayType.AND:
            return (
                f"Parallel fork at '{gw_name}': all {len(targets)} branches activate simultaneously. "
                f"Policies must implement enable(ruleA AND ruleB)."
            )
        if gw_type == GatewayType.OR:
            return (
                f"Inclusive fork at '{gw_name}': one or more branches among {len(targets)}. "
                f"Policies must implement enable(ruleA OR ruleB)."
            )
        if gw_type == GatewayType.EVENT_BASED_EXCLUSIVE:
            return f"Event-based exclusive fork at '{gw_name}': first event received."
        if gw_type == GatewayType.EVENT_BASED_PARALLEL:
            return f"Event-based parallel fork at '{gw_name}': all expected events."
        if gw_type == GatewayType.COMPLEX:
            return f"Complex fork at '{gw_name}': custom activation condition."
        return f"Unknown fork at '{gw_name}'."

    def _describe_join(self, gw_type: Optional[GatewayType], gw_name: str, sources: list[str]) -> str:
        if gw_type == GatewayType.AND:
            return (
                f"Synchronizing join at '{gw_name}': waits for completion of all {len(sources)} branches. "
                f"Deadlock risk if a policy blocks a branch."
            )
        if gw_type == GatewayType.XOR:
            return (
                f"Alternative join at '{gw_name}': first completed branch continues. "
                f"Other branches are ignored."
            )
        if gw_type == GatewayType.OR:
            return (
                f"Inclusive join at '{gw_name}': merge of incoming OR branches."
            )
        return f"Unknown join at '{gw_name}'."

    # ─────────────────────────────────────────
    #  Step 4 — ConnectionInfo
    # ─────────────────────────────────────────

    def _build_connections(self) -> list[ConnectionInfo]:
        """
        Build the list of all typed connections between activities.

        - Direct connections: activity → activity edges.
        - Gateway connections: activity → gateway → activity paths
          (to capture inter-fragment dependencies when the gateway
          is between two fragments, e.g. f1 --[GW]--> f2).
        """
        connections: list[ConnectionInfo] = []
        seen: set[tuple[str, str]] = set()  # (from_activity, to_activity) for deduplication

        # 1) Direct activity → activity edges
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

        # 2) activity → gateway → activity connections (inter-fragment dependencies)
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
        """Determine semantic connection type from an edge."""
        if edge.edge_type == EdgeType.MESSAGE:
            return "message"

        if edge.gateway_id:
            gw_node = self.graph.get_node(edge.gateway_id)
            if gw_node and isinstance(gw_node, GatewayNode):
                return gw_node.gateway_type.value.lower() if gw_node.gateway_type else "sequence"

        return "sequence"

    # ─────────────────────────────────────────
    #  Step 5 — CL and CG contexts
    # ─────────────────────────────────────────

    def _build_local_contexts(
        self,
        b2p_mappings: dict[str, ActivityB2PMapping],
        all_connections: list[ConnectionInfo],
    ) -> dict[str, FragmentContext]:
        """
        Build LOCAL context CL(Fi) for each fragment.

        CL(Fi) =
            - activities and gateways of Fi
            - B2P mappings of Fi activities
            - internal connections within Fi
            - dependencies with immediate neighbor fragments only
        """
        contexts: dict[str, FragmentContext] = {}

        for i, frag in enumerate(self.fragments):
            frag_id = frag.get("id", f"f{i}")

            activities = frag.get("activities", [])
            gateways = []
            gw_field = frag.get("gateways", [])
            if gw_field:
                # May be a list of names or gateway objects
                if isinstance(gw_field[0], str):
                    gateways = list(gw_field)
                else:
                    gateways = [gw.get("name", "") for gw in gw_field]

            # B2P mappings of this fragment's activities
            frag_mappings = [
                m for m in b2p_mappings.values() if m.fragment_id == frag_id
            ]

            # Internal connections
            internal_conns = [
                c for c in all_connections if c.from_fragment == frag_id and c.to_fragment == frag_id
            ]

            # Incoming dependencies (upstream)
            upstream = [c for c in all_connections if c.to_fragment == frag_id and c.is_inter]

            # Outgoing dependencies (downstream)
            downstream = [c for c in all_connections if c.from_fragment == frag_id and c.is_inter]

            contexts[frag_id] = FragmentContext(
                fragment_id=frag_id,
                activities=activities,
                gateways=gateways,
                b2p_mappings=frag_mappings,
                connections=internal_conns,
                upstream_deps=upstream,
                downstream_deps=downstream,
                global_graph_summary=None,  # Local → no global view
                all_inter_edges=None,
            )

        return contexts

    def _build_global_contexts(
        self,
        b2p_mappings: dict[str, ActivityB2PMapping],
        all_connections: list[ConnectionInfo],
    ) -> dict[str, FragmentContext]:
        """
        Build GLOBAL context CG(Fi) for each fragment.

        CG(Fi) = everything CL(Fi) contains +
            - full global graph summary
            - all inter-fragment edges
            - global ordering
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
                global_graph_summary=global_summary,  # Global view
                all_inter_edges=all_inter,  # All inter dependencies
            )

        return global_contexts