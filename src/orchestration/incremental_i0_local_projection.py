"""
**I0** projection — local heuristics without relying on Agent 1 aggregated connections.

- **Gateway patterns** (fork XOR/AND/OR) remain those detected on the formal BPMN:
  ``PolicyProjectionAgent._fpd_from_pattern`` still requires real outgoing edges from
  ``enriched.graph``.
- **Connections**, **upstream** / **downstream**, and the **global connection summary**
  are realigned from:
    1. ``fragments_enhanced.json`` (``connections``, ``ordering`` lists) if present
       under ``src/dataset/<scenario_id>/``;
    2. otherwise trivial **sequential** edges following each fragment's ``activities``
       list in ``fragments.json``.
- **B2P → activity mappings** are **recomputed** by scanning ``raw_b2p`` with the
  same ``target`` heuristic as Agent 1 (no recursion, edge count cap).

Complex or poorly documented **nested** dependencies in enhanced may still be
missing compared to I1 (Agent 1 graph truth on all edges).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from dataclasses import replace
from urllib.parse import urlparse

from agents.structural_analyzer import (
    ActivityB2PMapping,
    ConnectionInfo,
    FragmentContext,
    EnrichedGraph,
)

from orchestration.graph import ActivityNode, BPMNGraph


def load_fragments_enhanced_optional(project_root: Path, scenario_id: str) -> dict[str, Any] | None:
    """Read ``fragments_enhanced.json`` if available."""
    path = Path(project_root).resolve() / "src" / "dataset" / scenario_id / "fragments_enhanced.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def name_matches_odrl_target(activity_name: str, target_uri: str) -> bool:
    """
    Same heuristic strategy as ``StructuralAnalyzer._name_matches_target``:
    avoids an instance dependency on the analyzer.
    """
    normalized_name = activity_name.lower().replace(" ", "_").replace("-", "_")
    normalized_target = target_uri.lower().replace("-", "_").replace("/", "_")
    if normalized_name in normalized_target:
        return True

    try:
        path_part = (urlparse(target_uri).path or "").rstrip("/")
        if path_part:
            last_seg = path_part.split("/")[-1].lower().replace("-", "_")
            if len(last_seg) >= 3 and last_seg in normalized_name:
                return True
    except Exception:
        pass

    tokens = re.split(r"[\s_\-]+", activity_name.lower())
    tokens = [t for t in tokens if len(t) >= 3]
    if len(tokens) >= 2:
        hits = sum(1 for t in tokens if t in normalized_target)
        if hits >= max(2, (len(tokens) + 1) // 2):
            return True

    return False


def build_b2p_mappings_from_raw_policies(graph: BPMNGraph, raw_b2p: list[dict[str, Any]]) -> dict[str, ActivityB2PMapping]:
    """
    For each activity node in the BPMN graph built during ``analyze()``,
    read B2P JSON policies and apply ``name_matches_odrl_target``.
    """
    mappings: dict[str, ActivityB2PMapping] = {}

    for node in graph.all_nodes():
        if not isinstance(node, ActivityNode):
            continue

        matched_policy_ids: list[str] = []
        matched_rule_types: list[str] = []

        for policy in raw_b2p or []:
            if not isinstance(policy, dict):
                continue
            policy_uid = policy.get("uid", "")

            for rule_type in ("permission", "prohibition", "obligation"):
                rules = policy.get(rule_type)
                if rules is None:
                    continue
                if not isinstance(rules, list):
                    rules = [rules]
                for rule in rules:
                    if not isinstance(rule, dict):
                        continue
                    target = str(rule.get("target", "") or "")
                    if name_matches_odrl_target(node.name, target):
                        if policy_uid and policy_uid not in matched_policy_ids:
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


def _gateways_slugs_from_fragment(frag: dict[str, Any]) -> list[str]:
    gw_field = frag.get("gateways") or []
    if not gw_field:
        return []
    if isinstance(gw_field[0], str):
        return list(gw_field)
    return [str(g.get("name", "") or "") for g in gw_field]


def _is_likely_gateway(name: str, activity_set: set[str]) -> bool:
    if not name or name in activity_set:
        return False
    n = name.lower()
    return any(
        m in n
        for m in (
            "xor-sid",
            "parallel-sid",
            "event-based-exclusive",
            "event-based-parallel",
            "exclusive-sid",
            "gateway-sid",
            "complex-sid",
        )
    )


def _row_fragments(row: dict[str, Any], fa: str, ta: str) -> tuple[str, str, bool]:
    ff = row.get("from_primary_fragment_id") or (row.get("from_fragment_ids") or ["unknown"])[0]
    tf = row.get("to_primary_fragment_id") or (row.get("to_fragment_ids") or ["unknown"])[0]
    ff = ff if isinstance(ff, str) else "unknown"
    tf = tf if isinstance(tf, str) else "unknown"
    scope = str(row.get("scope") or "")
    inter = scope == "inter" or (ff != tf and ff != "unknown" and tf != "unknown")
    return ff, tf, inter


def _make_conn(
    from_act: str,
    to_act: str,
    ff: str,
    tf: str,
    ctype: str,
    gw_name: str | None,
    condition: str | None,
) -> ConnectionInfo:
    inter = ff != tf
    return ConnectionInfo(
        from_activity=from_act,
        to_activity=to_act,
        connection_type=(ctype or "sequence").lower(),
        gateway_name=gw_name,
        condition=condition,
        is_inter=inter,
        from_fragment=ff,
        to_fragment=tf,
    )


def derive_activity_level_connections(
    enhanced: dict[str, Any] | None,
    fragments: list[dict[str, Any]],
    *,
    max_connections: int,
    max_pairs_per_gateway: int = 96,
) -> list[ConnectionInfo]:
    """Activity→activity edges without deep recursion."""
    activity_set: set[str] = set()
    frag_by_id: dict[str, dict[str, Any]] = {}

    for i, frag in enumerate(fragments or []):
        fid = frag.get("id") or f"f{i}"
        fp = dict(frag)
        fp["id"] = fid
        frag_by_id[fid] = fp
        for a in fp.get("activities") or []:
            if isinstance(a, str):
                activity_set.add(a)

    bucket: dict[tuple[str, str, str, str], ConnectionInfo] = {}

    def add_ci(ci: ConnectionInfo) -> None:
        if len(bucket) >= max_connections:
            return
        key = (ci.from_activity, ci.to_activity, ci.from_fragment, ci.to_fragment)
        if key in bucket:
            return
        bucket[key] = ci

    conn_rows = (enhanced or {}).get("connections") if enhanced else None
    rows_ok = isinstance(conn_rows, list) and len(conn_rows) > 0
    if rows_ok:
        ingress: dict[str, list[dict[str, Any]]] = {}
        egress: dict[str, list[dict[str, Any]]] = {}

        for row in conn_rows:
            if not isinstance(row, dict):
                continue
            fa, ta = row.get("from"), row.get("to")
            if not isinstance(fa, str) or not isinstance(ta, str):
                continue

            ctype = str(row.get("type") or "sequence")
            gw_nm = row.get("gateway")
            gw_nm = gw_nm if isinstance(gw_nm, str) else None
            cond = row.get("condition")
            cond = cond if isinstance(cond, str) else None
            ff, tf, _ = _row_fragments(row, fa, ta)

            if fa in activity_set and ta in activity_set:
                add_ci(_make_conn(fa, ta, ff, tf, ctype, gw_nm, cond))
                continue

            if fa in activity_set and _is_likely_gateway(ta, activity_set):
                ingress.setdefault(ta, []).append(row)
            elif _is_likely_gateway(fa, activity_set) and ta in activity_set:
                egress.setdefault(fa, []).append(row)

        for gw_key, incoming in ingress.items():
            outgoing_list = egress.get(gw_key) or []
            if not outgoing_list:
                continue
            pair_count = 0
            for r_in in incoming:
                if pair_count >= max_pairs_per_gateway or len(bucket) >= max_connections:
                    break
                fa_i = str(r_in.get("from"))
                ffi, _, _ = _row_fragments(r_in, fa_i, str(r_in.get("to")))
                for r_out in outgoing_list:
                    if pair_count >= max_pairs_per_gateway or len(bucket) >= max_connections:
                        break
                    ta_o = str(r_out.get("to"))
                    _, tft, _ = _row_fragments(r_out, str(r_out.get("from")), ta_o)
                    ff_merge = ffi or "unknown"
                    tf_merge = tft or "unknown"
                    ctype_o = str(r_out.get("type") or r_in.get("type") or "sequence")
                    gw_name = gw_key if _is_likely_gateway(gw_key, activity_set) else gw_key
                    cond_out = r_out.get("condition")
                    cond_out = cond_out if isinstance(cond_out, str) else (
                        r_in.get("condition") if isinstance(r_in.get("condition"), str) else None
                    )
                    add_ci(
                        _make_conn(
                            fa_i,
                            ta_o,
                            ff_merge,
                            tf_merge,
                            ctype_o,
                            gw_name if isinstance(gw_name, str) else None,
                            cond_out,
                        )
                    )
                    pair_count += 1

    ord_global = (((enhanced or {}).get("ordering") or {}).get("global_activity_order"))
    # Without ``connections`` rows in enhanced, derive a lightweight global ordering.
    if isinstance(ord_global, list) and len(bucket) < max_connections and not rows_ok:
        prev_ord: str | None = None
        for name in ord_global:
            if not isinstance(name, str) or name not in activity_set:
                prev_ord = None
                continue
            if prev_ord:
                fid_a = None
                fid_b = None
                for fid, fp in frag_by_id.items():
                    acts = fp.get("activities") or []
                    if prev_ord in acts:
                        fid_a = fid
                    if name in acts:
                        fid_b = fid
                if fid_a and fid_b:
                    add_ci(_make_conn(prev_ord, name, fid_a, fid_b, "sequence", None, None))
            prev_ord = name

    # Intra-fragment fallback: order from the activities list
    if len(bucket) < max_connections:
        for fid, fp in frag_by_id.items():
            seq = fp.get("activities") or []
            if not isinstance(seq, list):
                continue
            for i in range(len(seq) - 1):
                if len(bucket) >= max_connections:
                    break
                a0, a1 = seq[i], seq[i + 1]
                if isinstance(a0, str) and isinstance(a1, str):
                    add_ci(_make_conn(a0, a1, fid, fid, "sequence", None, None))

    return list(bucket.values())


def _build_local_fragment_contexts(
    fragments: list[dict[str, Any]],
    b2p_mappings: dict[str, ActivityB2PMapping],
    all_connections: list[ConnectionInfo],
) -> dict[str, FragmentContext]:
    contexts: dict[str, FragmentContext] = {}

    for i, frag in enumerate(fragments or []):
        frag_id = frag.get("id", f"f{i}")

        activities = list(frag.get("activities") or [])
        gateways = _gateways_slugs_from_fragment(frag)

        frag_mappings = [m for m in b2p_mappings.values() if m.fragment_id == frag_id]

        internal_conns = [
            c for c in all_connections if c.from_fragment == frag_id and c.to_fragment == frag_id
        ]
        upstream = [c for c in all_connections if c.to_fragment == frag_id and c.is_inter]
        downstream = [c for c in all_connections if c.from_fragment == frag_id and c.is_inter]

        contexts[frag_id] = FragmentContext(
            fragment_id=frag_id,
            activities=activities,
            gateways=gateways,
            b2p_mappings=frag_mappings,
            connections=internal_conns,
            upstream_deps=upstream,
            downstream_deps=downstream,
            global_graph_summary=None,
            all_inter_edges=None,
        )

    return contexts


def apply_i0_local_projection(
    enriched: EnrichedGraph,
    fragments: list[dict[str, Any]],
    scenario_id: str,
    project_root: Path | str,
    *,
    max_connections: int = 500,
) -> EnrichedGraph:
    """
    Realign B2P, connections, and fragment contexts for **I0**; leave Agent 1 graph /
    patterns unchanged for XOR/AND/OR FPd generation already wired on the BPMN.
    """
    root = Path(project_root).resolve()
    enhanced_doc = load_fragments_enhanced_optional(root, scenario_id)
    connections = derive_activity_level_connections(
        enhanced_doc,
        fragments,
        max_connections=max_connections,
    )
    raw_b2p = enriched.raw_b2p if isinstance(enriched.raw_b2p, list) else []
    b2p_mappings = build_b2p_mappings_from_raw_policies(enriched.graph, raw_b2p)
    contexts = _build_local_fragment_contexts(fragments, b2p_mappings, connections)

    return replace(
        enriched,
        b2p_mappings=b2p_mappings,
        connections=connections,
        fragment_contexts=contexts,
        global_contexts={},
    )
