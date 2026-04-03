"""
Validation sémantique déterministe et au niveau du lot (Agent 3).

- Alignement FPa ↔ B2P (_source_b2p) : types de règles, opérateurs, parties, cible vs activité.
- Cohérence globale : contradictions ciblées, FPd vs graphe, références rule/policy, homonymie d'activités.

Sans validation syntaxique RDF/SHACL ; complète le juge LLM dans constraint_validator.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from orchestration.graph import ActivityNode

from ..structural_analyzer import EnrichedGraph

BASE_URI = "http://example.com"

RULE_TYPES = ("permission", "prohibition", "obligation")


def _uri_asset(activity_name: str) -> str:
    """Aligné sur policy_projection_agent.uri_asset (évite import circulaire)."""
    slug = activity_name.replace(" ", "_").replace("-", "_")
    return f"{BASE_URI}/asset/{slug}"


def _norm_operator(op: Any) -> str:
    if op is None:
        return ""
    if isinstance(op, dict):
        v = op.get("@id") or op.get("id")
        if isinstance(v, str):
            return v.split(":")[-1].strip().lower() if ":" in v else v.strip().lower()
        return str(v).strip().lower() if v is not None else ""
    return str(op).strip().lower()


def _collect_constraint_like_entries(rule: dict) -> list[dict]:
    """Contraintes directes + refinements sous action[]."""
    out: list[dict] = []
    if not isinstance(rule, dict):
        return out
    c = rule.get("constraint")
    if isinstance(c, list):
        out.extend(x for x in c if isinstance(x, dict))
    elif isinstance(c, dict):
        out.append(c)
    actions = rule.get("action")
    if actions is None:
        return out
    if not isinstance(actions, list):
        actions = [actions]
    for a in actions:
        if not isinstance(a, dict):
            continue
        refs = a.get("refinement")
        if isinstance(refs, list):
            out.extend(x for x in refs if isinstance(x, dict))
        elif isinstance(refs, dict):
            out.append(refs)
    return out


def _operators_sequence(rule: dict) -> list[str]:
    return [_norm_operator(x.get("operator")) for x in _collect_constraint_like_entries(rule)]


def _operator_field_paths(rule_type: str, rule_idx: int, rule: dict) -> list[str]:
    """Chemins JSON alignés sur la structure réelle de la règle (constraint vs refinement)."""
    paths: list[str] = []
    c = rule.get("constraint")
    if isinstance(c, list):
        for i, item in enumerate(c):
            if isinstance(item, dict) and "operator" in item:
                paths.append(f"{rule_type}[{rule_idx}].constraint[{i}].operator")
    elif isinstance(c, dict) and "operator" in c:
        paths.append(f"{rule_type}[{rule_idx}].constraint[0].operator")
    actions = rule.get("action")
    if not isinstance(actions, list):
        actions = [actions] if isinstance(actions, dict) else []
    for ai, a in enumerate(actions):
        if not isinstance(a, dict):
            continue
        refs = a.get("refinement")
        if isinstance(refs, list):
            for ri, item in enumerate(refs):
                if isinstance(item, dict) and "operator" in item:
                    paths.append(
                        f"{rule_type}[{rule_idx}].action[{ai}].refinement[{ri}].operator"
                    )
        elif isinstance(refs, dict) and "operator" in refs:
            paths.append(f"{rule_type}[{rule_idx}].action[{ai}].refinement[0].operator")
    return paths


def _activity_names_ambiguous(graph: EnrichedGraph) -> set[str]:
    """Noms d'activité partagés par plusieurs activity_id (homonymie)."""
    by_name: dict[str, set[str]] = defaultdict(set)
    for m in graph.b2p_mappings.values():
        by_name[m.activity_name].add(m.activity_id)
    return {name for name, ids in by_name.items() if len(ids) > 1}


def _graph_activity_names(graph: EnrichedGraph) -> set[str]:
    names: set[str] = set()
    for node in graph.graph.all_nodes():
        if isinstance(node, ActivityNode):
            names.add(node.name)
    return names


def _build_uid_indexes(fp_results: dict) -> tuple[set[str], set[str]]:
    policy_uids: set[str] = set()
    rule_uids: set[str] = set()
    for fps in fp_results.values():
        for pol in fps.all_policies():
            pu = pol.get("uid")
            if isinstance(pu, str) and pu:
                policy_uids.add(pu)
            for rt in RULE_TYPES:
                for rule in pol.get(rt) or []:
                    if isinstance(rule, dict):
                        ru = rule.get("uid")
                        if isinstance(ru, str) and ru:
                            rule_uids.add(ru)
    return policy_uids, rule_uids


def _collect_string_refs(obj: Any, out: list[str]) -> None:
    if isinstance(obj, str):
        if obj.startswith("http://") or obj.startswith("https://"):
            out.append(obj)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("_"):
                continue
            _collect_string_refs(v, out)
        return
    if isinstance(obj, list):
        for x in obj:
            _collect_string_refs(x, out)


def _fpd_ref_warnings(
    fpd: dict,
    frag_id: str,
    policy_uids: set[str],
    rule_uids: set[str],
) -> list[str]:
    """Vérifie que les IRIs de règle/politique référencées existent dans le lot."""
    refs: list[str] = []
    body = {k: v for k, v in fpd.items() if not str(k).startswith("_")}
    _collect_string_refs(body, refs)
    seen = set()
    w: list[str] = []
    uid_fpd = fpd.get("uid", "?")
    for r in refs:
        if r in seen:
            continue
        seen.add(r)
        if r in policy_uids or r in rule_uids:
            continue
        if "/rules/" in r or "/policy:" in r:
            w.append(
                f"[FPd {frag_id}] Politique '{uid_fpd}' référence un UID inconnu dans le lot : {r}"
            )
    return w


def deterministic_fpa_b2p_alignment(
    pol: dict,
    frag_id: str,
    b2p: dict,
    graph: EnrichedGraph,
) -> tuple[list[dict], list[str]]:
    """
    Compare une FPa (ODRL) à sa source B2P.

    Retourne (hints pour Agent 4, warnings informatifs).
    """
    hints: list[dict] = []
    warnings: list[str] = []
    pol_uid = pol.get("uid") or ""
    activity = pol.get("_activity") or ""

    expected_target = _uri_asset(activity) if activity else ""

    for rt in RULE_TYPES:
        b_rules = b2p.get(rt) if isinstance(b2p.get(rt), list) else []
        o_rules = pol.get(rt) if isinstance(pol.get(rt), list) else []
        if b_rules and not o_rules:
            warnings.append(
                f"FPa {pol_uid} : le B2P définit « {rt} » mais la politique générée ne contient pas ce type de règle."
            )
            continue
        if not b_rules:
            continue
        n = min(len(b_rules), len(o_rules))
        for i in range(n):
            br = b_rules[i] if isinstance(b_rules[i], dict) else {}
            orule = o_rules[i] if isinstance(o_rules[i], dict) else {}

            ot = orule.get("target")
            bt = br.get("target")
            if isinstance(bt, str) and isinstance(ot, str) and bt and ot and bt != ot:
                hints.append(
                    {
                        "policy_uid": pol_uid,
                        "field_path": f"{rt}[{i}].target",
                        "issue": "target ODRL diffère du target dans la règle B2P source.",
                        "suggested_fix": bt,
                        "odrl_template_key": "constraint_right_operand",
                        "confidence": 0.9,
                    }
                )
            elif (
                activity
                and expected_target
                and isinstance(ot, str)
                and ot
                and ot != expected_target
                and (not isinstance(bt, str) or not bt or bt == expected_target)
            ):
                hints.append(
                    {
                        "policy_uid": pol_uid,
                        "field_path": f"{rt}[{i}].target",
                        "issue": (
                            f"target ODRL ({ot}) ne correspond pas à l'asset canonique pour "
                            f"l'activité « {activity} »."
                        ),
                        "suggested_fix": expected_target,
                        "odrl_template_key": "constraint_right_operand",
                        "confidence": 0.85,
                    }
                )

            for field in ("assigner", "assignee"):
                bv, ov = br.get(field), orule.get(field)
                if isinstance(bv, str) and bv and isinstance(ov, str) and ov and bv != ov:
                    hints.append(
                        {
                            "policy_uid": pol_uid,
                            "field_path": f"{rt}[{i}].{field}",
                            "issue": f"{field} ne correspond pas à la source B2P.",
                            "suggested_fix": bv,
                            "odrl_template_key": "constraint_right_operand",
                            "confidence": 0.9,
                        }
                    )

            b_ops = _operators_sequence(br)
            o_ops = _operators_sequence(orule)
            if b_ops and o_ops and b_ops != o_ops:
                o_paths = _operator_field_paths(rt, i, orule)
                for j, (bo, oo) in enumerate(zip(b_ops, o_ops)):
                    if bo != oo:
                        fpath = o_paths[j] if j < len(o_paths) else f"{rt}[{i}].constraint[{j}].operator"
                        hints.append(
                            {
                                "policy_uid": pol_uid,
                                "field_path": fpath,
                                "issue": f"opérateur ODRL « {oo} » ≠ B2P « {bo} ».",
                                "suggested_fix": bo,
                                "odrl_template_key": "constraint_operator",
                                "confidence": 0.95,
                            }
                        )
                        break

    return hints, warnings


def batch_semantic_checks(
    fp_results: dict,
    graph: EnrichedGraph,
) -> tuple[list[dict], list[str]]:
    """
    Cohérence du lot : contradictions sur cibles FPa, FPd vs graphe, références, XOR.

    Les hints produits sont limités aux cas avec chemin ODRL clair ; le reste part en warnings.
    """
    hints: list[dict] = []
    warnings: list[str] = []

    policy_uids, rule_uids = _build_uid_indexes(fp_results)
    act_in_graph = _graph_activity_names(graph)
    amb_names = _activity_names_ambiguous(graph)
    if amb_names:
        warnings.append(
            "Homonymie d'activités (même nom, plusieurs activity_id) : "
            f"{', '.join(sorted(amb_names)[:12])}"
            f"{' …' if len(amb_names) > 12 else ''} — risque d'alignement FPa/FPd."
        )

    for frag_id, fps in fp_results.items():
        for pol in fps.fpa_policies:
            if pol.get("_source_b2p"):
                continue
            uid = pol.get("uid", "?")
            warnings.append(
                f"[{frag_id}] FPa « {uid} » sans _source_b2p — politique minimale / par défaut, "
                "non alignée à une entrée B2P explicite."
            )

    target_modes: dict[str, set[str]] = defaultdict(set)
    for frag_id, fps in fp_results.items():
        for pol in fps.fpa_policies:
            for rt in ("permission", "prohibition"):
                for rule in pol.get(rt) or []:
                    if not isinstance(rule, dict):
                        continue
                    t = rule.get("target")
                    if isinstance(t, str) and t:
                        target_modes[t].add(rt)
    for tgt, modes in target_modes.items():
        if "permission" in modes and "prohibition" in modes:
            warnings.append(
                f"Cohérence globale : la cible « {tgt} » a à la fois permission et prohibition dans les FPa."
            )

    seq_pairs = {
        (c.from_activity, c.to_activity)
        for c in graph.connections
        if c.connection_type in ("sequence", "message")
    }

    for frag_id, fps in fp_results.items():
        for fpd in fps.fpd_policies:
            uid = fpd.get("uid", "?")

            for act in fpd.get("_activities") or []:
                if act not in act_in_graph:
                    warnings.append(
                        f"[FPd {frag_id} « {uid} »] Activité « {act} » absente du graphe enrichi."
                    )

            if fpd.get("_flow") == "sequence":
                acts = fpd.get("_activities") or []
                if len(acts) >= 2:
                    pair = (acts[0], acts[1])
                    if pair not in seq_pairs:
                        warnings.append(
                            f"[FPd {frag_id} « {uid} »] Séquence {acts[0]} → {acts[1]} "
                            "non trouvée comme connexion sequence/message dans le graphe."
                        )

            if fpd.get("_gateway") == "XOR":
                conds = fpd.get("_conditions") or []
                if len(conds) >= 2 and conds[0] == conds[1]:
                    warnings.append(
                        f"[FPd {frag_id} « {uid} »] XOR : conditions identiques ({conds[0]!r})."
                    )

            warnings.extend(_fpd_ref_warnings(fpd, frag_id, policy_uids, rule_uids))

    return hints, warnings


def run_deterministic_semantic_validation(
    fp_results: dict,
    graph: EnrichedGraph,
) -> tuple[list[dict], list[str]]:
    """
    Point d'entrée : FPa↔B2P par politique, puis contrôles de lot.

    Returns
    -------
    hints
        Dicts compatibles avec SemanticHint / semantic_hints Agent 4.
    warnings
        Messages informatifs (homonymie, absences structurelles, etc.).
    """
    all_hints: list[dict] = []
    all_warnings: list[str] = []

    b2p_by_uid = {p.get("uid"): p for p in graph.raw_b2p if p.get("uid")}

    for frag_id, fps in fp_results.items():
        for pol in fps.fpa_policies:
            src = pol.get("_source_b2p")
            if not src:
                continue
            b2p = b2p_by_uid.get(src)
            if not b2p:
                all_warnings.append(
                    f"[{frag_id}] FPa « {pol.get('uid')} » : _source_b2p « {src} » introuvable dans raw_b2p."
                )
                continue
            h, w = deterministic_fpa_b2p_alignment(pol, frag_id, b2p, graph)
            all_hints.extend(h)
            all_warnings.extend(w)

    h2, w2 = batch_semantic_checks(fp_results, graph)
    all_hints.extend(h2)
    all_warnings.extend(w2)

    return all_hints, all_warnings


def merge_semantic_hints(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """Dédoublonne par (policy_uid, field_path, suggested_fix)."""
    seen: set[tuple[str, str, str]] = set()
    out: list[dict] = []
    for h in primary + secondary:
        key = (
            str(h.get("policy_uid", "")),
            str(h.get("field_path", "")),
            str(h.get("suggested_fix", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out
