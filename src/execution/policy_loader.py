"""Charge les politiques ODRL exportées (FPa/FPd) et les métadonnées de fragments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agents.Agent_4.odrl_deterministic_templates import FragmentPolicySet


@dataclass
class PolicyBundle:
    """Ensemble prêt pour l'exécution : fragments + politiques indexées."""

    scenario_id: str
    fragments_enhanced: dict[str, Any]
    bp_model: dict[str, Any]
    fp_results: dict[str, FragmentPolicySet] = field(default_factory=dict)
    policies_flat: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fragment_ids_in_order(self) -> list[str]:
        ordering = self.fragments_enhanced.get("ordering") or {}
        return list(ordering.get("fragment_ids_in_process_order") or [])

    @property
    def global_activity_order(self) -> list[str]:
        ordering = self.fragments_enhanced.get("ordering") or {}
        return list(ordering.get("global_activity_order") or [])

    def fragment_for_activity(self, activity: str) -> Optional[str]:
        per = (self.fragments_enhanced.get("ordering") or {}).get(
            "activity_order_per_fragment"
        ) or {}
        for fid, acts in per.items():
            if activity in acts:
                return fid
        return None


def _rule_id_from_uri(uri: str) -> str:
    if not uri:
        return ""
    return uri.rstrip("/").split("/")[-1]


def _activity_from_asset_target(target: str) -> Optional[str]:
    if not target or "/asset/" not in target:
        return None
    return target.rstrip("/").split("/")[-1]


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_policies_from_export_dir(
    policies_dir: str | Path,
    *,
    scenario_id: str = "",
) -> dict[str, FragmentPolicySet]:
    """Lit les .jsonld sous ``policies_dir/<fragment_id>/``."""
    root = Path(policies_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dossier de politiques introuvable : {root}")

    results: dict[str, FragmentPolicySet] = {}
    for frag_dir in sorted(root.iterdir()):
        if not frag_dir.is_dir():
            continue
        fid = frag_dir.name
        fps = FragmentPolicySet(fragment_id=fid)
        for fp in sorted(frag_dir.glob("*.jsonld")):
            pol = load_json(fp)
            name = fp.name
            if name.startswith("FPa_"):
                pol["_type"] = "FPa"
                pol["_fragment_id"] = fid
                fps.fpa_policies.append(pol)
            elif name.startswith("FPd_"):
                pol["_type"] = "FPd"
                pol["_fragment_id"] = fid
                fps.fpd_policies.append(pol)
            else:
                if "permission" in pol and any(
                    _is_enable_action(a) for a in _iter_actions(pol)
                ):
                    pol["_type"] = "FPd"
                else:
                    pol["_type"] = "FPa"
                pol["_fragment_id"] = fid
                (fps.fpd_policies if pol["_type"] == "FPd" else fps.fpa_policies).append(
                    pol
                )
        if fps.all_policies():
            results[fid] = fps
    return results


def _iter_actions(pol: dict) -> list[dict]:
    out: list[dict] = []
    for rt in ("permission", "prohibition", "obligation"):
        for rule in pol.get(rt) or []:
            if isinstance(rule, dict):
                for act in rule.get("action") or []:
                    if isinstance(act, dict):
                        out.append(act)
    return out


def _is_enable_action(action: dict) -> bool:
    val = action.get("rdf:value")
    if isinstance(val, dict):
        return val.get("@id") in ("odrl:enable", "http://www.w3.org/ns/odrl/2.2/enable")
    return str(val) in ("odrl:enable", "enable")


def load_policy_bundle(
    scenario_dir: str | Path,
    *,
    policies_dir: Optional[str | Path] = None,
    scenario_id: str = "",
) -> PolicyBundle:
    """
    Charge ``fragments_enhanced.json``, ``bp_global.json`` et les politiques ODRL.

    ``policies_dir`` : si absent, tente ``<scenario>/odrl_policies`` puis ground truth benchmark.
    """
    base = Path(scenario_dir)
    sid = scenario_id or base.name

    fe_path = base / "fragments_enhanced.json"
    bp_path = base / "bp_global.json"
    if not fe_path.is_file():
        raise FileNotFoundError(f"fragments_enhanced.json manquant : {fe_path}")
    if not bp_path.is_file():
        raise FileNotFoundError(f"bp_global.json manquant : {bp_path}")

    fragments_enhanced = load_json(fe_path)
    bp_model = load_json(bp_path)

    pol_root: Optional[Path] = None
    if policies_dir:
        pol_root = Path(policies_dir)
    else:
        for candidate in (
            base / "odrl_policies",
            base / "odrl_fragment_policies",
            Path(__file__).resolve().parents[2]
            / "system_evaluation"
            / "benchmark"
            / "Ground_truth"
            / sid,
        ):
            if candidate.is_dir() and any(candidate.rglob("*.jsonld")):
                pol_root = candidate
                break

    fp_results: dict[str, FragmentPolicySet] = {}
    if pol_root and pol_root.is_dir():
        if any(p.is_dir() for p in pol_root.iterdir()):
            fp_results = load_policies_from_export_dir(pol_root, scenario_id=sid)
        else:
            fp_results = _load_flat_ground_truth(pol_root, sid)

    policies_flat: list[dict] = []
    for fps in fp_results.values():
        policies_flat.extend(fps.all_policies())

    return PolicyBundle(
        scenario_id=sid,
        fragments_enhanced=fragments_enhanced,
        bp_model=bp_model,
        fp_results=fp_results,
        policies_flat=policies_flat,
    )


def _load_flat_ground_truth(root: Path, scenario_id: str) -> dict[str, FragmentPolicySet]:
    """Ground truth : fichiers plats FPa_*.json / FPd_*.jsonld."""
    results: dict[str, FragmentPolicySet] = {}
    for fp in sorted(root.iterdir()):
        if not fp.is_file():
            continue
        if fp.suffix not in (".json", ".jsonld"):
            continue
        pol = load_json(fp)
        name = fp.stem
        if name.startswith("FPa_"):
            parts = name.split("_", 2)
            fid = parts[1] if len(parts) > 1 else "f1"
            pol["_type"] = "FPa"
            pol["_fragment_id"] = fid
            fps = results.get(fid) or FragmentPolicySet(fragment_id=fid)
            fps.fpa_policies.append(pol)
            results[fid] = fps
        elif name.startswith("FPd_"):
            pol["_type"] = "FPd"
            pol["_fragment_id"] = pol.get("_fragment_id") or "shared"
            fid = pol["_fragment_id"]
            fps = results.get(fid) or FragmentPolicySet(fragment_id=fid)
            fps.fpd_policies.append(pol)
            results[fid] = fps
    return results


def build_rule_activity_index(bundle: PolicyBundle) -> dict[str, str]:
    """Mappe un identifiant de règle (suffixe IRI) → slug d'activité BPMN."""
    index: dict[str, str] = {}
    for pol in bundle.policies_flat:
        if pol.get("_type") != "FPa":
            continue
        activity = pol.get("_activity")
        for rt in ("permission", "prohibition", "obligation"):
            for rule in pol.get(rt) or []:
                if not isinstance(rule, dict):
                    continue
                rid = _rule_id_from_uri(str(rule.get("uid") or ""))
                tgt_act = _activity_from_asset_target(str(rule.get("target") or ""))
                act = activity or tgt_act
                if rid and act:
                    index[rid] = act
                if tgt_act:
                    index.setdefault(tgt_act, tgt_act)
    return index


def build_activity_fpa_index(bundle: PolicyBundle) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for pol in bundle.policies_flat:
        if pol.get("_type") != "FPa":
            continue
        act = pol.get("_activity")
        if not act:
            for rt in ("permission", "prohibition", "obligation"):
                for rule in pol.get(rt) or []:
                    act = _activity_from_asset_target(str((rule or {}).get("target", "")))
                    if act:
                        break
                if act:
                    break
        if act:
            out.setdefault(act, []).append(pol)
    return out
