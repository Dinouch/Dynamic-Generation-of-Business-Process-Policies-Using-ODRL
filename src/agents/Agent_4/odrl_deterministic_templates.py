"""
Deterministic ODRL templates (JSON-LD) for Agent 4 — FPa / FPd projection.

Edit here to adjust policy shapes without changing multi-agent routing.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

# ─────────────────────────────────────────────
#  Patch keys (Agent 3 hints)
# ─────────────────────────────────────────────

TEMPLATE_KEY_MAP: dict[str, str] = {
    "constraint_operator": "operator",
    "constraint_left_operand": "leftOperand",
    "constraint_right_operand": "rightOperand",
    "action_id": "action",
    "rule_type": "rule_type",
}

# ─────────────────────────────────────────────
#  Helpers — project ODRL URIs
# ─────────────────────────────────────────────

BASE_URI = "http://example.com"


def uri_policy(uid: str) -> str:
    return f"{BASE_URI}/policy:{uid}"


def uri_rule(name: str) -> str:
    return f"{BASE_URI}/rules/{name.replace(' ', '_').replace('-', '_')}"


def uri_asset(name: str) -> str:
    return f"{BASE_URI}/asset/{name.replace(' ', '_').replace('-', '_')}"


def uri_collection(name: str) -> str:
    return f"{BASE_URI}/assets/{name.replace(' ', '_').replace('-', '_')}"


def uri_message(fi: str, fj: str) -> str:
    return f"{BASE_URI}/messages/msg_{fi}_{fj}"


def new_uid() -> str:
    return str(uuid.uuid4())[:8]


def is_deterministic_template_fpd(pol: dict[str, Any]) -> bool:
    """
    FPd produced by Agent 4 template (XOR/AND/OR/sequence/message) — not LLM unmapped.

    These policies have a fixed structure; only constraints may be corrected
    surgically (operator, @type, @context), never rewritten entirely.
    """
    if pol.get("_type") != "FPd":
        return False
    if pol.get("_unmapped_pattern"):
        return False
    return bool(pol.get("_gateway") or pol.get("_flow"))


# "enable" action string; for pyld / SHACL expansion
ODRL_ACTION_ENABLE: dict[str, str] = {"@id": "odrl:enable"}


def coerce_odrl_action_from_hint(action: object) -> object:
    """Normalize the action field (unmapped hint) for pyld / SHACL."""
    if action is None:
        return ODRL_ACTION_ENABLE
    if isinstance(action, dict):
        return action if action.get("@id") else ODRL_ACTION_ENABLE
    if isinstance(action, str):
        s = action.strip()
        if not s or s.lower() == "enable":
            return ODRL_ACTION_ENABLE
        if " " in s or "(" in s or "\n" in s or len(s) > 72:
            return ODRL_ACTION_ENABLE
        if s.startswith("odrl:"):
            return {"@id": s}
        if s.startswith("http://") or s.startswith("https://"):
            return {"@id": s}
        return ODRL_ACTION_ENABLE
    return ODRL_ACTION_ENABLE


_RE_BPMN_FRAGMENT_TOKEN = re.compile(r"^f\d+$", re.IGNORECASE)


def sanitize_unmapped_odrl_constraints(pol: dict[str, Any]) -> None:
    """Remove constraints where a BPMN fragment (f1, …) is misused as rightOperand."""
    for rt in ("permission", "prohibition", "obligation"):
        rules = pol.get(rt)
        if not isinstance(rules, list):
            continue
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            cons = rule.get("constraint")
            if cons is None:
                continue
            was_list = isinstance(cons, list)
            items = list(cons) if was_list else [cons]
            kept: list[dict[str, Any]] = []
            for c in items:
                if not isinstance(c, dict):
                    kept.append(c)
                    continue
                lo = str(c.get("leftOperand") or "").lower()
                ro = c.get("rightOperand")
                ro_s = ro.strip() if isinstance(ro, str) else ""
                if ro_s and _RE_BPMN_FRAGMENT_TOKEN.match(ro_s):
                    continue
                if lo == "spatial" and ro_s and _RE_BPMN_FRAGMENT_TOKEN.match(ro_s):
                    continue
                kept.append(c)
            if not kept:
                rule.pop("constraint", None)
            elif was_list:
                rule["constraint"] = kept
            else:
                rule["constraint"] = kept[0] if len(kept) == 1 else kept


def compact_jsonld_fpdm_inline_ids(serialized: str) -> str:
    """Post-serialization: compact @id blobs for FPd message exports."""
    s = re.sub(
        r'"rdf:value"\s*:\s*\{\s*\r?\n\s*"@id"\s*:\s*"odrl:enable"\s*\r?\n\s*\}',
        '"rdf:value": { "@id": "odrl:enable" }',
        serialized,
    )
    return re.sub(
        r'"rightOperand"\s*:\s*\{\s*\r?\n\s*"@id"\s*:\s*"odrl:policyUsage"\s*\r?\n\s*\}',
        '"rightOperand": { "@id": "odrl:policyUsage" }',
        s,
    )


# ─────────────────────────────────────────────
#  Output structures
# ─────────────────────────────────────────────


@dataclass
class FragmentPolicySet:
    """Fragment FP: FPa + FPd policy lists."""

    fragment_id: str
    fpa_policies: list[dict] = field(default_factory=list)
    fpd_policies: list[dict] = field(default_factory=list)

    def all_policies(self) -> list[dict]:
        return self.fpa_policies + self.fpd_policies

    def to_odrl(self) -> list[dict]:
        return [{k: v for k, v in p.items() if not k.startswith("_")} for p in self.all_policies()]

    def summary(self) -> dict:
        return {
            "fragment_id": self.fragment_id,
            "fpa_count": len(self.fpa_policies),
            "fpd_count": len(self.fpd_policies),
            "total": len(self.all_policies()),
        }


# ─────────────────────────────────────────────
#  FPa — templates
# ─────────────────────────────────────────────


def template_fpa_from_b2p(
    *,
    policy_uid: str,
    fragment_id: str,
    activity_name: str,
    source_b2p_uid: Optional[str],
    b2p: dict[str, Any],
) -> dict[str, Any]:
    fpa: dict[str, Any] = {
        "@context": "http://www.w3.org/ns/odrl.jsonld",
        "uid": policy_uid,
        "@type": b2p.get("@type", "Set"),
        "_fragment_id": fragment_id,
        "_activity": activity_name,
        "_type": "FPa",
        "_source_b2p": source_b2p_uid,
    }
    for rule_type in ("permission", "prohibition", "obligation"):
        if rule_type in b2p:
            fpa[rule_type] = b2p[rule_type]
    return fpa


def template_fpa_default_without_b2p(
    *,
    policy_uid: str,
    fragment_id: str,
    activity_name: str,
    rule_uid: str,
) -> dict[str, Any]:
    return {
        "@context": "http://www.w3.org/ns/odrl.jsonld",
        "uid": policy_uid,
        "@type": "Set",
        "_fragment_id": fragment_id,
        "_activity": activity_name,
        "_type": "FPa",
        "_source_b2p": None,
        "permission": [{
            "uid": rule_uid,
            "target": uri_asset(activity_name),
            "action": "trigger",
            "constraint": [{
                "leftOperand": "dateTime",
                "operator": "gteq",
                "rightOperand": {"@value": "2024-01-01", "@type": "xsd:date"},
            }],
        }],
    }


# ─────────────────────────────────────────────
#  FPd — templates (patterns / flows)
# ─────────────────────────────────────────────


def template_fpd_xor_pair(
    *,
    policy_uid: str,
    fragment_id: str,
    gw_name: str,
    act_i: str,
    act_j: str,
    ruleij_uri: str,
    ruleik_uri: str,
    perm_rule_ij_uid: str,
    perm_rule_ik_uid: str,
    condition_i: str,
    condition_j: str,
) -> dict[str, Any]:
    return {
        "@context": "http://www.w3.org/ns/odrl.jsonld",
        "uid": policy_uid,
        "@type": "Set",
        "_fragment_id": fragment_id,
        "_type": "FPd",
        "_gateway": "XOR",
        "_gateway_name": gw_name,
        "_activities": [act_i, act_j],
        "_conditions": [condition_i, condition_j],
        "permission": [
            {
                "uid": perm_rule_ij_uid,
                "target": ruleij_uri,
                "action": [{
                    "rdf:value": {"@id": "odrl:enable"},
                    "refinement": [{"leftOperand": "product", "operator": "eq",
                                   "rightOperand": condition_i}],
                }],
            },
            {
                "uid": perm_rule_ik_uid,
                "target": ruleik_uri,
                "action": [{
                    "rdf:value": {"@id": "odrl:enable"},
                    "refinement": [{"leftOperand": "product", "operator": "eq",
                                   "rightOperand": condition_j}],
                }],
            },
        ],
    }


def template_fpd_and_pair(
    *,
    policy_uid: str,
    fragment_id: str,
    gw_name: str,
    act_i: str,
    act_j: str,
    ruleij_uri: str,
    ruleik_uri: str,
    collection_uid: str,
    obligation_rule_uid: str,
) -> dict[str, Any]:
    return {
        "@context": "http://www.w3.org/ns/odrl.jsonld",
        "uid": policy_uid,
        "@type": "Set",
        "_fragment_id": fragment_id,
        "_type": "FPd",
        "_gateway": "AND",
        "_gateway_name": gw_name,
        "_activities": [act_i, act_j],
        "obligation": [{
            "uid": obligation_rule_uid,
            "target": {"@type": "AssetCollection", "uid": collection_uid},
            "action": ODRL_ACTION_ENABLE,
        }],
        "_asset_collection": [
            {"@type": "dc:Document", "@id": ruleij_uri,
             "dc:title": "concurrent rules", "odrl:partOf": collection_uid},
            {"@type": "dc:Document", "@id": ruleik_uri,
             "dc:title": "concurrent rules", "odrl:partOf": collection_uid},
        ],
    }


def template_fpd_or_pair(
    *,
    policy_uid: str,
    fragment_id: str,
    gw_name: str,
    act_i: str,
    act_j: str,
    ruleij_uri: str,
    ruleik_uri: str,
    obligation_ij_uid: str,
    obligation_ik_uid: str,
) -> dict[str, Any]:
    return {
        "@context": "http://www.w3.org/ns/odrl.jsonld",
        "uid": policy_uid,
        "@type": "Set",
        "_fragment_id": fragment_id,
        "_type": "FPd",
        "_gateway": "OR",
        "_gateway_name": gw_name,
        "_activities": [act_i, act_j],
        "obligation": [
            {
                "uid": obligation_ij_uid,
                "target": ruleij_uri,
                "action": ODRL_ACTION_ENABLE,
                "consequence": [{"target": ruleik_uri, "action": ODRL_ACTION_ENABLE}],
            },
            {
                "uid": obligation_ik_uid,
                "target": ruleik_uri,
                "action": ODRL_ACTION_ENABLE,
                "consequence": [{"target": ruleij_uri, "action": ODRL_ACTION_ENABLE}],
            },
        ],
    }


def template_fpd_flow_sequence(
    *,
    policy_uid: str,
    fragment_id: str,
    from_activity: str,
    to_activity: str,
    rule_source_uri: str,
    downstream_fpa_uid: str,
    permission_rule_uid: str,
) -> dict[str, Any]:
    return {
        "@context": "http://www.w3.org/ns/odrl.jsonld",
        "uid": policy_uid,
        "@type": "Set",
        "_fragment_id": fragment_id,
        "_type": "FPd",
        "_flow": "sequence",
        "_activities": [from_activity, to_activity],
        "permission": [{
            "uid": permission_rule_uid,
            "target": rule_source_uri,
            "action": ODRL_ACTION_ENABLE,
            "duty": [{"action": "nextPolicy", "uid": downstream_fpa_uid}],
            "constraint": [{
                "leftOperand": "event",
                "operator": "gt",
                "rightOperand": {"@id": "odrl:policyUsage"},
            }],
        }],
    }


def template_fpd_message(
    *,
    policy_uid: str,
    fragment_id: str,
    from_frag: str,
    to_frag: str,
    from_activity: str,
    to_activity: str,
    rule_source_uri: str,
    rule_target_uri: str,
    permission_rule_uid: str,
) -> dict[str, Any]:
    return {
        "@context": "http://www.w3.org/ns/odrl.jsonld",
        "uid": policy_uid,
        "@type": "Set",
        "_fragment_id": fragment_id,
        "_type": "FPd",
        "_flow": "message",
        "_from_fragment": from_frag,
        "_to_fragment": to_frag,
        "_activities": [from_activity, to_activity],
        "permission": [{
            "uid": permission_rule_uid,
            "target": uri_message(from_frag, to_frag),
            "assignee": rule_source_uri,
            "action": [{
                "rdf:value": "odrl:transfer",
                "refinement": [{
                    "leftOperand": "recipient",
                    "operator": "eq",
                    "rightOperand": rule_target_uri,
                }],
            }],
            "duty": [{
                "target": rule_target_uri,
                "action": [{
                    "rdf:value": {"@id": "odrl:enable"},
                    "refinement": [{
                        "leftOperand": "event",
                        "operator": "gt",
                        "rightOperand": {"@id": "odrl:policyUsage"},
                    }],
                }],
            }],
        }],
    }


def template_fallback_unmapped_fpd(
    *,
    policy_uid: str,
    rule_uid: str,
    fragment_id: str,
    pattern_type: str,
    gateway_name: Optional[str],
    hint_text: Optional[str],
    rule_kind: str,
    target_asset_uri: str,
) -> dict[str, Any]:
    rule_body: dict[str, Any] = {
        "uid": rule_uid,
        "target": target_asset_uri,
        "action": ODRL_ACTION_ENABLE,
    }
    out: dict[str, Any] = {
        "@context": "http://www.w3.org/ns/odrl.jsonld",
        "uid": policy_uid,
        "@type": "Set",
        "_fragment_id": fragment_id,
        "_type": "FPd",
        "_unmapped_pattern": pattern_type,
        "_gateway_name": gateway_name,
        "_hint_text": hint_text,
    }
    if hint_text:
        out["dct:description"] = hint_text
    out[rule_kind] = [rule_body]
    return out
