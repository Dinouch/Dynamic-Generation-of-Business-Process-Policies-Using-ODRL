"""
Baseline I0 — **100% LLM** ODRL generation (one prompt, entire scenario).

- Raw inputs: ``b2p_policies``, ``fragments.json``, ``fragments_enhanced.json``, ``bp_model``.
- **No** Agent 1, **no** Agent 4 deterministic templates (``template_fpa_*``, ``_fpd_from_pattern``, etc.).
- One model call per scenario (optional per-fragment fallback if ``I0_PER_FRAGMENT=1``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Optional

from openai import AzureOpenAI, OpenAI

from .odrl_deterministic_templates import BASE_URI, FragmentPolicySet, new_uid, uri_policy

_I0_SYSTEM = (
    "You are an ODRL 2.2 / BPMN fragment-policy expert. "
    "You produce ONLY valid JSON (no markdown fences). "
    "Use https://www.w3.org/TR/odrl-vocab/ vocabulary and "
    '"@context": "http://www.w3.org/ns/odrl.jsonld". '
    "You must invent every policy yourself from the raw inputs — "
    "there is NO template engine, NO pre-built rule catalog, and NO deterministic filler."
)

_I0_USER_ALL_FRAGMENTS = """# Increment I0 — generate the FULL ODRL policy set in ONE response

You receive **all** raw inputs for one business process scenario. Your job is to output **every**
fragment policy (FPa + FPd) that a complete ODRL pipeline would need — **from scratch**, using
reasoning only.

## What to generate (for ALL fragments f1…fN)

1. **FPa** — typically one ODRL policy per BPMN activity, aligned with B2P targets where possible.
2. **FPd** — all dependency / control-flow policies implied by the data:
   - XOR / AND / OR gateway branches
   - intra-fragment **sequence** dependencies
   - inter-fragment **message** flows (`scope: inter`)
   - loops, event-based exclusive gateways, any unmapped-looking pattern — still express as ODRL
3. Cover **supported and unmapped-looking** structures; do not skip "simple" cases.

## Hard constraints

- Do NOT reference or mimic any internal template names (no `template_fpa_*`, no automatic XOR pairing).
- Each policy: `"@type": "Set"`, unique `"uid"` under `{base_uri}/policy:...`
- Each rule: `"uid"`, `"target"` (IRI), `"action"` with ODRL vocabulary (`odrl:execute`, `odrl:transfer`, …)
- Tag each policy with `"_fragment_id": "f1"` (etc.), `"_type": "FPa"|"FPd"`, `"_subtype": "sequence"|"message"|"xor"|"and"|"or"|"loop"|"other"`

## Raw inputs (JSON)

### Business process model (activities / metadata)
```json
{bp_model_json}
```

### Fragment list (`fragments.json`)
```json
{fragments_json}
```

### Enhanced fragments (connections, gateways, ordering) — primary source for FPd
```json
{enhanced_json}
```

### B2P policies (activity-level)
```json
{b2p_json}
```

## Output format (JSON only)

{{
  "process_summary": "one short paragraph",
  "fragments": {{
    "f1": [ {{ ... full ODRL policy objects with _fragment_id, _type, _subtype ... }}, ... ],
    "f2": [ ... ],
    ...
  }}
}}

Include **every** fragment id present in the inputs. Empty list only if truly no policy applies.
"""


def _parse_llm_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def _build_llm_client(
    *,
    api_key: str | None,
    azure_endpoint: str | None,
    azure_api_version: str | None,
    azure_deployment: str | None,
) -> tuple[Any, bool, str]:
    """Return ``(client, use_azure, model_name)``."""
    model_default = os.environ.get("OPENAI_MODEL", "gpt-4o")
    key_azure = (api_key or os.environ.get("AZURE_OPENAI_KEY") or os.environ.get("AZURE_OPENAI_API_KEY") or "").strip()
    endpoint = (azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
    if key_azure and endpoint:
        deployment = (
            (azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or os.environ.get("AZURE_OPENAI_MODEL") or model_default)
        )
        api_ver = azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        client = AzureOpenAI(api_key=key_azure, api_version=api_ver, azure_endpoint=endpoint)
        return client, True, deployment
    key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "[I0 baseline] LLM key required: OPENAI_API_KEY or AZURE_OPENAI_KEY + AZURE_OPENAI_ENDPOINT."
        )
    return OpenAI(api_key=key), False, model_default


def _call_llm(client: Any, *, use_azure: bool, model: str, system: str, user: str) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if not use_azure:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("[I0 baseline] Empty LLM response.")
    return content


def _normalize_i0_policy(pol: dict[str, Any], default_fragment: str) -> dict[str, Any]:
    out = dict(pol)
    frag = str(out.get("_fragment_id") or default_fragment or "f1")
    out["_fragment_id"] = frag
    if "@context" not in out:
        out["@context"] = "http://www.w3.org/ns/odrl.jsonld"
    if not out.get("uid"):
        out["uid"] = uri_policy(f"I0_{frag}_{new_uid()}")
    if "@type" not in out:
        out["@type"] = "Set"
    if not out.get("_type"):
        out["_type"] = "FPa"
    return out


def _policies_to_fragment_sets(
    data: dict[str, Any],
    expected_frag_ids: list[str],
) -> dict[str, FragmentPolicySet]:
    results: dict[str, FragmentPolicySet] = {fid: FragmentPolicySet(fragment_id=fid) for fid in expected_frag_ids}
    fr_block = data.get("fragments")
    if isinstance(fr_block, dict):
        for fid, policies in fr_block.items():
            fid_s = str(fid)
            if fid_s not in results:
                results[fid_s] = FragmentPolicySet(fragment_id=fid_s)
            fps = results[fid_s]
            for pol in policies if isinstance(policies, list) else []:
                if not isinstance(pol, dict):
                    continue
                norm = _normalize_i0_policy(pol, fid_s)
                ptype = str(norm.get("_type", "FPa")).upper()
                if ptype.startswith("FPD") or ptype == "FPd":
                    fps.fpd_policies.append(norm)
                else:
                    fps.fpa_policies.append(norm)
        return results
    # Fallback: flat list with _fragment_id on each policy
    flat = data.get("policies")
    if isinstance(flat, list):
        for pol in flat:
            if not isinstance(pol, dict):
                continue
            fid_s = str(pol.get("_fragment_id") or expected_frag_ids[0] if expected_frag_ids else "f1")
            if fid_s not in results:
                results[fid_s] = FragmentPolicySet(fragment_id=fid_s)
            norm = _normalize_i0_policy(pol, fid_s)
            ptype = str(norm.get("_type", "FPa")).upper()
            if ptype.startswith("FPD") or ptype == "FPd":
                results[fid_s].fpd_policies.append(norm)
            else:
                results[fid_s].fpa_policies.append(norm)
    return results


def _expected_fragment_ids(
    fragments: list[dict[str, Any]],
    fragments_enhanced: Optional[dict[str, Any]],
) -> list[str]:
    if fragments_enhanced and isinstance(fragments_enhanced.get("fragments"), list):
        ids = [str(f.get("id")) for f in fragments_enhanced["fragments"] if f.get("id")]
        if ids:
            return ids
    ids = [str(f.get("id")) for f in fragments if f.get("id")]
    return ids or ["f1"]


def generate_i0_baseline_for_scenario(
    *,
    scenario_id: str,
    fragments: list[dict[str, Any]],
    fragments_enhanced: Optional[dict[str, Any]],
    b2p_policies: list[dict[str, Any]],
    bp_model: dict[str, Any],
    api_key: str | None = None,
    azure_endpoint: str | None = None,
    azure_api_version: str | None = None,
    azure_deployment: str | None = None,
) -> dict[str, FragmentPolicySet]:
    """
    Generate all scenario policies via **one global LLM prompt** (I0 baseline).
    """
    _ = scenario_id
    frag_ids = _expected_fragment_ids(fragments, fragments_enhanced)
    client, use_azure, model = _build_llm_client(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        azure_deployment=azure_deployment,
    )

    per_frag = os.environ.get("I0_PER_FRAGMENT", "").strip().lower() in ("1", "true", "yes")
    if per_frag:
        return _generate_per_fragment_fallback(
            frag_ids=frag_ids,
            fragments=fragments,
            fragments_enhanced=fragments_enhanced,
            b2p_policies=b2p_policies,
            bp_model=bp_model,
            client=client,
            use_azure=use_azure,
            model=model,
        )

    enhanced_payload = fragments_enhanced if fragments_enhanced is not None else {"note": "no fragments_enhanced.json"}
    user = _I0_USER_ALL_FRAGMENTS.format(
        base_uri=BASE_URI,
        bp_model_json=json.dumps(bp_model, ensure_ascii=False, indent=2)[:80000],
        fragments_json=json.dumps(fragments, ensure_ascii=False, indent=2)[:40000],
        enhanced_json=json.dumps(enhanced_payload, ensure_ascii=False, indent=2)[:120000],
        b2p_json=json.dumps(b2p_policies, ensure_ascii=False, indent=2)[:80000],
    )
    print(f"[I0 baseline] Single prompt — scenario, {len(frag_ids)} fragment(s), model={model}")
    raw = _call_llm(client, use_azure=use_azure, model=model, system=_I0_SYSTEM, user=user)
    data = _parse_llm_json_object(raw)
    summary = data.get("process_summary") or data.get("interpretation")
    if summary:
        print(f"[I0 baseline] LLM summary: {str(summary)[:200]}{'…' if len(str(summary)) > 200 else ''}")
    results = _policies_to_fragment_sets(data, frag_ids)
    for fid in frag_ids:
        fps = results.get(fid) or FragmentPolicySet(fragment_id=fid)
        n = len(fps.fpa_policies) + len(fps.fpd_policies)
        print(f"[I0 baseline] '{fid}': {n} LLM policy/policies")
    return results


def _generate_per_fragment_fallback(
    *,
    frag_ids: list[str],
    fragments: list[dict[str, Any]],
    fragments_enhanced: Optional[dict[str, Any]],
    b2p_policies: list[dict[str, Any]],
    bp_model: dict[str, Any],
    client: Any,
    use_azure: bool,
    model: str,
) -> dict[str, FragmentPolicySet]:
    """Dev fallback: one LLM call per fragment (``I0_PER_FRAGMENT=1``). Still no templates."""
    print("[I0 baseline][WARN] I0_PER_FRAGMENT=1 mode — multiple LLM calls.")
    b2p_json = json.dumps(b2p_policies, ensure_ascii=False, indent=2)
    results: dict[str, FragmentPolicySet] = {}
    for fid in frag_ids:
        slice_doc = {
            "fragment": next((f for f in fragments if str(f.get("id")) == fid), {"id": fid}),
            "bp_model": bp_model,
        }
        if fragments_enhanced:
            slice_doc["enhanced"] = fragments_enhanced
        user = (
            f"Generate ALL ODRL policies for fragment {fid} only. "
            f"Inputs:\n{json.dumps(slice_doc, ensure_ascii=False, indent=2)[:100000]}\n"
            f"B2P:\n{b2p_json[:60000]}\n"
            'Respond JSON: {{"fragment_id":"'
            + fid
            + '","policies":[...]}}'
        )
        raw = _call_llm(client, use_azure=use_azure, model=model, system=_I0_SYSTEM, user=user)
        data = _parse_llm_json_object(raw)
        results.update(_policies_to_fragment_sets({"fragments": {fid: data.get("policies", [])}}, [fid]))
    return results


def export_i0_fp_results(
    fp_results: dict[str, FragmentPolicySet],
    output_dir: str,
) -> dict[str, list[str]]:
    """Export JSON-LD (no Agent 4 / template logic)."""
    os.makedirs(output_dir, exist_ok=True)
    exported: dict[str, list[str]] = {}
    for fragment_id, fps in fp_results.items():
        frag_dir = os.path.join(output_dir, fragment_id)
        os.makedirs(frag_dir, exist_ok=True)
        exported[fragment_id] = []
        for i, policy in enumerate(fps.all_policies()):
            odrl = {k: v for k, v in policy.items() if not str(k).startswith("_")}
            ptype = str(policy.get("_type", "I0"))
            subtype = str(policy.get("_subtype", "policy"))
            uid = str(odrl.get("uid") or odrl.get("@id") or f"idx_{i}")
            uid_short = hashlib.sha1(uid.encode("utf-8")).hexdigest()[:8]
            filename = f"I0_{ptype}_{subtype}_{uid_short}.jsonld".replace(" ", "_")
            filepath = os.path.join(frag_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json.dumps(odrl, indent=2, ensure_ascii=False))
            exported[fragment_id].append(filepath)
    total = sum(len(v) for v in exported.values())
    print(f"[I0 baseline] Export: {total} .jsonld file(s) → {output_dir}")
    return exported
