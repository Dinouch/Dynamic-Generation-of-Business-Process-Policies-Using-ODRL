"""
Human-facing HITL narratives in plain English.

- Goal: one business question — “Does this rule reflect what I want in my process?”
- No ODRL/URI/JSON jargon in the text shown to reviewers.
- If OPENAI_API_KEY is set, a dedicated LLM pass rewrites the draft (unless HITL_NARRATIVE_USE_LLM=0).
  Without a key, a deterministic template still avoids technical leakage.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any


_RULE_TITLE: dict[str, str] = {
    "permission": "Permission",
    "prohibition": "Prohibition",
    "obligation": "Obligation",
}


_OP_WORDS: dict[str, str] = {
    "eq": "equals",
    "neq": "does not equal",
    "lt": "is before",
    "lteq": "is no later than",
    "gt": "is after",
    "gteq": "is no earlier than",
    "isPartOf": "is part of",
    "isA": "is a kind of",
}


def _humanize_slug(s: str) -> str:
    t = (s or "").strip().replace("_", "-")
    if not t:
        return ""
    parts = re.split(r"[\s\-]+", t)
    return " ".join(p.capitalize() for p in parts if p).strip()


def _strip_urls_and_namespaces(text: str) -> str:
    if not text:
        return ""
    t = str(text)
    t = re.sub(r"https?://[^\s)\]]+", "", t)
    t = re.sub(r"urn:[^\s)\]]+", "", t)
    t = re.sub(r"\bodrl:[A-Za-z0-9_-]+\b", "", t)
    t = re.sub(r"\bxsd:\w+\b", "", t)
    t = re.sub(r"\brdf:\w+\b", "", t)
    t = re.sub(r"@[a-z]+", "", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def _sanitize_free_text(s: str, *, max_len: int = 2400) -> str:
    raw = (s or "").strip()
    if not raw:
        return ""
    raw = _strip_urls_and_namespaces(raw)
    if len(raw) > max_len:
        raw = raw[: max_len - 1] + "…"
    return raw


def _operand_plain(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, dict):
        if "@value" in v:
            return str(v.get("@value", "")).strip()
        if "@id" in v:
            return _humanize_slug(_strip_urls_and_namespaces(str(v.get("@id", ""))).split("/")[-1])
        return _sanitize_free_text(json.dumps(v, ensure_ascii=False)[:200])
    return _strip_urls_and_namespaces(str(v))


def _constraint_natural(c: Any) -> str | None:
    if not isinstance(c, dict):
        return None
    lo = c.get("leftOperand") or c.get("left_operand")
    op = str(c.get("operator") or c.get("op") or "").strip()
    ro = c.get("rightOperand") or c.get("right_operand")
    lo_p = _operand_plain(lo) if lo is not None else ""
    ro_p = _operand_plain(ro)
    op_w = _OP_WORDS.get(op, op.replace("_", " ") if op else "")
    lo_h = _humanize_slug(lo_p) if lo_p and not lo_p.islower() else (lo_p or "this aspect")
    if lo_p and "date" in lo_p.lower():
        lo_h = "the time window"
    parts = [p for p in (lo_h, op_w, ro_p) if p]
    if len(parts) < 2:
        return None
    return " ".join(parts)


def _gather_constraints_nl(struct: dict[str, Any]) -> list[str]:
    raw = struct.get("constraints")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for c in raw:
        line = _constraint_natural(c)
        if line:
            out.append(line)
    return out


def _action_plain(struct: dict[str, Any]) -> str:
    act = struct.get("action")
    if act is None:
        return ""
    if isinstance(act, dict):
        if "rdf:value" in act and isinstance(act["rdf:value"], dict):
            inner = act["rdf:value"].get("@id", "")
            inner = _strip_urls_and_namespaces(str(inner))
            if "trigger" in inner.lower():
                return "Start or trigger the related handling step when the condition is met."
            return _humanize_slug(inner.split(":")[-1]) or "Carry out the described step."
        aid = act.get("@id") or act.get("id") or act.get("name")
        if aid:
            tail = _strip_urls_and_namespaces(str(aid)).split("/")[-1].split(":")[-1]
            if tail.lower() in ("enable", "use", "execute", "trigger"):
                return "Proceed with the activity as described in the process."
            return f"Perform the action described as « {_humanize_slug(tail)} »."
    if isinstance(act, str):
        return _sanitize_free_text(act) or ""
    return ""


def _first_str(d: dict[str, Any], keys: tuple[str, ...]) -> str:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        s = _sanitize_free_text(str(v).strip())
        if s:
            return s
    return ""


def _role_for_activity(hitl_ctx: dict[str, Any], activity_slug: str) -> str:
    ad = hitl_ctx.get("activity_directory")
    if not isinstance(ad, list):
        return ""
    target = (activity_slug or "").replace(" ", "-").lower()
    for row in ad:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if name.lower() == target or _humanize_slug(name).lower() == activity_slug.lower():
            role = str(row.get("role") or "").strip()
            if role:
                return _humanize_slug(role.replace("_", "-"))
    return ""


def _default_assigner_assignee(
    prop: dict[str, Any],
    struct: dict[str, Any],
    hitl_ctx: dict[str, Any],
    main_activity: str,
) -> tuple[str, str]:
    """Business-style labels — never 'not specified' machine phrasing."""
    a1 = _first_str(struct, ("assigner", "controller", "issuer", "decision_owner"))
    a2 = _first_str(struct, ("assignee", "beneficiary", "actor", "operational_role"))
    if a1 and not re.search(r"https?://|urn:|odrl:", a1, re.I):
        assigner = _humanize_slug(a1) if "-" in a1 or "_" in a1 else a1
    else:
        r = _role_for_activity(hitl_ctx, main_activity)
        mgr = _role_for_activity(hitl_ctx, "make-decision")
        assigner = (
            f"Management / {mgr} (typical owner for this stage)"
            if mgr
            else "Process management (whoever owns this stage in your organisation)"
        )
    if a2 and not re.search(r"https?://|urn:|odrl:", a2, re.I):
        assignee = _humanize_slug(a2) if "-" in a2 or "_" in a2 else a2
    else:
        r = _role_for_activity(hitl_ctx, main_activity)
        assignee = (
            f"Staff — {r} lane"
            if r
            else f"People performing « {main_activity} » and neighbouring steps"
        )
    return assigner, assignee


def _process_title(content: dict[str, Any]) -> str:
    ctx = content.get("hitl_context") if isinstance(content.get("hitl_context"), dict) else {}
    t = str(ctx.get("process_title") or "").strip()
    if t:
        return t
    sid = str(ctx.get("scenario_id") or "").strip()
    if sid:
        return f"Your process ({sid.replace('_', ' ')})"
    return "Your business process"


def _main_activity(prop: dict[str, Any], struct: dict[str, Any]) -> str:
    t = _first_str(
        struct,
        ("target_activity", "target", "targetActivity", "activity", "focus_activity"),
    )
    if t:
        return _humanize_slug(t) if "-" in t or "_" in t else t
    acts = prop.get("involved_activity_names") or []
    if isinstance(acts, list) and acts:
        return _humanize_slug(str(acts[0]))
    gw = str(prop.get("gateway_name") or "").strip()
    return _humanize_slug(gw) if gw else "This step in the process"


def _pattern_when_line(pattern_type: Any, gateway: str) -> str:
    pt = str(pattern_type or "").strip()
    gw_h = _humanize_slug(gateway) if gateway and gateway != "?" else "the decision point"
    if "event_based" in pt or "fork_event" in pt:
        return f"When the process reaches « {gw_h} », whichever relevant event happens first steers what happens next."
    if pt == "loop":
        return "While the process cycles through the same activities until an exit condition is met."
    if pt.startswith("fork_"):
        return f"When the flow splits at « {gw_h} », only the branch that matches the situation should apply."
    if pt.startswith("join_") or pt == "sync":
        return f"When parallel branches meet at « {gw_h} », they must align before the process continues."
    if pt == "conditional_flow":
        return "When a labelled condition on a sequence flow applies."
    if pt == "default_flow":
        return "When no other branch condition matches, the default path should apply."
    return f"When the process behaves according to the situation around « {gw_h} »."


def template_hitl_narrative_en(content: dict[str, Any]) -> str:
    """Deterministic, non-technical narrative (single proposal = usually one rule)."""
    prop = content.get("proposal")
    if not isinstance(prop, dict):
        prop = {}
    hitl_ctx = content.get("hitl_context") if isinstance(content.get("hitl_context"), dict) else {}

    rt = str(prop.get("odrl_rule_type") or "permission").strip().lower()
    rule_title = _RULE_TITLE.get(rt, "Permission")

    fragment = str(prop.get("fragment_id") or "—").strip()
    gateway = str(prop.get("gateway_name") or "").strip()
    pattern_type = prop.get("pattern_type")

    struct = prop.get("odrl_structure_hint") if isinstance(prop.get("odrl_structure_hint"), dict) else {}
    struct = struct or {}

    main_act = _main_activity(prop, struct)
    assigner, assignee = _default_assigner_assignee(prop, struct, hitl_ctx, main_act)

    what = _action_plain(struct)
    if not what:
        what = _sanitize_free_text(str(prop.get("hint_text") or ""))[:400] or (
            "Describe how this part of the process should behave when the situation applies."
        )

    when_line = _pattern_when_line(pattern_type, gateway)
    constraints = _gather_constraints_nl(struct)
    constraint_block = (
        "; ".join(constraints[:6])
        if constraints
        else "No extra limits were spelled out beyond the situation above."
    )

    hint = _sanitize_free_text(str(prop.get("hint_text") or ""))
    justification = _sanitize_free_text(str(prop.get("justification") or ""))

    acts_all = prop.get("involved_activity_names") or []
    act_line = ""
    if isinstance(acts_all, list) and acts_all:
        act_line = ", ".join(_humanize_slug(str(a)) for a in acts_all[:10])
        if len(acts_all) > 10:
            act_line += " …"

    progress_head = ""
    accepted = content.get("accepted_so_far")
    remaining = content.get("remaining_after")
    cur, total = 1, 1
    if accepted is not None and remaining is not None:
        try:
            cur = int(accepted) + 1
            total = int(accepted) + int(remaining) + 1
        except (TypeError, ValueError):
            pass
    progress_head = f"# Proposal {cur} of {total}\n\n"

    proc = _process_title(content)

    lines = [
        progress_head.rstrip(),
        f"**Process:** {proc}",
        f"**Fragment:** {fragment}",
        "",
        "This proposal adds **one rule** to how your process should run.",
        "",
        "---",
        "",
        f"## Rule 1: {rule_title}",
        "",
        f"**Activity:** {main_act}",
        f"**Assigner:** {assigner}",
        f"**Assignee:** {assignee}",
        "",
        f"**What:** {what}",
        f"**When:** {when_line}",
        f"**Constraint:** {constraint_block}",
        "",
        "**Effect if you accept:** The pipeline will keep this behaviour in the generated checks for your process.",
        "**Effect if you reject:** This behaviour will not be added; the run can still continue for other parts.",
        "",
    ]
    if hint:
        lines.extend(["**In plain words:**", "", hint, ""])
    if justification:
        lines.extend(["", "**Why the automation suggested this:**", "", justification, ""])

    if act_line:
        lines.extend(["", "**Other activities in the same slice:**", act_line, ""])

    lines.extend(
        [
            "",
            "---",
            "",
            "**Summary:** Read the rule above as a single behaviour you want staff and systems to follow at that moment in the process.",
            "",
            "**Your question:** Does this rule reflect what you want in your process? Use **Accept** or **Reject**.",
        ]
    )
    return "\n".join(lines).strip()


EXAMPLE_FORMAT = r"""
# Proposal 1 of 5

**Process:** Credit Application BP
**Fragment:** f3

This proposal will generate 2 rules:

## Rule 1: Permission

**Activity:** Receive Missing Documents
**Assigner:** Credit Application Manager
**Assignee:** Staff — Document Reception
**What:** Allow processing the reception of missing documents
**When:** Documents are received before the deadline expires
**Constraint:** Only if this is the first event to occur at the documentation follow-up gateway
**Effect:** The deadline-handling path is automatically discarded
**In plain words:** If the client sends the required documents in time, the staff can go ahead and process them — the deadline path is dropped entirely.

## Rule 2: Prohibition

**Activity:** Handle Documentation Deadline
**Assigner:** Credit Application Manager
**Assignee:** Staff — Deadline Management
**What:** Block processing the deadline-expired path
**When:** Documents have already been received first
**Constraint:** Cannot be triggered if the documents-received event has already fired
**Effect:** Both paths can never be active at the same time
**In plain words:** Once the documents arrive, it's too late to take the deadline path — it's automatically locked out.

**Summary:** This is a first-wins situation — whichever event happens first (documents received or deadline expired) takes over and blocks the other path permanently.

**If you accept:** Both behaviours above are kept in the automated checks for this process.

**If you reject:** Neither behaviour is added; the rest of the pipeline may still continue.

**Your question:** Does this rule reflect what you want in your process?
""".strip()


def _llm_narrative_en(content: dict[str, Any], template_fallback: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        return template_fallback

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return template_fallback

    model = os.environ.get("HITL_NARRATIVE_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=key)
    payload = json.dumps(content, ensure_ascii=False, default=str)[:14000]

    user = f"""You write the **human review** text for a business stakeholder who does **not** know policy languages or BPMN internals.

Raw data (JSON — may include machine artefacts; **do not copy** URIs, namespaces, or field names into the answer):
{payload}

Write **only** markdown for the reviewer. Match the **spirit** of this example (layout and tone):

---EXAMPLE---
{EXAMPLE_FORMAT}
---END---

Rules:
1. Use heading `# Proposal {{cur}} of {{total}}` using accepted_so_far and remaining_after from the JSON (cur = accepted_so_far + 1, total = accepted_so_far + remaining_after + 1). If those fields are missing, use "Proposal 1 of 1".
2. **Process:** use hitl_context.process_title if present; otherwise a short sensible name from scenario_id or say "Your business process".
3. **Fragment:** use proposal.fragment_id.
4. Usually there is **one** rule → say "This proposal adds **one rule**" and a single "## Rule 1: Permission|Prohibition|Obligation" from proposal.odrl_rule_type.
5. **Activity:** main business activity in Title Case plain English (from target_activity or involved_activity_names or gateway — humanise slugs).
6. **Assigner** and **Assignee:** short **job/organisational roles** (e.g. "Credit Application Manager", "Staff — Document Reception"). Infer from hint_text, activity names, and hitl_context.activity_directory roles if present. Never paste URLs. Never leave a machine placeholder.
7. **What / When / Constraint / Effect:** full sentences, concrete process language. No "odrl", "URI", "leftOperand", "trigger" tokens.
8. **In plain words:** 2–4 sentences, conversational.
9. **Summary:** one short paragraph tying the rule(s) to the business situation (e.g. first-wins gateway, loop, etc.) when relevant.
10. Add **If you accept:** and **If you reject:** one line each — business impact only.
11. End with: **Your question:** Does this rule reflect what you want in your process?

Maximum ~4500 characters. Output markdown only, no preamble."""

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You explain process rules to business reviewers in clear English. You never output URLs, ODRL, RDF, or JSON key names.",
                },
                {"role": "user", "content": user},
            ],
            max_tokens=1800,
            temperature=0.35,
        )
        out = (r.choices[0].message.content or "").strip()
        if len(out) < 120:
            return template_fallback
        if re.search(r"https?://|odrl:|leftOperand|@id|urn:", out, re.I):
            return template_fallback
        return out
    except Exception:
        return template_fallback


async def build_hitl_narrative_async(content: dict[str, Any]) -> str:
    template = template_hitl_narrative_en(content)
    mode = os.environ.get("HITL_NARRATIVE_USE_LLM", "auto").strip().lower()
    has_key = bool(os.environ.get("OPENAI_API_KEY"))

    if mode in {"0", "false", "no", "off"}:
        return template
    if mode in {"1", "true", "yes", "on"} or (mode in {"auto", ""} and has_key):
        return await asyncio.to_thread(_llm_narrative_en, content, template)
    return template
