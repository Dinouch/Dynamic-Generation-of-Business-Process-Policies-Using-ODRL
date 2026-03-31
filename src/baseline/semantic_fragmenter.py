"""
Fragmentation par LLM (regroupement en phases métier) + rattachement des gateways BPMN.

Entrée : un ``bp_model`` déjà parsé (activities, gateways, flows) avec noms résolus
(comme après ``BPMNParser.convert_ids_to_names``).

Le LLM reçoit soit cette liste, soit un tableau d objets ``task`` + métadonnées gateway
(``_structured_payload_for_llm``) lorsque ``decompose_tasks_with_llm(..., bp_model=...)`` est utilisé.

Sortie : liste de fragments au même format que ``dataset/scenario1/fragments.json`` :
activités et noms de gateways en **slugs** (kebab-case, sans ponctuation) ;
``[{ "id": "f1", "activities": ["check-for-completeness", ...], "gateways": [...] }, ...]``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from collections import Counter, deque
from json import JSONDecodeError
from typing import Any, Optional

_logger = logging.getLogger(__name__)

from openai import AzureOpenAI, BadRequestError, OpenAI

# Déploiement distribué + contiguïté (liste linéaire) + regroupement gateway
FRAGMENTATION_SYSTEM_PROMPT = """You are a business process analyst assistant.

Input: a JSON array of task objects in process order. Each object has:
- "task": the task name (required, copy exactly in output)
- "gateway_group": if present, this task is one outgoing branch of a gateway split
- "gateway_type": type of that gateway (XOR, AND, OR, EVENT_BASED_EXCLUSIVE...)
- "condition": the branch condition if any (e.g. ">500", "complete")

Gateway grouping rules:
- XOR or EVENT_BASED_EXCLUSIVE: branches are mutually exclusive (only one runs at runtime).
  Tasks sharing the same gateway_group of this type MUST be in the same fragment.
- AND or OR: branches run in parallel. Tasks sharing the same gateway_group of this type
  SHOULD be in different fragments (independent deployable units).

Output format
Return only one valid JSON object, no markdown fences, no text before or after.
{"fragments":[{"activities":["name1","name2"]},{"activities":["name3"]}]}

Rules
- Each fragment must be ONE contiguous slice of the input list.
- Every task must appear exactly once.
- Output multiple fragments (typically 3-6) grouping phases of the workflow.
- Copy each task name exactly as given in the "task" field.
"""

# Déterminisme pour démo (température 0 + seed si l API l accepte)
LLM_FRAGMENTATION_TEMPERATURE = 0.0
LLM_FRAGMENTATION_SEED = 42

# Au-delà de ce nombre de tâches, un seul fragment est rejeté (relance LLM).
LLM_FRAGMENTATION_MIN_TASKS_FOR_MULTI_FRAGMENT = 5


def _max_fragmentation_attempts() -> int:
    """
    Nombre max d appels LLM pour la fragmentation. 0 = illimité jusqu au succès.
    Variable d environnement : LLM_FRAGMENTATION_MAX_ATTEMPTS (entier, défaut 0).
    """
    raw = (os.environ.get("LLM_FRAGMENTATION_MAX_ATTEMPTS") or "").strip()
    if not raw:
        return 0
    try:
        v = int(raw)
    except ValueError:
        return 0
    return max(0, v)


def normalize_fragment_slug(name: str) -> str:
    """
    Libellé BPMN → identifiant style ``dataset/scenario1/fragments.json`` :
    minuscules, tirets, sans ponctuation (points, virgules, etc.).

    Ex. ``Check for Completeness`` → ``check-for-completeness``.
    Idempotent si le nom est déjà au format slug.
    """
    s = (name or "").strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("_", " ")
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "-", s.strip())
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def _apply_slug_normalization_to_fragments(fragments: list[dict[str, Any]]) -> None:
    """Mutateur : remplace activités et noms de gateways par ``normalize_fragment_slug``."""
    for frag in fragments:
        frag["activities"] = [normalize_fragment_slug(a) for a in frag.get("activities", [])]
        for gw in frag.get("gateways", []):
            if isinstance(gw, dict) and "name" in gw and gw["name"] is not None:
                gw["name"] = normalize_fragment_slug(str(gw["name"]))


def _scrub_activity_label(name: str) -> str:
    """
    Retire espaces superflus guillemets typographiques et ponctuation finale inutile
    pour stabiliser l entrée LLM tout en gardant le sens du libellé.
    """
    s = (name or "").strip()
    s = (
        s.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    s = re.sub(r"\s+", " ", s).strip()
    while len(s) > 1 and s[-1] in ".!?;:,…":
        s = s[:-1].rstrip()
    return s.strip(' "\'')


def _labels_for_llm(ordered_tasks: list[str]) -> tuple[list[str], list[str]]:
    """
    Retourne (labels_envoyes, ordered_tasks) même longueur.
    Si plusieurs tâches donnent le même libellé scrubbé on garde le nom d origine pour désembiguïser.
    """
    scrubs = [_scrub_activity_label(t) for t in ordered_tasks]
    counts = Counter(scrubs)
    labels: list[str] = []
    for orig, sc in zip(ordered_tasks, scrubs):
        labels.append(orig if counts[sc] > 1 else sc)
    return labels, ordered_tasks


def _map_llm_token_to_original(
    token: str,
    ordered_tasks: list[str],
    labels_sent: list[str],
) -> Optional[str]:
    token = (token or "").strip()
    for orig, sent in zip(ordered_tasks, labels_sent):
        if token == sent or token == orig:
            return orig
    scrub_t = _scrub_activity_label(token)
    for orig, sent in zip(ordered_tasks, labels_sent):
        if scrub_t == _scrub_activity_label(sent) or scrub_t == _scrub_activity_label(orig):
            return orig
    return None


def _loads_llm_json(raw: str) -> Any:
    """
    Parse le premier JSON objet ou tableau valide dans la réponse (préambule markdown
    ou texte après le JSON ignorés — fréquent sur Azure sans response_format json_object).
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    if not text:
        raise ValueError("Empty LLM response body")

    try:
        return json.loads(text)
    except JSONDecodeError:
        dec = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                return dec.raw_decode(text, i)[0]
            except JSONDecodeError:
                continue
        raise ValueError(
            "Could not parse JSON from LLM response (preview): "
            f"{text[:500]!r}{'…' if len(text) > 500 else ''}"
        ) from None


def _parse_llm_fragments_payload(raw: str) -> list[list[str]]:
    data = _loads_llm_json(raw)

    if isinstance(data, dict) and "fragments" in data:
        out: list[list[str]] = []
        for item in data["fragments"]:
            if isinstance(item, dict) and "activities" in item:
                out.append([str(x) for x in item["activities"]])
            elif isinstance(item, list):
                out.append([str(x) for x in item])
        return out

    if isinstance(data, list):
        out = []
        for item in data:
            if isinstance(item, dict) and "activities" in item:
                out.append([str(x) for x in item["activities"]])
            elif isinstance(item, list):
                out.append([str(x) for x in item])
        return out

    raise ValueError("LLM response JSON must be an object with key fragments or a list of fragments")


def _validate_and_remap_groups(
    groups: list[list[str]],
    ordered_tasks: list[str],
    labels_sent: list[str],
) -> list[list[str]]:
    if not groups:
        raise ValueError("LLM returned no fragments")

    remapped: list[list[str]] = []
    pos = {t: i for i, t in enumerate(ordered_tasks)}

    for g in groups:
        if not g:
            continue
        chunk: list[str] = []
        for token in g:
            orig = _map_llm_token_to_original(token, ordered_tasks, labels_sent)
            if orig is None:
                raise ValueError(f"LLM returned unknown task name {token!r}")
            chunk.append(orig)
        idxs = [pos[t] for t in chunk]
        if len(set(idxs)) != len(idxs):
            raise ValueError("Duplicate task within one fragment in LLM output")
        lo, hi = min(idxs), max(idxs)
        expected_range = list(range(lo, hi + 1))
        if set(idxs) != set(expected_range):
            raise ValueError(
                "Fragment activities must be contiguous in the process order — "
                f"got positions {idxs}. Each fragment must be one uninterrupted block "
                "from the input list (no tasks in between assigned to other fragments)."
            )
        if idxs != expected_range:
            raise ValueError(
                "Task order inside a fragment must follow the process order "
                f"(got positions {idxs}, expected {expected_range})"
            )
        remapped.append(chunk)

    flat = [t for g in remapped for t in g]
    if Counter(flat) != Counter(ordered_tasks):
        raise ValueError("LLM fragments must include each input task exactly once")

    remapped = [g for g in remapped if g]
    remapped.sort(key=lambda g: min(pos[t] for t in g))
    if (
        len(remapped) == 1
        and len(ordered_tasks) >= LLM_FRAGMENTATION_MIN_TASKS_FOR_MULTI_FRAGMENT
    ):
        raise ValueError(
            "Fragmentation must use at least 2 fragments for this process size "
            "(a single fragment covering all tasks is not allowed)."
        )
    return remapped


def _create_llm_client() -> tuple[Any, str, bool]:
    """
    (client, model_or_azure_deployment, is_azure)
    Variables d environnement alignées sur le reste du projet.
    """
    endpoint = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip()
    key_azure = (
        os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("AZURE_OPENAI_KEY")
        or ""
    ).strip()
    api_ver = (os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-02-15-preview").strip()
    deploy = (
        os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        or ""
    ).strip()

    if endpoint and key_azure and deploy:
        client = AzureOpenAI(
            api_key=key_azure,
            api_version=api_ver,
            azure_endpoint=endpoint,
        )
        return client, deploy, True

    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "Missing LLM credentials set OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT "
            "AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT for fragmentation"
        )
    model = (os.environ.get("OPENAI_MODEL") or "gpt-4o").strip()
    return OpenAI(api_key=key), model, False


def _call_fragmentation_llm(
    user_json: str,
    *,
    temperature: float = LLM_FRAGMENTATION_TEMPERATURE,
    seed: int = LLM_FRAGMENTATION_SEED,
    remediation_user_message: Optional[str] = None,
) -> str:
    client, model, is_azure = _create_llm_client()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": FRAGMENTATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_json},
    ]
    if remediation_user_message:
        _logger.info(
            "Fragmentation LLM — message utilisateur additionnel (relance validation / parsing) :\n%s",
            remediation_user_message,
        )
        messages.append({"role": "user", "content": remediation_user_message})
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }
    # json_object : OpenAI direct Azure selon modèle peut refuser on suit le pattern Agent 4
    if not is_azure:
        kwargs["response_format"] = {"type": "json_object"}
    # Reproductibilité (OpenAI récent Azure selon version déploiement)
    kwargs["seed"] = seed

    try:
        resp = client.chat.completions.create(**kwargs)
    except BadRequestError:
        kwargs.pop("seed", None)
        resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content
    if not content:
        raise ValueError("Empty LLM response for fragmentation")
    return content


def decompose_by_semantic_cohesion(
    activities: list[str],
    *,
    threshold: float = 0.55,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[list[str]]:
    """
    Compatibilité d import : la fragmentation utilise le LLM ``threshold`` et
    ``model_name`` sont ignorés.
    """
    _ = (threshold, model_name)
    return decompose_tasks_with_llm(activities)


def decompose_tasks_with_llm(
    ordered_tasks: list[str],
    *,
    bp_model: Optional[dict[str, Any]] = None,
    llm_temperature: float = LLM_FRAGMENTATION_TEMPERATURE,
    llm_seed: int = LLM_FRAGMENTATION_SEED,
) -> list[list[str]]:
    """
    Regroupe les tâches ordonnées en fragments via le LLM.
    Si ``bp_model`` est fourni, l entrée utilisateur est un tableau d objets tâche + gateway
    (voir ``_structured_payload_for_llm``) ; sinon un simple tableau de libellés.
    En cas d erreur de parsing ou de validation, les appels sont relancés jusqu à
    un résultat valide (voir LLM_FRAGMENTATION_MAX_ATTEMPTS pour une limite optionnelle).
    """
    if not ordered_tasks:
        return []

    if bp_model is not None:
        user_payload, labels_sent, originals = _structured_payload_for_llm(
            ordered_tasks, bp_model
        )
    else:
        labels_sent, originals = _labels_for_llm(ordered_tasks)
        user_payload = json.dumps(labels_sent, ensure_ascii=False)

    _logger.info(
        "Fragmentation LLM — entrée utilisateur (JSON exact du rôle « user », 1er message) :\n%s",
        user_payload,
    )

    max_attempts = _max_fragmentation_attempts()
    remediation: Optional[str] = None
    last_err: Optional[Exception] = None
    last_raw = ""

    attempt = 0
    while True:
        attempt += 1
        if max_attempts and attempt > max_attempts:
            preview = (last_raw or "")[:1200]
            msg = (
                f"Fragmentation LLM : abandon après {max_attempts} tentative(s) "
                f"(LLM_FRAGMENTATION_MAX_ATTEMPTS). Dernière erreur : {last_err}"
            )
            if preview:
                msg += f"\n--- LLM raw (truncated) ---\n{preview}" + (
                    "…" if len(last_raw or "") > 1200 else ""
                )
            raise ValueError(msg) from last_err

        # Légère variation de seed après le 1er essai pour sortir des réponses répétées
        call_seed = llm_seed if attempt == 1 else llm_seed + attempt - 1
        raw = _call_fragmentation_llm(
            user_payload,
            temperature=llm_temperature,
            seed=call_seed,
            remediation_user_message=remediation,
        )
        last_raw = raw
        remediation = None

        try:
            groups = _parse_llm_fragments_payload(raw)
        except ValueError as e:
            last_err = e
            _logger.warning(
                "Fragmentation tentative %d : échec parsing JSON — %s",
                attempt,
                e,
            )
            remediation = (
                f"Your previous answer could not be parsed as valid JSON. Error: {e}. "
                "Return ONLY one JSON object with key \"fragments\" and an array of "
                'objects {"activities":["..."]}, no markdown fences, no text before or after. '
                'Activity names must match the "task" values from the input.'
            )
            continue

        try:
            return _validate_and_remap_groups(groups, originals, labels_sent)
        except ValueError as e:
            last_err = e
            err_line = str(e).split("\n---")[0].strip()
            _logger.warning(
                "Fragmentation tentative %d : échec validation — %s",
                attempt,
                err_line,
            )
            remediation = (
                "Your previous JSON failed validation. Error: "
                f"{err_line}\n\n"
                "Output a corrected JSON object only (same shape). "
                "Each fragment must be exactly one contiguous block of the INPUT array. "
                "Within each fragment, list activities in the SAME ORDER as the input. "
                "Use **several** fragments for a long list (do not merge everything into one fragment)."
            )
            continue


def _is_task_activity(act: dict[str, Any]) -> bool:
    """
    Tâches utilisateur du processus. Les nœuds ``type: event`` (start/end séparés)
    sont exclus ; les tâches avec ``start``/``end`` booléens (format dataset) restent
    des activités métier.
    """
    t = (act.get("type") or "task").lower()
    if t == "event":
        return False
    return True


def _task_names(bp_model: dict[str, Any]) -> set[str]:
    return {a["name"] for a in bp_model.get("activities", []) if _is_task_activity(a)}


def linearize_tasks_topological(bp_model: dict[str, Any]) -> list[str]:
    """
    Ordonne les tâches (hors start/end) selon un tri topologique du graphe
    induit par les flux séquence (activités + gateways).

    Les cycles sont gérés en identifiant des arêtes de retour (DFS depuis les starts),
    puis en les excluant du tri de Kahn. Si le tri reste incomplet, complément par DFS
    depuis les starts puis les nœuds restants.
    """
    flows = [f for f in bp_model.get("flows", []) if f.get("type", "sequence") == "sequence"]
    nodes: set[str] = set()
    for f in flows:
        nodes.add(f["from"])
        nodes.add(f["to"])

    adj: dict[str, list[str]] = {n: [] for n in nodes}
    indeg: dict[str, int] = {n: 0 for n in nodes}
    for f in flows:
        u, v = f["from"], f["to"]
        if u not in nodes or v not in nodes:
            continue
        adj[u].append(v)
        indeg[v] += 1

    tasks = _task_names(bp_model)

    start_names = {a["name"] for a in bp_model.get("activities", []) if a.get("start")}
    back_edges: set[tuple[str, str]] = set()

    visited: set[str] = set()
    in_stack: set[str] = set()

    def find_back_edges(u: str) -> None:
        visited.add(u)
        in_stack.add(u)
        for v in adj.get(u, []):
            if v not in visited:
                find_back_edges(v)
            elif v in in_stack:
                back_edges.add((u, v))
        in_stack.discard(u)

    for s in start_names:
        if s in nodes and s not in visited:
            find_back_edges(s)
    for n in sorted(nodes):
        if n not in visited:
            find_back_edges(n)

    indeg = {n: 0 for n in nodes}
    for f in flows:
        u, v = f["from"], f["to"]
        if u not in nodes or v not in nodes:
            continue
        if (u, v) not in back_edges:
            indeg[v] += 1

    q = deque([n for n in nodes if indeg[n] == 0])
    order: list[str] = []
    tmp_indeg = dict(indeg)

    while q:
        n = q.popleft()
        order.append(n)
        for v in adj.get(n, []):
            if (n, v) in back_edges:
                continue
            tmp_indeg[v] -= 1
            if tmp_indeg[v] == 0:
                q.append(v)

    if len(order) < len(nodes):
        visited2: set[str] = set(order)

        def dfs(u: str) -> None:
            if u in visited2:
                return
            visited2.add(u)
            order.append(u)
            for w in adj.get(u, []):
                dfs(w)

        for s in sorted(start_names):
            if s in nodes:
                dfs(s)
        for n in sorted(nodes):
            if n not in visited2:
                dfs(n)

    task_order = [n for n in order if n in tasks]

    for name in sorted(tasks - set(task_order)):
        task_order.append(name)

    return task_order


def _normalize_gateway_type(gw: dict[str, Any]) -> str:
    t = str(gw.get("type", "XOR")).upper().replace("-", "_")
    if t == "EVENT":
        return "EVENT_BASED_EXCLUSIVE"
    return t


def _structured_payload_for_llm(
    ordered_tasks: list[str],
    bp_model: dict[str, Any],
) -> tuple[str, list[str], list[str]]:
    """
    Construit le JSON utilisateur pour le LLM : une entrée par tâche ordonnée, avec
    ``gateway_group`` / ``gateway_type`` / ``condition`` lorsque le flux est une branche
    sortante d une gateway vers cette tâche (source = gateway, cible = tâche).
    """
    labels_sent, originals = _labels_for_llm(ordered_tasks)

    task_set = set(originals)
    gw_map = {gw["name"]: gw for gw in bp_model.get("gateways", [])}

    gw_branches: dict[str, dict[str, Any]] = {}
    for flow in bp_model.get("flows", []):
        src = flow.get("from", "")
        tgt = flow.get("to", "")
        gw_name = flow.get("gateway")
        if not gw_name or src not in gw_map or tgt not in task_set:
            continue
        if gw_name not in gw_branches:
            gw_branches[gw_name] = {
                "type": _normalize_gateway_type(gw_map[gw_name]),
                "branches": [],
            }
        gw_branches[gw_name]["branches"].append((tgt, flow.get("condition")))

    gw_splits = {k: v for k, v in gw_branches.items() if len(v["branches"]) > 1}

    task_gw_info: dict[str, tuple[str, str, Optional[str]]] = {}
    for gw_name, info in gw_splits.items():
        for task_name, condition in info["branches"]:
            task_gw_info[task_name] = (gw_name, info["type"], condition)

    items: list[dict[str, Any]] = []
    for orig, sent in zip(originals, labels_sent):
        item: dict[str, Any] = {"task": sent}
        if orig in task_gw_info:
            gw_name, gw_type, condition = task_gw_info[orig]
            item["gateway_group"] = gw_name
            item["gateway_type"] = gw_type
            if condition:
                item["condition"] = condition
        items.append(item)

    return json.dumps(items, ensure_ascii=False), labels_sent, originals


def _assign_gateways_to_fragments(
    fragments: list[dict[str, Any]],
    bp_model: dict[str, Any],
) -> None:
    activity_to_frag: dict[str, int] = {}
    for i, frag in enumerate(fragments):
        for a in frag.get("activities", []):
            activity_to_frag[a] = i

    gw_by_name = {g["name"]: g for g in bp_model.get("gateways", [])}

    for frag in fragments:
        frag["gateways"] = []

    seen: list[set[str]] = [set() for _ in fragments]

    for flow in bp_model.get("flows", []):
        if flow.get("type", "sequence") != "sequence":
            continue
        gwn = flow.get("gateway")
        if not gwn or gwn not in gw_by_name:
            continue
        fa, ta = flow["from"], flow["to"]
        idx: Optional[int] = None
        if fa in activity_to_frag and ta in activity_to_frag:
            idx = (
                activity_to_frag[fa]
                if activity_to_frag[fa] == activity_to_frag[ta]
                else activity_to_frag[fa]
            )
        elif fa in activity_to_frag:
            idx = activity_to_frag[fa]
        elif ta in activity_to_frag:
            idx = activity_to_frag[ta]
        if idx is None:
            continue
        g = gw_by_name[gwn]
        rec = {"name": g["name"], "type": _normalize_gateway_type(g)}
        key = rec["name"]
        if key not in seen[idx]:
            seen[idx].add(key)
            fragments[idx]["gateways"].append(rec)


def build_fragments_from_bp_model(
    bp_model: dict[str, Any],
    *,
    semantic_threshold: float = 0.55,
    model_name: str = "all-MiniLM-L6-v2",
    llm_temperature: float = LLM_FRAGMENTATION_TEMPERATURE,
    llm_seed: int = LLM_FRAGMENTATION_SEED,
) -> list[dict[str, Any]]:
    """
    Construit les fragments (format dataset) à partir d un bp_model parsé.
    ``semantic_threshold`` et ``model_name`` sont des paramètres hérités sans effet.
    """
    _ = (semantic_threshold, model_name)

    ordered_tasks = linearize_tasks_topological(bp_model)
    if not ordered_tasks:
        return []

    groups = decompose_tasks_with_llm(
        ordered_tasks,
        bp_model=bp_model,
        llm_temperature=llm_temperature,
        llm_seed=llm_seed,
    )

    fragments: list[dict[str, Any]] = []
    for i, acts in enumerate(groups):
        if not acts:
            continue
        fragments.append({"id": f"f{i + 1}", "activities": list(acts), "gateways": []})

    _assign_gateways_to_fragments(fragments, bp_model)
    _apply_slug_normalization_to_fragments(fragments)
    return fragments


class SemanticFragmenter:
    """
    Fragmentation pilotée par LLM + rattachement des gateways BPMN.
    """

    def __init__(
        self,
        bp_model: dict[str, Any],
        *,
        semantic_threshold: float = 0.55,
        model_name: str = "all-MiniLM-L6-v2",
        llm_temperature: float = LLM_FRAGMENTATION_TEMPERATURE,
        llm_seed: int = LLM_FRAGMENTATION_SEED,
    ):
        self.bp_model = bp_model
        self.semantic_threshold = semantic_threshold
        self.model_name = model_name
        self.llm_temperature = llm_temperature
        self.llm_seed = llm_seed
        self.fragments: list[dict[str, Any]] = []
        self.fragment_dependencies: list[dict[str, Any]] = []

    def fragment_process(self, strategy: str | None = None) -> list[dict[str, Any]]:
        _ = strategy
        self.fragments = build_fragments_from_bp_model(
            self.bp_model,
            semantic_threshold=self.semantic_threshold,
            model_name=self.model_name,
            llm_temperature=self.llm_temperature,
            llm_seed=self.llm_seed,
        )
        for i, frag in enumerate(self.fragments):
            frag["id"] = i
        return self.fragments

    def save_fragments(self, output_dir: str) -> list[str]:
        os.makedirs(output_dir, exist_ok=True)
        saved: list[str] = []
        for i, fragment in enumerate(self.fragments):
            fp = os.path.join(output_dir, f"fragment_{i + 1}.json")
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(fragment, f, indent=2, ensure_ascii=False)
            saved.append(fp)
        dep = os.path.join(output_dir, "fragment_dependencies.json")
        with open(dep, "w", encoding="utf-8") as f:
            json.dump(self.fragment_dependencies, f, indent=2)
        saved.append(dep)
        return saved


def save_fragments_json(fragments: list[dict[str, Any]], path: str) -> None:
    """Écrit un unique ``fragments.json`` (ids ``f1``, ``f2``, ...)."""
    out = []
    for i, frag in enumerate(fragments):
        item = {
            "id": frag.get("id") if isinstance(frag.get("id"), str) else f"f{i + 1}",
            "activities": list(frag.get("activities", [])),
            "gateways": list(frag.get("gateways", [])),
        }
        out.append(item)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def bpmn_xml_to_fragments_json(
    bpmn_path: str,
    fragments_json_path: str,
    *,
    semantic_threshold: float = 0.55,
    model_name: str = "all-MiniLM-L6-v2",
    llm_temperature: float = LLM_FRAGMENTATION_TEMPERATURE,
    llm_seed: int = LLM_FRAGMENTATION_SEED,
) -> list[dict[str, Any]]:
    """
    Parse un fichier BPMN XML, fragmente, écrit ``fragments.json``.
    """
    from baseline.bpmn_parser import BPMNParser

    parser = BPMNParser()
    model = parser.parse_file(bpmn_path)
    if not model:
        raise ValueError(f"Échec du parsing BPMN : {bpmn_path}")
    model = parser.convert_ids_to_names(model)
    fr = SemanticFragmenter(
        model,
        semantic_threshold=semantic_threshold,
        model_name=model_name,
        llm_temperature=llm_temperature,
        llm_seed=llm_seed,
    )
    fr.fragment_process()
    normalized = []
    for i, frag in enumerate(fr.fragments):
        d = dict(frag)
        d["id"] = f"f{i + 1}"
        normalized.append(d)
    save_fragments_json(normalized, fragments_json_path)
    return normalized


if __name__ == "__main__":
    import argparse
    import sys

    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src not in sys.path:
        sys.path.insert(0, _src)

    ap = argparse.ArgumentParser(
        description="Parse BPMN XML → fragmentation LLM → fragments.json"
    )
    ap.add_argument("bpmn", help="Fichier .bpmn / .xml")
    ap.add_argument(
        "-o",
        "--output",
        default="fragments.json",
        help="Chemin du fragments.json (défaut: fragments.json)",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=LLM_FRAGMENTATION_TEMPERATURE,
        help=f"Température LLM (défaut: {LLM_FRAGMENTATION_TEMPERATURE})",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=LLM_FRAGMENTATION_SEED,
        help=f"Seed API pour reproductibilité (défaut: {LLM_FRAGMENTATION_SEED})",
    )
    args = ap.parse_args()
    bpmn_xml_to_fragments_json(
        args.bpmn,
        args.output,
        llm_temperature=args.temperature,
        llm_seed=args.seed,
    )
    print(f"Écrit : {args.output}")
