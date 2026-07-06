import json
import os
import re
from typing import Any, Dict, List, Tuple


def _first_existing(*paths: str) -> str:
    for p in paths:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "No file found among: " + ", ".join(os.path.basename(x) for x in paths)
    )


def load_scenario(scenario_id: str, *, base_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load a scenario from the dataset folder.

    Expected structure under ``{base_dir}/src/dataset/{scenario_id}/``:

    - BP model: ``bp_global.json`` (preferred) or ``bp_model.json``
    - Fragments: ``fragments.json``
    - ODRL policies: ``B2P.json`` (preferred) or ``b2p_policies.json``
    """
    scenario_dir = os.path.join(base_dir, "src", "dataset", scenario_id)

    bp_path = _first_existing(
        os.path.join(scenario_dir, "bp_global.json"),
        os.path.join(scenario_dir, "bp_model.json"),
    )
    fr_path = os.path.join(scenario_dir, "fragments.json")
    b2p_path = _first_existing(
        os.path.join(scenario_dir, "B2P.json"),
        os.path.join(scenario_dir, "b2p_policies.json"),
    )

    with open(bp_path, "r", encoding="utf-8") as f:
        bp_model = json.load(f)

    with open(fr_path, "r", encoding="utf-8") as f:
        fragments = json.load(f)

    with open(b2p_path, "r", encoding="utf-8") as f:
        b2p_policies = json.load(f)

    return bp_model, fragments, b2p_policies


def discover_scenarios(
    base_dir: str,
    *,
    pattern: str = r"^scenario\d{3}$",
) -> List[str]:
    """
    List scenario identifiers under ``src/dataset/`` whose folder name
    matches the regular expression (default ``scenario001`` … ``scenario999`` with 3 digits).

    Parameters
    ----------
    base_dir
        Project root (folder containing ``src/dataset``).
    pattern
        Regex applied to the folder name only.
    """
    ds = os.path.join(base_dir, "src", "dataset")
    if not os.path.isdir(ds):
        return []
    rx = re.compile(pattern)
    out: List[str] = []
    for name in sorted(os.listdir(ds)):
        path = os.path.join(ds, name)
        if not os.path.isdir(path):
            continue
        if rx.match(name):
            out.append(name)
    return out
