import json
import os
import re
from typing import Any, Dict, List, Tuple


def _first_existing(*paths: str) -> str:
    for p in paths:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Aucun fichier trouvé parmi : " + ", ".join(os.path.basename(x) for x in paths)
    )


def load_scenario(scenario_id: str, *, base_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Charge un scénario depuis le dossier dataset.

    Structure attendue sous ``{base_dir}/src/dataset/{scenario_id}/`` :

    - Modèle BP : ``bp_global.json`` (priorité) ou ``bp_model.json``
    - Fragments : ``fragments.json``
    - Policies ODRL : ``B2P.json`` (priorité) ou ``b2p_policies.json``
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
    Liste les identifiants de scénarios présents sous ``src/dataset/`` dont le nom
    correspond à l'expression régulière (par défaut ``scenario001`` … ``scenario999`` sur 3 chiffres).

    Parameters
    ----------
    base_dir
        Racine du projet (dossier contenant ``src/dataset``).
    pattern
        Regex appliquée au nom du dossier uniquement.
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