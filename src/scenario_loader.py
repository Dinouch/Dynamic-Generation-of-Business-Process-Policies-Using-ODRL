import json
import os
from typing import Any, Dict, List, Tuple


def load_scenario(scenario_id: str, *, base_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load a scenario from JSON files.

    Expected structure:
      {base_dir}/src/dataset/{scenario_id}/
        - bp_model.json
        - fragments.json
        - b2p_policies.json
    """
    scenario_dir = os.path.join(base_dir, "src", "dataset", scenario_id)

    bp_path = os.path.join(scenario_dir, "bp_model.json")
    fr_path = os.path.join(scenario_dir, "fragments.json")
    b2p_path = os.path.join(scenario_dir, "b2p_policies.json")

    with open(bp_path, "r", encoding="utf-8") as f:
        bp_model = json.load(f)

    with open(fr_path, "r", encoding="utf-8") as f:
        fragments = json.load(f)

    with open(b2p_path, "r", encoding="utf-8") as f:
        b2p_policies = json.load(f)

    return bp_model, fragments, b2p_policies