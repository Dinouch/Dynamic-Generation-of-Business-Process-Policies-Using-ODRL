import json
import os
from fragmenter import Fragmenter
from policy_generator import PolicyGenerator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    with open(os.path.join(BASE_DIR, "bpmn_model.json"), "r") as f:
        bp_model = json.load(f)
    with open(os.path.join(BASE_DIR, "process_policy.json"), "r") as f:
        bp_policy = json.load(f)

    # Fragment the model
    fragmenter = Fragmenter(bp_model)
    fragments = fragmenter.fragment_process()

    # Save each fragment in a separate file
    for i, fragment in enumerate(fragments):
        path = os.path.join(BASE_DIR, f"output_fragment_{i+1}.json")
        with open(path, "w") as f:
            json.dump(fragment, f, indent=2)

    print(f"{len(fragments)} fragments générés.")

    # Generate policies
    activity_policies, dependency_policies = PolicyGenerator(bp_model, bp_policy).generate_policies()

    with open(os.path.join(BASE_DIR, "output_activity_policies.json"), "w") as f:
        json.dump(activity_policies, f, indent=2)

    with open(os.path.join(BASE_DIR, "output_dependency_policies.json"), "w") as f:
        json.dump(dependency_policies, f, indent=2)

    print("Politiques sauvegardées.")

if __name__ == "__main__":
    main()