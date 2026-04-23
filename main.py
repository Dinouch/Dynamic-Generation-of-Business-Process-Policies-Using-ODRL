"""
main.py
═══════════════════════════════════════════════════════════════════════════════
Pipeline complet — Agent 1 → Exception handling agent <→ Agent 3 → Agent 4 <→ Agent 5
Scénario: src/dataset/scenario1 (fragments BP + B2P).

Mode standalone (sans LLM) 

Mode complet (avec LLM) 

Mode pipeline : orchestration asynchrone (bus ACL), via ``run_looped_orchestration``.

"""

import io
import os
import sys


def _configure_windows_console_stdio() -> None:
    """Évite UnicodeEncodeError sous Windows (cp1252) lors des print Unicode (═, etc.)."""
    if sys.platform != "win32":
        return
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        elif hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        elif hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
    except Exception:
        pass


_configure_windows_console_stdio()

_ROOT = os.path.abspath(os.path.dirname(__file__))
_SRC = os.path.join(_ROOT, "src")
sys.path.insert(0, _SRC)


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_ROOT, ".env"), override=True)
        load_dotenv(os.path.join(_SRC, ".env"), override=True)
        return
    except ImportError:
        pass
    for path in [os.path.join(_ROOT, ".env"), os.path.join(_SRC, ".env")]:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, _, v = line.partition("=")
                            os.environ[k.strip()] = v.strip()
            except Exception:
                pass


_load_env()

from orchestration.scenario_loader import load_scenario
from orchestration.runners import (
    run_looped_orchestration,
    run_sequential_agents,
)

# Scénario par défaut
_SCENARIO = "scenario1"

# Mode d'exécution : True = pipeline async (bus + ACL), False = démo séquentielle locale
# -> il suffit de modifier cette constante.
USE_PIPELINE_MODE = True


def main():
    bp_model, fragments, b2p_policies = load_scenario(_SCENARIO, base_dir=_ROOT)
    context_mode = "local"

    if USE_PIPELINE_MODE:
        run_looped_orchestration(
            bp_model=bp_model,
            fragments=fragments,
            b2p_policies=b2p_policies,
            context_mode=context_mode,
        )
    else:
        run_sequential_agents(
            bp_model=bp_model,
            fragments=fragments,
            b2p_policies=b2p_policies,
            context_mode=context_mode,
        )


if __name__ == "__main__":
    main()
