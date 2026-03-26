"""
Point d'entrée à la racine du projet : délègue vers legacy_evaluation.run_evaluation.

Usage (depuis la racine BPFragmentODRLProject) :
  python run_evaluation.py [options]
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from legacy_evaluation.run_evaluation import main

if __name__ == "__main__":
    main()
