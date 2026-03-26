# legacy_evaluation

Anciens scripts et métriques **non branchés** sur le pipeline actuel (`src/pipeline.py` + `src/agents/`).

## Exécution

Depuis la racine du projet `BPFragmentODRLProject` :

```bash
python run_evaluation.py --help
```

Ou directement :

```bash
python src/legacy_evaluation/run_evaluation.py --help
```

Assurez-vous que `src` est importable (les scripts ci-dessus ajoutent `src` au `sys.path`).

## Limites connues

- `evaluation_pipeline.py` attend encore des modules `enhanced_policy_generator` / `enhanced_policy_generator_llm` s’ils ne sont pas présents sous `src/`.
- `run_experiments.py` nécessite un module `policy_generator` (hors dépôt).
