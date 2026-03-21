## Reproducible Eval Runs

- Baseline:
  - `python run_eval_config.py --config eval_configs/baseline.json`

- Fix 3 A/B thresholds:
  - `python run_eval_config.py --config eval_configs/ab_fix3_a_t001.json`
  - `python run_eval_config.py --config eval_configs/ab_fix3_b_t003.json`
  - `python run_eval_config.py --config eval_configs/ab_fix3_c_t005.json`

Notes:
- Determinism is enforced via `seed` in config and stable ordering in scorer.
- Every config writes outputs with its own `write_prefix`, so files do not overwrite each other.
- For custom tests, copy any json in this folder and edit only `overrides`.
