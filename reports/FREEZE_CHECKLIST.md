# Final Freeze Checklist

1. Refit origin-safe XGBoost for checkouts and returns.
2. Run XGBoost dry runs and the validation reproduction check successfully.
3. Merge `dylan/returns` and `dhanraj/final-test` into `main`.
4. Record the frozen commit: `git rev-parse HEAD`.
5. Send the locked 2026 file only after recording that hash.
6. Dylan runs `sbatch slurm/final_test.sh <hash>` exactly once.
7. Expect 326,400 rows per model per target.
8. Rebuild the dashboard with `--test-dir`.

## Freeze rule

**No changes after the freeze.** Do not edit code, data, configuration, results,
or the dashboard after the frozen hash is recorded and before the one test run.

## Required final-report disclosure

The final report must state that the original XGBoost used leaky short lags,
that an audit found the issue, and that it was fixed before the locked test.
