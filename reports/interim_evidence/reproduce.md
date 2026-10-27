# Evidence Reproduction

Run from the repository root after activating `.\.venv\Scripts\Activate.ps1`:

```powershell
python -m src.check_trip_availability
python -m src.probe_weather_coverage
python -m src.baselines --model seasonal_naive
python -m src.baselines --model negative_binomial
python -m src.train_transformer --epochs 5 --batch-size 32
python -m src.evaluate_transformer --batch-size 32
python -m src.make_figures
python -m src.build_interim_evidence
python -m pytest -q
```

The checked-in experiment settings are `src/config.py`: seed 42, training 2023--2024, validation 2025, test locked behind `--final`, 50 stations, and the 19-quantile 0.05--0.95 grid. The committed scripts regenerate the tables and figures; `data/` and `checkpoints/` are intentionally ignored and must be supplied/generated locally.

The initial transformer checkpoint and loss log came from the five-epoch weekly-stride command above. It is preliminary; the next scheduled experiment is daily-stride training with early stopping, not a rerun of the completed evaluation.
