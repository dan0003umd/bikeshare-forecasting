# Evidence Reproduction

Run from the repository root after activating `.\.venv\Scripts\Activate.ps1`:

```powershell
python -m src.check_trip_availability
python -m src.probe_weather_coverage
python -m src.baselines --model seasonal_rolling
python -m src.baselines --model seasonal_frozen
python -m src.baselines --model negative_binomial
python -m src.train_transformer --daily-stride --patience 3 --run-name transformer_daily_embed_noweather
python -m src.evaluate_transformer --checkpoint checkpoints/transformer_daily_embed_noweather_best.pt --model-name hf_transformer_daily_embed_noweather --batch-size 32
python -m src.train_transformer --daily-stride --no-station-embedding --patience 3 --run-name transformer_daily_noembed_noweather
python -m src.evaluate_transformer --checkpoint checkpoints/transformer_daily_noembed_noweather_best.pt --model-name hf_transformer_daily_noembed_noweather --batch-size 32
python -m src.train_transformer --daily-stride --observed-weather-oracle --patience 3 --run-name transformer_daily_embed_observed_weather_oracle
python -m src.evaluate_transformer --checkpoint checkpoints/transformer_daily_embed_observed_weather_oracle_best.pt --model-name hf_transformer_daily_embed_observed_weather_oracle --observed-weather-oracle --batch-size 32
python -m src.make_figures
python -m src.build_interim_evidence
python -m pytest -q
```

The checked-in experiment settings are `src/config.py`: seed 42, training 2023--2024, validation 2025, test locked behind `--final`, 50 stations, and the 19-quantile 0.05--0.95 grid. The committed scripts regenerate the tables and figures; `data/` and `checkpoints/` are intentionally ignored and must be supplied/generated locally.

The initial transformer checkpoint and loss log came from a five-epoch weekly-stride run. The daily-stride commands above produce the ablation rows and their loss CSVs. The final command is an ORACLE diagnostic: it passes realized future observed weather and is not a deployable model.
