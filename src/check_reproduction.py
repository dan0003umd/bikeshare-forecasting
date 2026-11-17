"""Compare reproduced validation scores with the saved reference results."""
import csv
import sys

MODELS = [
    "seasonal_naive_rolling_8week",
    "xgboost_shared_quantile",
    "lstm_quantile",
    "hf_transformer_daily_embed_observed_weather_oracle",
    "hf_transformer_daily_embed_forecast_weather_lead1",
    "hf_transformer_daily_embed_noweather",
]
COLUMNS = ("pinball_loss", "crps", "coverage_90", "median_mae")


def tolerance(model):
    return 1e-6 if model.startswith(("seasonal_naive", "lstm")) else 1e-4


def read(path):
    with open(path, newline="") as handle:
        return {row["model"]: row for row in csv.DictReader(handle)}


def main(reference_path, new_path):
    reference, new = read(reference_path), read(new_path)
    failed = False
    for model in MODELS:
        if model not in reference or model not in new:
            print(f"MISSING  {model}")
            failed = True
            continue
        ref, got = reference[model], new[model]
        if int(float(ref["n_observations"])) != int(float(got["n_observations"])):
            print(f"FAIL     {model}: n_observations differs")
            failed = True
            continue
        worst = max(abs(float(ref[c]) - float(got[c])) for c in COLUMNS)
        ok = worst <= tolerance(model)
        failed = failed or not ok
        print(f"{'OK  ' if ok else 'FAIL'}     {model}: max abs diff {worst:.2e} (tolerance {tolerance(model):.0e})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])