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


def tolerance(model, metric):
    if model.startswith(("seasonal_naive", "lstm")):
        return 1e-6
    return 1e-4 if metric in ("pinball_loss", "crps") else 1e-3


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
        ref_rows, got_rows = int(float(ref["n_observations"])), int(float(got["n_observations"]))
        row_ok = ref_rows == got_rows == 435600
        print(f"{'PASS' if row_ok else 'FAIL'}     {model}: n_observations ref={ref_rows} new={got_rows} expected=435600")
        failing = [] if row_ok else ["n_observations"]
        for metric in COLUMNS:
            difference = abs(float(ref[metric]) - float(got[metric]))
            allowed = tolerance(model, metric)
            ok = difference <= allowed
            print(f"{'PASS' if ok else 'FAIL'}     {model}: {metric} abs diff {difference:.2e} (tolerance {allowed:.0e})")
            if not ok:
                failing.append(metric)
        if model == "xgboost_shared_quantile":
            print(f"INFO     {model}: time_budget_reached ref={ref.get('time_budget_reached', '')} new={got.get('time_budget_reached', '')}")
        if failing:
            print(f"FAIL     {model}: failing metrics: {', '.join(failing)}")
            failed = True
        else:
            print(f"PASS     {model}: all comparisons passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
