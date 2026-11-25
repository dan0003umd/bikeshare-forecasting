#!/usr/bin/env python3
"""Build dashboard/index.html from saved validation predictions and score tables.

Usage (from the repo root):
  python dashboard/build_dashboard.py --dry-run
  python dashboard/build_dashboard.py --predictions-dir data/predictions --returns-predictions-dir data/predictions/returns --out dashboard/index.html

Validation data only. Optional --test-dir adds results_test_2026_<model>.csv scoreboards.
"""
import argparse
import base64
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TZ = "America/New_York"
QCOLS = ["q05", "q10", "q15", "q20", "q25", "q30", "q35", "q40", "q45", "q50",
         "q55", "q60", "q65", "q70", "q75", "q80", "q85", "q90", "q95"]
PICK = [0, 4, 9, 14, 18]  # q05, q25, q50, q75, q95
TAUS = np.round(np.arange(1, 20) * 0.05, 2)
HOURS = 168

MODEL_ORDER = [
    "seasonal_naive_rolling_8week",
    "xgboost_shared_quantile",
    "lstm_quantile",
    "hf_transformer_daily_embed_noweather",
    "hf_transformer_daily_embed_forecast_weather_lead1",
    "hf_transformer_daily_embed_observed_weather_oracle",
]

# Edit these patterns if file names differ on the machine you run on.
# Checkouts are searched in --predictions-dir, returns in --returns-predictions-dir.
MODEL_FILES = {
    "seasonal_naive_rolling_8week": ["seasonal_naive_rolling_8week_validation_2025.parquet"],
    "xgboost_shared_quantile": ["xgboost_shared_quantile_validation_2025.parquet"],
    "lstm_quantile": ["lstm_quantile*validation_2025.parquet", "lstm*validation_2025.parquet"],
    "hf_transformer_daily_embed_noweather": ["hf_transformer_daily_embed_noweather_validation_2025.parquet"],
    "hf_transformer_daily_embed_forecast_weather_lead1": [
        "hf_transformer_daily_embed_forecast_weather_lead1*validation_2025.parquet",
        "*forecast_weather*validation_2025.parquet"],
    "hf_transformer_daily_embed_observed_weather_oracle": [
        "hf_transformer_daily_embed_observed_weather_oracle_validation_2025.parquet"],
}

# Names the score CSVs may use for each model (exact match first).
SCORE_ALIASES = {
    "seasonal_naive_rolling_8week": ["seasonal_naive_rolling_8week"],
    "xgboost_shared_quantile": ["xgboost_shared_quantile"],
    "lstm_quantile": ["lstm_quantile"],
    "hf_transformer_daily_embed_noweather": ["hf_transformer_daily_embed_noweather"],
    "hf_transformer_daily_embed_forecast_weather_lead1": [
        "hf_transformer_daily_embed_forecast_weather_lead1",
        "hf_transformer_daily_embed_forecast_weather"],
    "hf_transformer_daily_embed_observed_weather_oracle": [
        "hf_transformer_daily_embed_observed_weather_oracle"],
}
# Fallback: unique name containing all tokens and none of the excluded tokens.
SCORE_TOKENS = {
    "seasonal_naive_rolling_8week": (["seasonal", "rolling"], []),
    "xgboost_shared_quantile": (["xgboost", "shared"], []),
    "lstm_quantile": (["lstm"], []),
    "hf_transformer_daily_embed_noweather": (["daily_embed", "noweather"], ["oracle", "seed", "long"]),
    "hf_transformer_daily_embed_forecast_weather_lead1": (["forecast_weather"], ["oracle"]),
    "hf_transformer_daily_embed_observed_weather_oracle": (["oracle"], []),
}
PINBALL_TOL = {"seasonal_naive_rolling_8week": 1e-6}
DEFAULT_TOL = 1e-4

DST_DAYS = [pd.Timestamp("2025-03-09"), pd.Timestamp("2025-11-02")]
TARGET_REPORT_DIR = {"checkouts": "interim_evidence", "returns": "returns_evidence"}


def match_model(name, key, candidates):
    """Return True if csv model name 'name' belongs to model 'key'."""
    if name in SCORE_ALIASES[key]:
        return True
    toks, excl = SCORE_TOKENS[key]
    exact_elsewhere = any(name in v for k, v in SCORE_ALIASES.items() if k != key)
    if exact_elsewhere:
        return False
    low = name.lower()
    if all(t in low for t in toks) and not any(e in low for e in excl):
        # only accept the token match when it is unambiguous
        hits = [c for c in candidates if all(t in c.lower() for t in toks)
                and not any(e in c.lower() for e in excl)
                and not any(c in v for k, v in SCORE_ALIASES.items() if k != key)]
        return len(hits) == 1
    return False


def model_for_name(name, candidates):
    for key in MODEL_ORDER:
        if match_model(name, key, candidates):
            return key
    return None


def find_files(pred_dir, patterns):
    if pred_dir is None:
        return None
    for pat in patterns:
        hits = sorted(glob.glob(str(Path(pred_dir) / pat)))
        if hits:
            return hits[0]
    return None


def read_scores(csv_path):
    """Return {model_key: {...}} from a results CSV (validation or test)."""
    if isinstance(csv_path, pd.DataFrame):
        df = csv_path
    else:
        p = Path(csv_path)
        if not p.exists():
            return {}
        df = pd.read_csv(p)
    names = [str(n) for n in df["model"].unique()]
    out = {}
    for key in MODEL_ORDER:
        rows = df[df["model"].map(lambda n: match_model(str(n), key, names))]
        if rows.empty:
            continue
        r = rows.iloc[0]
        def g(col):
            return float(r[col]) if col in rows.columns and pd.notna(r[col]) else None
        out[key] = {
            "pinball": g("pinball_loss"), "crps": g("crps"),
            "cov90": g("coverage_90"), "mae": g("median_mae"),
            "n": int(r["n_observations"]) if "n_observations" in rows.columns and pd.notna(r["n_observations"]) else None,
        }
    return out


def read_pairs(csv_path):
    p = Path(csv_path)
    if not p.exists():
        return []
    df = pd.read_csv(p)
    names = sorted(set(df["model_a"].astype(str)) | set(df["model_b"].astype(str)))
    mapping = {n: model_for_name(n, names) for n in names}
    out = []
    for _, r in df.iterrows():
        a, b = mapping.get(str(r["model_a"])), mapping.get(str(r["model_b"]))
        if a is None or b is None or str(r["metric"]) != "pinball":
            continue
        out.append({"a": a, "b": b, "d": float(r["difference_a_minus_b"]),
                    "lo": float(r["ci95_low"]), "hi": float(r["ci95_high"])})
    return out


def raw_losses(y, q):
    d = y[:, None] - q
    return np.maximum(TAUS * d, (TAUS - 1.0) * d)


def read_validation_weather(path):
    """Read only weather rows before the locked-test calendar year."""
    weather_path = Path(path)
    schema = pq.ParquetFile(weather_path).schema_arrow
    ts_type = schema.field("ts_utc").type
    if not pa.types.is_timestamp(ts_type):
        raise TypeError(f"weather_hourly.parquet ts_utc must be a timestamp, got {ts_type}")
    cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    weather = pd.read_parquet(
        weather_path,
        columns=["ts_utc", "temperature_2m", "precipitation"],
        filters=[("ts_utc", "<", cutoff)],
    )
    weather["ts_utc"] = pd.to_datetime(weather["ts_utc"], utc=True)
    if (weather["ts_utc"] >= cutoff).any():
        raise RuntimeError("weather filter admitted a row on or after 2026-01-01 UTC")
    return weather


def quantile_diagnostics(y, q):
    """Validation-only empirical calibration and raw quantile-quality checks."""
    valid = np.isfinite(y) & np.isfinite(q).all(axis=1)
    qv = q[valid]
    yv = y[valid]
    crossings = qv[:, 1:] < qv[:, :-1]
    return {
        "negative_row_share": float(np.mean(np.any(qv < 0, axis=1))) if len(qv) else None,
        "crossing_pair_share": float(np.mean(crossings)) if len(qv) else None,
        "calibration": [float(np.mean(yv <= qv[:, i])) if len(qv) else None
                        for i in range(len(QCOLS))],
        "zero_actual_share": float(np.mean(yv == 0)) if len(qv) else None,
    }


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr.astype("<i2")).tobytes()).decode("ascii")


def hour_label(ts):
    h = ts.hour % 12 or 12
    return f"{ts.strftime('%a %b')} {ts.day}, {h} {'AM' if ts.hour < 12 else 'PM'}"


def pick_weeks(weather, events):
    wl = weather.copy()
    local = wl["ts_utc"].dt.tz_convert(TZ)
    wl["date"] = local.dt.tz_localize(None).dt.normalize()
    wl = wl[(wl["date"] >= "2025-01-01") & (wl["date"] <= "2025-12-31")]
    wl["wk"] = wl["date"] - pd.to_timedelta(wl["date"].dt.dayofweek, unit="D")
    agg = wl.groupby("wk").agg(temp=("temperature_2m", "mean"), rain=("precipitation", "sum"),
                               n=("temperature_2m", "size"))
    bad = {d - pd.Timedelta(days=d.dayofweek) for d in DST_DAYS}
    ok = agg[(agg.index >= "2025-01-06") & (agg.index <= "2025-12-22") & (agg["n"] >= 160)
             & (~agg.index.isin(list(bad)))].copy()
    chosen, reasons = [], []

    def take(order, reason):
        for wk in order:
            if wk not in chosen:
                chosen.append(wk)
                reasons.append(reason)
                return

    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    ev["flag"] = ev["event_flag"].astype(str).str.lower().eq("true")
    cherry = ev[ev["event_type"].astype(str).str.lower().str.contains("cherry") & ev["flag"]
                & (ev["date"].dt.year == 2025)]
    if len(cherry):
        cw = cherry["date"] - pd.to_timedelta(cherry["date"].dt.dayofweek, unit="D")
        counts = cw.value_counts()
        counts = counts[counts.index.isin(ok.index)]
        if len(counts):
            take(list(counts.sort_values(ascending=False, kind="stable").index), "Cherry Blossom Festival")
    take(list(ok.sort_values("temp", ascending=False).index), "Hottest week")
    take(list(ok.sort_values("rain", ascending=False).index), "Wettest week")
    take(list(ok.sort_values("temp", ascending=True).index), "Coldest week")
    return list(zip(chosen, reasons)), ev[ev["flag"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-dir", default="data/predictions")
    ap.add_argument("--returns-predictions-dir", default="data/predictions/returns")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--events", default="data/external/events.csv")
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--basemap", default=str(Path(__file__).resolve().parent / "basemap_dc.json"),
                    help="offline street basemap made by dashboard/fetch_basemap.py (optional)")
    ap.add_argument("--template", default=str(Path(__file__).resolve().parent / "template.html"))
    ap.add_argument("--out", default="dashboard/index.html")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    dirs = {"checkouts": a.predictions_dir, "returns": a.returns_predictions_dir}
    found = {}
    print(f"{'target':10s} {'model':52s} status")
    for t in ("checkouts", "returns"):
        for key in MODEL_ORDER:
            f = find_files(dirs[t], MODEL_FILES[key]) if Path(dirs[t]).is_dir() else None
            found[(t, key)] = f
            print(f"{t:10s} {key:52s} {'FOUND ' + f if f else 'MISSING'}")
    if a.dry_run:
        return 0

    proc = Path(a.processed_dir)
    mapping = pd.read_csv(proc / "station_metadata_mapping.csv", dtype={"station_id": str})
    info = pd.read_csv(proc / "station_info.csv", dtype=str)
    names = dict(zip(info["short_name"], info["name"]))
    matched = int(mapping["station_id"].isin(names).sum())
    print(f"STATION JOIN: {matched}/{len(mapping)} selected station_id values matched station_info.short_name")
    stations = [{"id": r.station_id, "name": names.get(r.station_id, "Station " + r.station_id),
                 "lat": float(r.lat), "lon": float(r.lon), "cap": int(r.capacity)}
                for r in mapping.itertuples()]
    sidx = {s["id"]: i for i, s in enumerate(stations)}
    n_st = len(stations)

    weather = read_validation_weather(proc / "weather_hourly.parquet")
    events = pd.read_csv(a.events)
    weeks, ev_flagged = pick_weeks(weather, events)
    if not weeks:
        print("ERROR: no featured weeks could be chosen")
        return 1
    tmax = float(weather["temperature_2m"].max())
    temp_unit = "F" if tmax > 60 else "C"

    base = pd.Timestamp("2025-01-01", tz="UTC")
    week_meta, week_h0 = [], []
    wser = weather.set_index("ts_utc")
    for wk, reason in weeks:
        start_utc = pd.Timestamp(wk).tz_localize(TZ).tz_convert("UTC")
        idx = pd.date_range(start_utc, periods=HOURS, freq="h")
        local = idx.tz_convert(TZ)
        days = [(wk + pd.Timedelta(days=i)) for i in range(7)]
        end = days[-1]
        evnames = []
        for d in days:
            hit = ev_flagged[ev_flagged["date"] == d]
            evnames.append(str(hit["event_type"].iloc[0]).replace("_", " ").title() if len(hit) else None)
        w = wser.reindex(idx)
        week_meta.append({
            "label": f"{wk:%b} {wk.day} to {end:%b} {end.day}, 2025: {reason}",
            "short": reason,
            "days": [f"{d:%a} {d:%b} {d.day}" for d in days],
            "hours": [hour_label(ts) for ts in local],
            "weatherTsUtc": [ts.isoformat() for ts in idx],
            "events": evnames,
            "temp": [None if pd.isna(v) else round(float(v), 1) for v in w["temperature_2m"]],
            "rain": [None if pd.isna(v) else round(float(v), 2) for v in w["precipitation"]],
        })
        week_h0.append(int((start_utc - base) / pd.Timedelta(hours=1)))

    targets = {}
    for t in ("checkouts", "returns"):
        rep = Path(a.reports_dir) / TARGET_REPORT_DIR[t]
        scores = read_scores(rep / "results_validation_2025.csv")
        pairs = read_pairs(rep / "paired_bootstrap_validation_2025.csv")
        entry = {"scores": scores, "pairs": pairs, "forecasts": {}, "pinball": {}, "actual": None,
                 "quantileQuality": {}, "calibration": {}, "zeroActualShare": None}
        for key in MODEL_ORDER:
            f = found[(t, key)]
            if not f:
                continue
            df = pd.read_parquet(f, columns=["station_id", "ts_utc", "actual"] + QCOLS)
            ts = pd.to_datetime(df["ts_utc"], utc=True)
            y = df["actual"].to_numpy(dtype="float64")
            q = df[QCOLS].to_numpy(dtype="float64")
            diag = quantile_diagnostics(y, q)
            entry["quantileQuality"][key] = {
                "negativeRowShare": diag["negative_row_share"],
                "crossingPairShare": diag["crossing_pair_share"],
            }
            entry["calibration"][key] = diag["calibration"]
            if entry["zeroActualShare"] is None:
                entry["zeroActualShare"] = diag["zero_actual_share"]
            print(
                f"QUANTILE CHECK {t}/{key}: negative rows "
                f"{diag['negative_row_share']:.1%}; crossing adjacent pairs "
                f"{diag['crossing_pair_share']:.1%}"
            )
            losses = raw_losses(y, q)
            raw = float(np.nanmean(losses))
            if key in scores and scores[key]["pinball"] is not None:
                tol = PINBALL_TOL.get(key, DEFAULT_TOL)
                ref = scores[key]["pinball"]
                verdict = "PASS" if abs(raw - ref) <= tol else "FAIL"
                print(f"PINBALL CHECK {t}/{key}: {verdict} raw={raw:.12f} csv={ref:.12f} tol={tol:g}")
            else:
                print(f"PINBALL CHECK {t}/{key}: SKIPPED (no score row in CSV) raw={raw:.12f}")
            si = df["station_id"].astype(str).map(sidx).fillna(-1).astype(int).to_numpy()
            hh = ((ts - base) // pd.Timedelta(hours=1)).to_numpy()
            rowloss = np.nanmean(losses, axis=1)
            fw, pw = [], []
            act_needed = entry["actual"] is None
            acts = []
            for h0 in week_h0:
                sel = (hh >= h0) & (hh < h0 + HOURS) & (si >= 0)
                sw, hw = si[sel], (hh[sel] - h0).astype(int)
                disp = np.clip(np.sort(q[sel], axis=1), 0, 3200)[:, PICK]
                val = np.where(np.isnan(disp), -1, np.rint(disp * 10))
                arr = np.full((n_st, HOURS, 5), -1, dtype=np.int16)
                arr[sw, hw, :] = val.astype(np.int16)
                fw.append(b64(arr))
                cnt = np.bincount(sw, minlength=n_st).astype(float)
                tot = np.bincount(sw, weights=rowloss[sel], minlength=n_st)
                with np.errstate(invalid="ignore", divide="ignore"):
                    pm = np.where(cnt > 0, tot / cnt, np.nan)
                pw.append([None if np.isnan(v) else round(float(v), 4) for v in pm])
                if act_needed:
                    ac = np.full((n_st, HOURS), -1, dtype=np.int16)
                    yy = y[sel]
                    ac[sw, hw] = np.where(np.isnan(yy), -1, np.rint(yy)).astype(np.int16)
                    acts.append(b64(ac))
            entry["forecasts"][key] = fw
            entry["pinball"][key] = pw
            if act_needed:
                entry["actual"] = acts
            del df, q, losses
        targets[t] = entry

    test = {"checkouts": {}, "returns": {}}
    if a.test_dir:
        for t, d in (("checkouts", Path(a.test_dir)), ("returns", Path(a.test_dir) / "returns")):
            frames = [pd.read_csv(p) for p in sorted(glob.glob(str(d / "results_test_2026_*.csv")))]
            if frames:
                test[t] = read_scores(pd.concat(frames, ignore_index=True))
    print("TEST SCORES LOADED:", {t: sorted(v) for t, v in test.items()})

    basemap = None
    bm_path = Path(a.basemap)
    if bm_path.exists():
        basemap = json.loads(bm_path.read_text(encoding="utf-8"))
        print(f"BASEMAP: embedded {bm_path} ({bm_path.stat().st_size / 1024:.0f} KB)")
    else:
        print(f"BASEMAP: {bm_path} not found; the map will show a plain grid. "
              "Run: python dashboard/fetch_basemap.py")

    data = {"stations": stations, "stationCount": n_st, "quantiles": [float(t) for t in TAUS],
            "weeks": week_meta, "tempUnit": temp_unit,
            "targets": targets, "test": test, "basemap": basemap}
    html = Path(a.template).read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("__DATA__", payload)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"BUILT {out} ({out.stat().st_size:,} bytes); featured weeks:")
    for w in week_meta:
        print("  " + w["label"])
    return 0


if __name__ == "__main__":
    sys.exit(main())