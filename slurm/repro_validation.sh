#!/bin/bash
#SBATCH --job-name=repro-validation
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL

# Re-scores all six models on the 2025 VALIDATION split with the patched code and compares
# with the committed results. No test data is read. Saved predictions are backed up and restored.
set -e
module load python/3.10.10
unset PYTHONPATH
source /scratch/zt1/project/msml612/user/dboone12/venvs/bikeshare/bin/activate
cd /scratch/zt1/project/msml612/user/dboone12/bikeshare-forecasting

test -z "$(git status --porcelain --untracked-files=no)" || { echo "working tree not clean"; exit 1; }
echo "commit under test: $(git rev-parse HEAD)"
OUT=slurm/logs/repro-$SLURM_JOB_ID
mkdir -p "$OUT"
cp -r data/predictions data/predictions_backup
restore_predictions() {
  rm -rf data/predictions
  mv data/predictions_backup data/predictions
}
trap restore_predictions EXIT
trap 'exit 1' TERM INT

status=0
for TARGET in checkouts returns; do
  export BIKESHARE_TARGET=$TARGET
  if [ "$TARGET" = "returns" ]; then REPORTS=reports/returns_evidence; else REPORTS=reports/interim_evidence; fi
  RESULTS=$REPORTS/results_validation_2025.csv
  git show "HEAD:$RESULTS" > "$OUT/ref_$TARGET.csv"

  python -u -m src.baselines --model seasonal_rolling
  python -u -m src.xgboost_baseline --max-minutes 60
  python -u -m src.evaluate_lstm --checkpoint lstm_quantile_best.pt --model-name lstm_quantile
  python -u -m src.evaluate_transformer --checkpoint transformer_daily_embed_noweather_best.pt \
      --model-name hf_transformer_daily_embed_noweather --batch-size 32
  python -u -m src.evaluate_transformer --checkpoint transformer_daily_embed_observed_weather_oracle_best.pt \
      --model-name hf_transformer_daily_embed_observed_weather_oracle --batch-size 32
  python -u -m src.evaluate_transformer --checkpoint transformer_daily_embed_observed_weather_oracle_best.pt \
      --model-name hf_transformer_daily_embed_forecast_weather_lead1 --forecast-weather 1 --batch-size 32

  cp "$RESULTS" "$OUT/new_$TARGET.csv"
  git checkout -- "$REPORTS"
  echo "=== reproduction check: $TARGET ==="
  python -u -m src.check_reproduction "$OUT/ref_$TARGET.csv" "$OUT/new_$TARGET.csv" || status=1
done
exit $status
