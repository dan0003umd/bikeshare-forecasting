#!/bin/bash
#SBATCH --job-name=final-test
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL

# FINAL TEST RUN (reads the locked 2026 file). Run once, only after the reproduction check passed.
# Usage: sbatch slurm/final_test.sh <frozen_commit_hash>
set -e
FROZEN=$1
test -n "$FROZEN" || { echo "usage: sbatch slurm/final_test.sh <frozen_commit_hash>"; exit 1; }
module load python/3.10.10
unset PYTHONPATH
source /scratch/zt1/project/msml612/user/dboone12/venvs/bikeshare/bin/activate
cd /scratch/zt1/project/msml612/user/dboone12/bikeshare-forecasting

test "$(git rev-parse HEAD)" = "$FROZEN" || { echo "HEAD is not the frozen commit"; exit 1; }
test -z "$(git status --porcelain --untracked-files=no)" || { echo "working tree not clean"; exit 1; }
echo "frozen commit: $FROZEN"

for TARGET in checkouts returns; do
  export BIKESHARE_TARGET=$TARGET
  echo "=== final test: $TARGET ==="
  python -u -m src.baselines --model seasonal_rolling --final
  python -u -m src.xgboost_baseline --max-minutes 60 --final
  python -u -m src.evaluate_lstm --checkpoint lstm_quantile_best.pt --model-name lstm_quantile --final
  python -u -m src.evaluate_transformer --checkpoint transformer_daily_embed_noweather_best.pt \
      --model-name hf_transformer_daily_embed_noweather --batch-size 32 --final
  python -u -m src.evaluate_transformer --checkpoint transformer_daily_embed_observed_weather_oracle_best.pt \
      --model-name hf_transformer_daily_embed_observed_weather_oracle --batch-size 32 --final
  python -u -m src.evaluate_transformer --checkpoint transformer_daily_embed_observed_weather_oracle_best.pt \
      --model-name hf_transformer_daily_embed_forecast_weather_lead1 --forecast-weather 1 --batch-size 32 --final
done