#!/bin/bash
#SBATCH --job-name=refit-xgboost-origin-safe
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL

# Refit the default origin-safe XGBoost model for both targets. This validation
# job uses only load_development_model_table(); it never unlocks 2026.
set -euo pipefail
module load python/3.10.10
unset PYTHONPATH
source /scratch/zt1/project/msml612/user/dboone12/venvs/bikeshare/bin/activate
cd /scratch/zt1/project/msml612/user/dboone12/bikeshare-forecasting

for TARGET in checkouts returns; do
  export BIKESHARE_TARGET="$TARGET"
  echo "=== origin-safe XGBoost validation: $TARGET ==="
  python -u -m src.xgboost_baseline --max-minutes 60
  python -u -m src.bootstrap_comparisons --draws 10000
done
