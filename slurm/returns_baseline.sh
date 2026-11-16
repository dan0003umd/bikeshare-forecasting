#!/bin/bash
#SBATCH --job-name=ret-baselines
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL

# Returns rerun: seasonal naive, XGBoost, and LSTM, same settings as the checkout runs.
set -e
module load python/3.10.10
unset PYTHONPATH
source /scratch/zt1/project/msml612/user/dboone12/venvs/bikeshare/bin/activate
cd /scratch/zt1/project/msml612/user/dboone12/bikeshare-forecasting
export BIKESHARE_TARGET=returns

python -u -m src.baselines --model seasonal_rolling
python -u -m src.xgboost_baseline --max-minutes 60
python -u -m src.train_lstm --run-name lstm_quantile --epochs 300 --patience 10 --max-minutes 60
python -u -m src.evaluate_lstm --checkpoint lstm_quantile_best.pt --model-name lstm_quantile