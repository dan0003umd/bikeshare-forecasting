#!/bin/bash
#SBATCH --job-name=train-lstm
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL

# Train the quantile LSTM baseline, then score it on every eligible 2025 day.
set -e  # stop if training fails, so we never evaluate a missing or stale checkpoint

module load python/3.10.10
unset PYTHONPATHt
source /scratch/zt1/project/msml612/user/dboone12/venvs/bikeshare/bin/activate
cd /scratch/zt1/project/msml612/user/dboone12/bikeshare-forecasting

python -u -m src.train_lstm --run-name lstm_quantile --epochs 300 --patience 10 --max-minutes 60
python -u -m src.evaluate_lstm --checkpoint lstm_quantile_best.pt --model-name lstm_quantile