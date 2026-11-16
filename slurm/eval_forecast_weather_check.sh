#!/bin/bash
#SBATCH --job-name=eval-fcw-check
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL

# Reproduction check: rerun the lead-1 forecast-weather evaluation on the GPU and
# compare with the Mac result (pinball 0.570856). Expect agreement to Monte Carlo
# error, not exact equality, because GPU random sampling differs from CPU.

module load python/3.10.10
unset PYTHONPATH
source /scratch/zt1/project/msml612/user/dboone12/venvs/bikeshare/bin/activate
cd /scratch/zt1/project/msml612/user/dboone12/bikeshare-forecasting

python -u -m src.evaluate_transformer \
    --checkpoint transformer_daily_embed_observed_weather_oracle_best.pt \
    --model-name hf_transformer_daily_embed_forecast_weather_lead1_zaratan_check \
    --forecast-weather 1 \
    --batch-size 32