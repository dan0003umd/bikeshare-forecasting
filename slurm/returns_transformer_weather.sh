#!/bin/bash
#SBATCH --job-name=ret-tf-weather
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL

# Returns rerun: weather transformer, trained on observed weather, then evaluated as the
# ORACLE (observed weather) and with lead-1 forecast weather (the deployable version).
set -e
module load python/3.10.10
unset PYTHONPATH
source /scratch/zt1/project/msml612/user/dboone12/venvs/bikeshare/bin/activate
cd /scratch/zt1/project/msml612/user/dboone12/bikeshare-forecasting
export BIKESHARE_TARGET=returns

python -u -m src.train_transformer --train-every-n-days 1 --validation-every-n-days 1 \
    --epochs 50 --patience 3 --max-minutes 150 --observed-weather-oracle \
    --run-name transformer_daily_embed_observed_weather_oracle
python -u -m src.evaluate_transformer --checkpoint transformer_daily_embed_observed_weather_oracle_best.pt \
    --model-name hf_transformer_daily_embed_observed_weather_oracle --batch-size 32
python -u -m src.evaluate_transformer --checkpoint transformer_daily_embed_observed_weather_oracle_best.pt \
    --model-name hf_transformer_daily_embed_forecast_weather_lead1 --forecast-weather 1 --batch-size 32