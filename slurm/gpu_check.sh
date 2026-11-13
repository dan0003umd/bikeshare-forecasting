#!/bin/bash
#SBATCH --job-name=gpu-check
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/%x-%j.out

# Same startup block we use when logging in.
module load python/3.10.10
unset PYTHONPATH
source /scratch/zt1/project/msml612/user/dboone12/venvs/bikeshare/bin/activate
cd /scratch/zt1/project/msml612/user/dboone12/bikeshare-forecasting

# What GPU did we get, and what driver version does the node run?
nvidia-smi

# Can PyTorch see the GPU and actually compute on it?
python - <<'EOF'
import torch
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device="cuda")
    print("matrix multiply on GPU ok:", (x @ x).sum().item())
EOF