#!/bin/bash
# Run vLLM on one GPU of a Slurm cluster and point the gateway at it.
#
#   sbatch deploy/vllm-slurm.sh
#   # find the node:   squeue -u $USER -o "%N"   (or read vllm-<jobid>.out)
#   # from your laptop: ssh -N -L 8000:<node>:8000 <login-node>
#   # then in .env:     VLLM_ENABLED=true  VLLM_BASE_URL=http://host.docker.internal:8000/v1
#   #   (or http://localhost:8000/v1 when running the gateway outside Docker)
#
# Adjust partition/account/module lines to your cluster.

#SBATCH --job-name=vllm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=vllm-%j.out

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8000}"
PREFIX_CACHING="${PREFIX_CACHING:-1}"     # PREFIX_CACHING=0 sbatch ... for the "off" arm of the experiment

# module load cuda/12.1                    # cluster-specific
# source ~/venvs/vllm/bin/activate         # a venv with `pip install vllm`
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

EXTRA=()
if [[ "$PREFIX_CACHING" == "1" ]]; then EXTRA+=(--enable-prefix-caching); else EXTRA+=(--no-enable-prefix-caching); fi

echo "vLLM on $(hostname):$PORT model=$MODEL prefix_caching=$PREFIX_CACHING"
exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  "${EXTRA[@]}"
