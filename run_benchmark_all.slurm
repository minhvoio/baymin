#!/bin/bash
#SBATCH --job-name=llmbn-ibm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=4:00:00
#SBATCH --constraint=r9
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "[INFO] Host: $(hostname)  JobID: ${SLURM_JOB_ID}  Time: $(date)"
cd "$SLURM_SUBMIT_DIR"

# env
source /home/mvo1/lb64_scratch/miniconda3/etc/profile.d/conda.sh
conda activate llm-bn

export OLLAMA_HOST="127.0.0.1:11434"
LOG="ollama-${SLURM_JOB_ID}.log"

# 1) Start Ollama directly (no srun), in the background
echo "[INFO] Starting ollama serve ..."
ollama serve >"$LOG" 2>&1 &
OLLAMA_PID=$!
trap 'echo "[INFO] Stopping ollama"; kill -TERM ${OLLAMA_PID} 2>/dev/null || true' EXIT

# 2) Wait for it to come up
for i in {1..60}; do
  if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    echo "[INFO] Ollama is up."
    break
  fi
  sleep 3
  if [[ $i -eq 60 ]]; then
    echo "[ERROR] Ollama did not start in time. See $LOG" >&2
    exit 1
  fi
done

# 3) Run your benchmark with srun (binds the GPU/CPUs cleanly)
echo "[INFO] Running benchmark ..."
srun python /home/mvo1/lb64_scratch/projects/llm-bn/benchmarking_all_models.py

echo "[INFO] Done at $(date)"