#!/bin/bash
#SBATCH --job-name=llm-bn-ollama-a40
#SBATCH --qos=desktopq
#SBATCH --partition=desktop
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=240000M
#SBATCH --time=12:00:00
#SBATCH --ntasks=13
#SBATCH --constraint=r9
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "[INFO] Host: $(hostname)  JobID: ${SLURM_JOB_ID}  Time: $(date)"
cd "$SLURM_SUBMIT_DIR"

# --- Activate your normal environment ---
source /home/mvo1/lb64_scratch/miniconda3/etc/profile.d/conda.sh
conda activate llm-bn

# --- Start Ollama (same as manual) ---
LOG="ollama-${SLURM_JOB_ID}.log"
echo "[INFO] Starting ollama serve ..."
srun --exclusive -N1 -n1 ollama serve >"$LOG" 2>&1 &

# Stop Ollama when job finishes
trap 'pkill -f "ollama serve" || true' EXIT

# --- Wait until Ollama is ready ---
for i in {1..60}; do
  if curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
    echo "[INFO] Ollama is up."
    break
  fi
  sleep 3
done

# --- Run your benchmark ---
srun python /home/mvo1/lb64_scratch/projects/llm-bn/benchmarking_all_models.py

echo "[INFO] Done at $(date)"