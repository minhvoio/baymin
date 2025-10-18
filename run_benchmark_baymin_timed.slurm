#!/bin/bash
#SBATCH --job-name=baymin-benchmark
#SBATCH --qos=desktopq
#SBATCH --partition=desktop
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=240000M
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --constraint=r9
#SBATCH --output=sbatch_log/%x-%j.out
#SBATCH --error=sbatch_log/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p sbatch_log

echo "[BOOT] $(date) host=$(hostname) job=${SLURM_JOB_ID}"
export OLLAMA_HOST="127.0.0.1:11434"
export PYTHONUNBUFFERED=1
# If your site needs modules: module load cuda/12.1

# Your Python env (only for python step)
source /home/mvo1/lb64_scratch/miniconda3/etc/profile.d/conda.sh
conda activate llm-bn

echo "[DBG] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

TOTAL_HOURS=48
INTERVAL_HOURS=4
ITERATIONS=$(( TOTAL_HOURS / INTERVAL_HOURS ))

for ((cycle=1; cycle<=ITERATIONS; cycle++)); do
  CYCLE_LOG="sbatch_log/ollama-${SLURM_JOB_ID}-cycle${cycle}.log"
  echo "===== CYCLE ${cycle}/${ITERATIONS} @ $(date) =====" | tee -a "$CYCLE_LOG"

  # Start ollama with CLEAN env (avoid conda CUDA libs)
  export OLLAMA_LOG_LEVEL=DEBUG
  export OLLAMA_NUM_GPU=1
  ( unset LD_LIBRARY_PATH CONDA_PREFIX CONDA_DEFAULT_ENV; \
    # module load cuda/12.1  # if required on your cluster
    ollama serve \
  ) >"$CYCLE_LOG" 2>&1 &
  OLLAMA_PID=$!

  # readiness probe
  for i in {1..60}; do
    curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1 && break
    sleep 3
    [[ $i -eq 60 ]] && { echo "[ERROR] Ollama not ready" | tee -a "$CYCLE_LOG"; kill -TERM "$OLLAMA_PID"; wait "$OLLAMA_PID"; exit 1; }
  done
  echo "[INFO] Ollama is up" | tee -a "$CYCLE_LOG"

  # Run benchmark (uses conda Python)
  python -u /home/mvo1/lb64_scratch/projects/llm-bn/benchmarking_baymin.py | tee -a "$CYCLE_LOG"

  # Stop and wait
  kill -TERM "$OLLAMA_PID" 2>/dev/null || true
  wait "$OLLAMA_PID" 2>/dev/null || true

  # Sleep to the next window
  (( cycle < ITERATIONS )) && { echo "[INFO] Sleeping ${INTERVAL_HOURS}h"; sleep $((INTERVAL_HOURS*3600)); }
done

echo "[INFO] All cycles completed @ $(date)"