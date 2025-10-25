#!/bin/bash
#SBATCH --job-name=net1030
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=240000M
#SBATCH --time=24:00:00
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
# module load cuda/12.1  # if required on your cluster

# Conda for python only (avoid leaking CUDA libs into ollama env)
source /home/mvo1/lb64_scratch/miniconda3/etc/profile.d/conda.sh
conda activate llm-bn

echo "[DBG] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

# ---------- helpers ----------
safe_kill_ollama() {
  echo "[CLEAN] Killing any leftover ollama (user=$USER) ..."
  # Try exact executable name, then command line match
  pkill -u "$USER" -TERM -x ollama 2>/dev/null || true
  pkill -u "$USER" -TERM -f 'ollama serve' 2>/dev/null || true
  # Give it a moment to exit cleanly
  sleep 2
  # Force kill if anything remains
  pkill -u "$USER" -KILL -x ollama 2>/dev/null || true
  pkill -u "$USER" -KILL -f 'ollama serve' 2>/dev/null || true
  # Show what’s left (if anything)
  pgrep -a -u "$USER" ollama || echo "[CLEAN] No ollama processes remain."
}

trap 'echo "[TRAP] Exiting—final ollama cleanup"; safe_kill_ollama' EXIT

TOTAL_HOURS=48
INTERVAL_HOURS=4   # ← change this to 3 if you want every 3 hours
ITERATIONS=$(( TOTAL_HOURS / INTERVAL_HOURS ))

# Pre-flight cleanup in case a prior run left processes
safe_kill_ollama

for ((cycle=1; cycle<=ITERATIONS; cycle++)); do
  CYCLE_LOG="sbatch_log/ollama-${SLURM_JOB_ID}-cycle${cycle}.log"
  echo "===== CYCLE ${cycle}/${ITERATIONS} @ $(date) =====" | tee -a "$CYCLE_LOG"

  # Extra safety: kill before (re)starting this cycle
  safe_kill_ollama

  # Start ollama with a clean env (avoid conda's LD_LIBRARY_PATH)
  export OLLAMA_LOG_LEVEL=DEBUG
  export OLLAMA_NUM_GPU=1
  (
    unset LD_LIBRARY_PATH CONDA_PREFIX CONDA_DEFAULT_ENV
    # module load cuda/12.1  # if your site needs it for ollama
    exec ollama serve
  ) >"$CYCLE_LOG" 2>&1 &
  OLLAMA_PID=$!

  # readiness probe
  for i in {1..60}; do
    if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
      echo "[INFO] Ollama is up" | tee -a "$CYCLE_LOG"
      break
    fi
    sleep 3
    if [[ $i -eq 60 ]]; then
      echo "[ERROR] Ollama not ready" | tee -a "$CYCLE_LOG"
      kill -TERM "$OLLAMA_PID" 2>/dev/null || true
      wait "$OLLAMA_PID" 2>/dev/null || true
      exit 1
    fi
  done

  # Run benchmark (conda Python)
  python -u /home/mvo1/lb64_scratch/projects/llm-bn/benchmarking_all_models_net1030.py | tee -a "$CYCLE_LOG"

  # Stop this cycle's ollama PID
  echo "[STOP] Stopping ollama PID ${OLLAMA_PID}" | tee -a "$CYCLE_LOG"
  kill -TERM "$OLLAMA_PID" 2>/dev/null || true
  wait "$OLLAMA_PID" 2>/dev/null || true

  # Post-cycle cleanup (catches any helper/runners)
  safe_kill_ollama

  # Sleep to the next window
  if (( cycle < ITERATIONS )); then
    echo "[INFO] Sleeping ${INTERVAL_HOURS}h" | tee -a "$CYCLE_LOG"
    sleep $((INTERVAL_HOURS*3600))
  fi
done

echo "[INFO] All cycles completed @ $(date)"
