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

echo "[INFO] Host: $(hostname)  JobID: ${SLURM_JOB_ID}  Time: $(date)"
cd "$SLURM_SUBMIT_DIR"

# === Environment setup ===
source /home/mvo1/lb64_scratch/miniconda3/etc/profile.d/conda.sh
conda activate llm-bn

export OLLAMA_HOST="127.0.0.1:11434"
mkdir -p sbatch_log

# === 3-hour restart loop ===
TOTAL_HOURS=48
RESTART_INTERVAL=3
ITERATIONS=$(( TOTAL_HOURS / RESTART_INTERVAL ))

for ((i=1; i<=ITERATIONS; i++)); do
  LOG="sbatch_log/ollama-${SLURM_JOB_ID}-cycle${i}.log"
  echo "[INFO] ===== Cycle $i of $ITERATIONS started at $(date) =====" | tee -a "$LOG"

  echo "[INFO] Starting ollama serve ..." | tee -a "$LOG"
  ollama serve >"$LOG" 2>&1 &
  OLLAMA_PID=$!

  # Wait for Ollama to come up
  for j in {1..60}; do
    if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
      echo "[INFO] Ollama is up." | tee -a "$LOG"
      break
    fi
    sleep 3
    if [[ $j -eq 60 ]]; then
      echo "[ERROR] Ollama did not start in time. See $LOG" >&2
      kill -TERM ${OLLAMA_PID} 2>/dev/null || true
      exit 1
    fi
  done

  # Run your benchmark for this cycle
  echo "[INFO] Running benchmark (cycle $i) ..." | tee -a "$LOG"
  srun python /home/mvo1/lb64_scratch/projects/llm-bn/benchmarking_baymin.py | tee -a "$LOG"

  # Kill Ollama and sleep until next cycle
  echo "[INFO] Stopping ollama (cycle $i) ..." | tee -a "$LOG"
  kill -TERM ${OLLAMA_PID} 2>/dev/null || true
  wait ${OLLAMA_PID} 2>/dev/null || true

  if (( i < ITERATIONS )); then
    echo "[INFO] Sleeping until next restart (3h) ..." | tee -a "$LOG"
    sleep $((RESTART_INTERVAL * 3600))
  fi
done

echo "[INFO] ===== All cycles completed at $(date) ====="