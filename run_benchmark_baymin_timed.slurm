#!/bin/bash
#SBATCH --job-name=baymin-benchmark
#SBATCH --qos=desktopq
#SBATCH --partition=desktop
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=240000M
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --constraint=r9
# Write the initial SLURM wrapper logs to cwd; the script will re-route into sbatch_log/
#SBATCH --output=sbatch_log/%x-%j.out
#SBATCH --error=sbatch_log/%x-%j.err

### -------------------- Strict mode & env --------------------
set -euo pipefail
mkdir -p sbatch_log

# Re-route all subsequent stdout/stderr into our folder (and keep it on screen)
exec > >(tee -a "sbatch_log/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "sbatch_log/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" >&2)

echo "[BOOT] $(date) host=$(hostname) job=${SLURM_JOB_ID}"
cd "$SLURM_SUBMIT_DIR"

# --- Activate your environment (adjust paths/modules as needed)
source /home/mvo1/lb64_scratch/miniconda3/etc/profile.d/conda.sh
conda activate llm-bn

# If your site uses environment modules for CUDA, uncomment and set the right version
# module load cuda/12.1

# Ollama host/port bound to loopback (good for single-node jobs)
export OLLAMA_HOST="127.0.0.1:11434"

### -------------------- Functions --------------------
wait_for_ollama() {
  local retries=60
  local delay=3
  for ((i=1; i<=retries; i++)); do
    if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
      echo "[INFO] Ollama is up after $((i*delay))s"
      return 0
    fi
    sleep "$delay"
  done
  return 1
}

start_ollama_step() {
  local log_file="$1"
  echo "[INFO] Starting ollama via srun step…" | tee -a "$log_file"
  # Use srun so the process is in the GPU cgroup and sees /dev/nvidia*
  srun --ntasks=1 --gres=gpu:A40:1 --exclusive -u bash -lc '
    echo "[STEP] $(hostname) $(date)"
    echo "[STEP] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    nvidia-smi || true
    exec ollama serve
  ' >"$log_file" 2>&1 &
  echo $!  # return PID of backgrounded srun step wrapper
}

stop_ollama_step() {
  local step_pid="$1"
  echo "[INFO] Stopping ollama step pid=${step_pid}"
  kill -TERM "${step_pid}" 2>/dev/null || true
  wait "${step_pid}" 2>/dev/null || true
}

### -------------------- 3-hour restart loop --------------------
TOTAL_HOURS=48
INTERVAL_HOURS=3
ITERATIONS=$(( TOTAL_HOURS / INTERVAL_HOURS ))

echo "[INFO] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

for ((cycle=1; cycle<=ITERATIONS; cycle++)); do
  CYCLE_LOG="sbatch_log/ollama-${SLURM_JOB_ID}-cycle${cycle}.log"
  echo "==================== CYCLE ${cycle}/${ITERATIONS} @ $(date) ====================" | tee -a "$CYCLE_LOG"

  # 1) Start Ollama in its own SLURM step (GPU-visible)
  OLLAMA_STEP_PID="$(start_ollama_step "$CYCLE_LOG")"

  # 2) Wait for readiness
  if ! wait_for_ollama; then
    echo "[ERROR] Ollama did not become ready in time. See ${CYCLE_LOG}" >&2
    stop_ollama_step "${OLLAMA_STEP_PID}"
    exit 1
  fi

  # 3) Run your benchmark for this cycle
  echo "[INFO] Running benchmark (cycle ${cycle})…" | tee -a "$CYCLE_LOG"
  # If Python uses the GPU too, keep it in a separate step; inherit the allocation
  # -u keeps output unbuffered; tee appends to the same cycle log
  srun --ntasks=1 -u python /home/mvo1/lb64_scratch/projects/llm-bn/benchmarking_baymin.py | tee -a "$CYCLE_LOG"

  # 4) Stop Ollama and sleep until next restart boundary
  stop_ollama_step "${OLLAMA_STEP_PID}"

  if (( cycle < ITERATIONS )); then
    # Ensure a 3h cadence: sleep the remaining time of the interval if your
    # benchmark was shorter; if it ran longer than 3h, continue immediately.
    CYCLE_SECS=$(( INTERVAL_HOURS * 3600 ))
    # Record end time and compute actual duration by reading timestamps:
    # (simpler heuristic: always sleep fixed 3h)
    echo "[INFO] Sleeping ${INTERVAL_HOURS}h before next restart…" | tee -a "$CYCLE_LOG"
    sleep "${CYCLE_SECS}"
  fi
done

echo "[INFO] All cycles completed @ $(date)"