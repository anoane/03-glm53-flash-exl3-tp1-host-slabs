#!/usr/bin/env bash
# memguard: kill the vLLM container BEFORE the kernel OOM-killer has to.
#
# With no swap, host overcommit ends in a fast clean OOM-kill -- but only after the
# kernel has already reclaimed everything it can, which stalls the box for a while.
# Polling `available` and killing at a threshold fails earlier and cleaner, and records
# the peak so the budget can be tuned from evidence instead of guesses.
NAME="${1:-glm53-vllm-tp1}"
MIN_AVAIL_GB="${2:-8}"
LOG="${3:-/root/workspace/bench_c_review/results/memguard.log}"
peak_used=0; peak_gpu=0
echo "$(date -u +%T) memguard start: container=$NAME min_avail=${MIN_AVAIL_GB}G" | tee -a "$LOG"
while true; do
  read -r _ total used _ _ _ avail < <(free -g | awk '/^Mem:/')
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ "${used:-0}" -gt "$peak_used" ] && peak_used=$used
  [ "${gpu:-0}" -gt "$peak_gpu" ] && peak_gpu=$gpu
  # Composition of host memory, so a kill explains itself: pinned slabs show as Mlocked/
  # Unevictable, torch's pinned cache and IPC as Shmem, file staging as Cached.
  mi=$(awk '/^(Shmem|Mlocked|Unevictable|Cached|AnonPages|MemAvailable):/{printf "%s=%dG ", substr($1,1,length($1)-1), $2/1048576}' /proc/meminfo)
  if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "$(date -u +%T) avail=${avail}G used=${used}G gpu=${gpu}MiB | $mi" >> "$LOG.trace"
    if [ "${avail:-99}" -lt "$MIN_AVAIL_GB" ]; then
      echo "$(date -u +%T) MEMGUARD KILL: host avail=${avail}G < ${MIN_AVAIL_GB}G  (peak host used ${peak_used}G, peak gpu ${peak_gpu}MiB) | $mi" | tee -a "$LOG"
      docker kill "$NAME" >/dev/null 2>&1
      exit 2
    fi
  fi
  sleep 3
done
