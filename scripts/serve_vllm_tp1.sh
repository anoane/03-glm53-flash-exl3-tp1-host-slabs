#!/usr/bin/env bash
# GLM-5.3-Flash tr3-4bpw on the b12x vLLM fork, adapted from the upstream TP=2 recipe to a
# SINGLE GPU. Upstream assumes two SM120 cards at --gpu-memory-utilization 0.986; the weights
# are 163.7 GiB so one 96 GiB card cannot hold them, hence --cpu-offload-gb.
#
# Removed vs upstream (all are multi-GPU constructs that are meaningless or invalid at TP=1):
#   --tensor-parallel-size 2      -> 1
#   --enable-expert-parallel      -> dropped (EP needs >1 rank)
#   --decode-context-parallel-size 2 / --dcp-comm-backend a2a -> dropped
#   VLLM_USE_B12X_DCP_A2A, VLLM_ENABLE_PCIE_ALLREDUCE, VLLM_PCIE_ALLREDUCE_BACKEND -> dropped
#   --disable-custom-all-reduce   -> dropped (no collectives at TP=1)
#   speculative-config (DFlash2)  -> dropped; draft model not fetched, and draft_tensor_parallel_size 2
#
# Kept: the b12x MoE backend, B12X_MLA_SPARSE attention, nvfp4_ds_mla KV, and the NVFP4 MLA
# calibration scales -- these are what make the checkpoint loadable at all.
set -euo pipefail
IMAGE="${IMAGE:-verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-language-only}"
MODEL="${MODEL:-/root/workspace/GLM-5.3-Flash-tr3-4bpw}"
PORT="${PORT:-8012}"
NAME="${NAME:-glm53-vllm-tp1}"
# Two offload backends exist. UVA (--cpu-offload-gb) is zero-copy from pinned memory and
# offloads non-selectively unless --cpu-offload-params names segments; prefetch groups
# layers and async-streams them, which is what MoE experts want. Either can TARGET the
# expert tensors by name segment ("experts"), which plain --cpu-offload-gb does not.
OFFLOAD_ARGS="${OFFLOAD_ARGS:-}"
# Loader working-set controls. Parallel shard staging and read-ahead prefetch were the
# ~37 GB of host memory the OOM-kill showed beyond the pinned slabs; serialising the
# loader bounds that to roughly one shard in flight.
LOAD_ARGS="${LOAD_ARGS:-}"
MAXLEN="${MAXLEN:-32768}"
UTIL="${UTIL:-0.92}"
CACHE="${CACHE:-/root/workspace/glm53-vllm-cache}"

mkdir -p "$CACHE"
docker rm -f "$NAME" >/dev/null 2>&1 || true
# VLLM_EXL3_PREFILL_BLOCK_M: upstream pins 128 for its TP=2 geometry (N=2048). At TP=1
# N doubles to 4096 and the W4A16 fused kernel finds no tile config at block 128
# (cta_m_blocks=8 fails _candidate_tile_fits for every tile); the code default is 64.
exec docker run --rm --name "$NAME" \
  --init --gpus '"device=0"' --ipc=host --shm-size 32g --network host \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_B12X_GLM_NOPE_NVFP4="${NOPE_NVFP4:-1}" \
  -e VLLM_NVFP4_MLA_DYNAMIC_SCALE=0 \
  -e VLLM_NVFP4_MLA_SCALES_FILE=/opt/glm53/calibration/glm53_nvfp4_mla_outer_scales_mtp_power2_v2.json \
  -e VLLM_EXL3_PREFILL_BLOCK_M="${PREFILL_BLOCK_M:-64}" \
  -e EXL3_GPU_SLAB_GIB="${SLAB_GIB:-0}" \
  -e VLLM_EXL3_PREFILL_TRELLIS=1 \
  -e B12X_GL53_ROUTE128_WIDE=1 \
  -e B12X_GL53_ROUTE128_HYBRID_TAIL=1 \
  -e KV_FP8_ROPE="${KV_FP8_ROPE:-0}" \
  -e OMP_NUM_THREADS=8 \
  -v "$MODEL:/model:ro" -v "$CACHE:/cache" \
  -v /root/workspace/bench_c_review/exl3_vllm_patched.py:/opt/infernal-invocation/vllm/vllm/model_executor/layers/quantization/exl3.py:ro \
  -v /root/workspace/bench_c_review/b12x_prepare_patched.py:/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a16/prepare.py:ro \
  "$IMAGE" serve /model \
  --served-model-name GLM-5.3-Flash-EXL3-4bpw \
  --host 127.0.0.1 --port "$PORT" \
  --language-model-only \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  $OFFLOAD_ARGS \
  --dtype bfloat16 \
  --load-format safetensors \
  --moe-backend b12x \
  --attention-backend B12X_MLA_SPARSE \
  --kv-cache-dtype "${KV_DTYPE:-nvfp4_ds_mla}" \
  --max-model-len "$MAXLEN" \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 \
  $LOAD_ARGS \
  --gpu-memory-utilization "$UTIL" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --generation-config /model \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  "$@"
