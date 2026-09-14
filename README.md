# GLM-5.3-Flash tr3-4bpw on ONE 96 GB GPU via the b12x vLLM fork

Status: **works, but not competitive.** The fork's single-GPU path was blocked by eight
independent walls; all are patched here and the server comes up at 256k context with
coherent, correct output. Decode is ~11 tok/s because host-resident experts are read over
PCIe per token. exllamav3 with CPU expert compute does 17.6 tok/s at the same quality on
the same box (see the sibling repo `04-glm53-flash-exl3-dual-gpu-1m`). Kept as a reference for the
patches and the method, not as a serving recommendation.

### Hardware this was built and measured on

| | |
|---|---|
| Host | Proxmox VE 9.2.x, kernel 7.0.x-pve, Secure Boot **disabled** |
| CPU | AMD Ryzen 9 9950X3D (16C/32T) — 24 vCPU passed to the guest |
| RAM | 160 GiB DDR5 allocated to the guest (157 GiB usable) — **no swap, deliberately** |
| GPU 0 | NVIDIA RTX PRO 6000 Blackwell Workstation — 97,887 MiB, `sm_120`, `10de:2bb1`, PCIe Gen5 x16 (~42 GB/s H2D measured), 400 W default limit / 600 W max |
| GPU 1 | NVIDIA CMP 170HX — 65,536 MiB **after unlock** (8 GB stock), `sm_80` (GA100), `10de:20c2`, PCIe **Gen2 x1 (~0.38 GB/s)**, 200 W default limit / 250 W max |
| Guest | Ubuntu 24.04.4 LTS, kernel 6.8.0-139-generic, NVIDIA driver 610.43.02 |
| Storage | 5.8 TB NVMe (~93 GB free with all packs resident) |

**This recipe uses GPU 0 only**, with ~50 expert slabs left in host-registered DDR5 and read
over PCIe per token. The CMP 170HX is not involved. It was tuned when the guest had 128 GB
(host 107 GB used / 17 GB free at steady state); the box now runs 160 GiB.

> **This is a point-in-time recipe and may be slightly outdated or incomplete.**
> It was transcribed from a working system rather than written as a clean-room guide: driver,
> engine and image versions move quickly, some steps that were obvious in the moment are
> under-documented, and a few numbers were measured once rather than averaged. Every measured
> figure below is specific to the hardware in the table above — on a different PCIe topology,
> a different RAM size, or a card without the CMP's x1 bottleneck, the tuning will differ.
> Read it as a worked example with its reasoning shown, not as a turnkey script.

## What was blocking single-GPU, and the fix for each

| # | wall | root cause | fix |
|---|---|---|---|
| 1 | `rank-sliced EXL3 checkpoint TP does not match runtime: checkpoint=2, runtime=1` | metadata is **hardcoded** in `exl3.py:1268` and declares `source_layout="unsliced_tp_stream"` — the tensors are full; `tp: 2` is provenance | `ep_from_unsliced` no longer requires `use_ep` |
| 2 | host OOM at ~118 GB Shmem | PyTorch's pinned allocator rounds every slab to the next power of two (2.25→4 GiB) | allocate pageable at exact size + `cudaHostRegister(Portable\|Mapped)`; Shmem 0 |
| 3 | `expert payload lost its slab alias` after load | `process_weights_after_loading` re-homes every `exl3_tensors` entry to the layer device, copying host slabs back to GPU | skip params that have `exl3_backing` |
| 4 | `full-rotation W4A16 tile selection requires CUDA` | prepare path derives kernel device from `w13.device` (storage) at 4 sites + `_pointer_table` | route through the layer's compute device |
| 5 | b12x `trellis_t256 W4A16 weights require CUDA storage` | `prepare_trellis256_moe_weights` asserts storage==CUDA before consulting its `device` kwarg | explicit `device` is authoritative; storage may be host; caller passes it |
| 6 | b12x `trellis_t256 w13 must be on cuda:0, got cpu` | `_trellis256_flat_native_view` — pure validation returning a zero-copy view | accept cpu storage for a cuda compute device |
| 7 | scale slabs would spill to host | budget spilled every slab once full, incl. ~1 MB suh/svh | slabs < 64 MiB always stay on GPU |
| 8 | `no valid W4A16 tile config for M/N/K=2048/4096/4096, moe_block_size=128` | upstream's `PREFILL_BLOCK_M=128` targets a hand-qualified TP=2 (N=2048) schedule; at TP=1 N=4096 falls to the generic path and `_shared_memory_footprint` scales with `cta_m_blocks` | use the code default **64** |

Why in-place host slabs work at all: the b12x kernels read expert tiles by address through
a pointer table (`cp.async`), and `prepare_*` returns only metadata (codebook, bits,
rotations, tile config) — it never repacks the trellis. Under CUDA unified addressing a
pinned host allocation is dereferenceable from the device, so the slab-alias invariant
(`tensor.data_ptr() == backing[expert].data_ptr()`) holds with the slab on the host.
Verified with a standalone `cudaHostRegister` + `cudaHostGetDevicePointer` test (identical
addresses) and by CUDA-graph capture succeeding.

## Working configuration

```
SLAB_GIB=58 PREFILL_BLOCK_M=64 KV_DTYPE=nvfp4_ds_mla NOPE_NVFP4=1 MAXLEN=262144 UTIL=0.95 SHM=8g \
LOAD_ARGS="--max-parallel-loading-workers 1 --safetensors-prefetch-num-threads 1" ./scripts/serve_vllm_tp1.sh
```
Result: 120/120 shards, 202 GPU slabs + 50 host-registered slabs, GPU 92.0 GiB, host
107 GB used / 17 GB free, KV cache 2.64 M tokens, `Application startup complete`.
`EXL3_GPU_SLAB_GIB` is the GPU budget for expert slabs; everything above it is placed in
pinned host memory. Read it inside the container (`docker exec … env`) — a lost `-e` cost a run.

## Limits found

- **fp8 KV is not available for this model on `B12X_MLA_SPARSE`**: the 512-head NoPE path
  (`qk_rope_head_dim=0`) has exactly one KV record format, `nvfp4_ds_mla`. `fp8_ds_mla` is a
  different packed layout for the 576-head geometry. `FLASHMLA_SPARSE`/`FLASHINFER_MLA_SPARSE`
  accept fp8, but the DSA indexer (`B12xNonCompressedIndexerBackend`, `KpoolTailBackend`) is
  B12X-specific, so switching backends likely forfeits sparse attention — unverified.
- Throughput is PCIe-bound (~4 GB of expert reads per token at top-8 across 42 layers).

## Guard rails (these made it safe to iterate)

- **No swap.** Swap turned a fast, clean OOM-kill into a machine-wide hang that needed a
  Proxmox hard reset. Without swap the kernel kills the process in seconds and the box stays up.
- `scripts/memguard.sh` kills the container at 8 GB host-available and logs `/proc/meminfo`
  composition (Shmem/Mlocked/AnonPages/Cached) — that trace is what exposed the power-of-two
  rounding.
- Verify env *inside* the container before trusting a run.

## Provenance — what these patches actually apply to

The two patched files ship **inside a pinned container image**, not in any checkout on this
host. Both are git working trees within that image, and the two files are **not in the same
state** — which decides how each patch can be reused:

- **`prepare.py` is byte-identical to upstream** at `36bce2c`. Its patch applies cleanly to a
  fresh clone of `local-inference-lab/b12x`.
- **`exl3.py` is heavily vendor-modified**: upstream `6dc2f5166` has it at 98,815 bytes, the
  image ships **212,888 bytes** (2.15x). Its patch assumes that vendor state and will **not**
  apply to a clean upstream clone.

```
image     verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-language-only
digest    sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692
built     2026-08-28 · vLLM 0.1.dev20111+g7f1e92bec.d20260827
```

| tree in image | upstream | commit | patched file vs upstream |
|---|---|---|---|
| `/opt/infernal-invocation/b12x` | `local-inference-lab/b12x` | `36bce2c` (master) | **identical** (92,215 B) |
| `/opt/infernal-invocation/vllm` | `local-inference-lab/vllm` | `6dc2f5166` | **vendor-modified**, 98,815 -> 212,888 B |

Both patches are also published as fork branches, so they can be read as ordinary commits:

| fork | branch | base |
|---|---|---|
| [`anoane/b12x`](https://github.com/anoane/b12x/tree/anoane) | `anoane` | `36bce2c` — one commit, applies cleanly to upstream |
| [`anoane/vllm`](https://github.com/anoane/vllm/tree/anoane) | `anoane` | `6dc2f5166` + one commit recording the image's vendor state, **then** my delta as a second commit so the change is reviewable in isolation |

Verify you have the right bytes before patching — these are the pre-patch files:

```
b7636111082af0244243681ae770e53f7954a3e16433c9e844e39a4199f38e69  b12x/moe/_shared/kernels/w4a16/prepare.py
ae92592ea8fcd249978134357ea3cd2510fe2aa9bdb1d1a3ab02afdbaeb39f45  vllm/model_executor/layers/quantization/exl3.py
```

`scripts/serve_vllm_tp1.sh` bind-mounts the patched copies over the image at those paths, so
the image is never rebuilt and stays byte-identical to the digest above.

## Files

- `patches/exl3_tp1_host_slabs.patch` — against
  `vllm/model_executor/layers/quantization/exl3.py` (walls 1,3,4,5-caller,7 + host-slab placement + diagnostics)
- `patches/b12x_prepare_compute_device.patch` — against
  `b12x/moe/_shared/kernels/w4a16/prepare.py` (walls 5,6)
- `scripts/serve_vllm_tp1.sh` — mounts both patched files over the pinned image; all knobs env-driven
- `scripts/memguard.sh` — host-memory watchdog
- `results/` — smoke generation, memguard traces, run summaries

Image: see **Provenance** above (pinned by digest).
Checkpoint: `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (163.56 GiB; 145.55 GiB routed experts).
