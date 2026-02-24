# SageAttention Performance Optimization Strategies vs Baseline FlashAttention

This document summarizes the main strategies used by `SageAttention` (`/workspace/SageAttention`) to improve performance relative to baseline FlashAttention kernels, with emphasis on what is algorithmic (numeric/data representation) vs what is kernel engineering (fusion, architecture specialization, launch tuning).

It also highlights the SageAttention3 “Blackwell” path (implemented under `sageattention3_blackwell`) as a reference for low-precision + correction strategies, even though its CUDA implementation is currently gated to SM120/SM121 in this repo snapshot (`/workspace/SageAttention/sageattention3_blackwell/sageattn3/blackwell/api.cu:222`).

---

## Executive Summary

SageAttention speedups are not just from a faster attention kernel schedule. The main gains come from a combination of:

1. **Low-precision attention math inputs**
   - `Q/K` quantized to **INT8**
   - `V` quantized to **FP8** (SageAttention2) or **FP4 + FP8 scales** (SageAttention3)
2. **Smoothing / correction terms**
   - Mean subtraction on `K` (and optionally `Q` / `V`) to make quantization accurate
   - LSE / score correction terms to recover accuracy
3. **Kernel fusion**
   - Fuse mean/scale/quant/layout transforms with attention kernels
4. **Architecture-specific kernels**
   - Separate implementations for SM80 / SM89 / SM90 and Blackwell (SageAttention3)
5. **Two-level accumulation**
   - Hybrid accumulation (e.g., `fp16+fp32`, `fp32+fp32`) to improve accuracy without giving up too much throughput

Relative to baseline FlashAttention, the distinguishing advantage is primarily the **quantized dataflow + fused correction path**, not only tile scheduling.

---

## SageAttention Implementations and Dispatch Structure

### Public API variants (SageAttention2)

`/workspace/SageAttention/README.md:111`

- `sageattn` (auto-dispatch)
- `sageattn_qk_int8_pv_fp16_triton`
- `sageattn_qk_int8_pv_fp16_cuda`
- `sageattn_qk_int8_pv_fp8_cuda`
- `sageattn_qk_int8_pv_fp8_cuda_sm90` (Hopper-specific)
- `sageattn_varlen`

### Auto-dispatch by GPU architecture

`/workspace/SageAttention/sageattention/core.py:79`

`sageattn()` chooses implementations based on arch:

- `sm80` -> INT8 QK + FP16 PV CUDA path (`pv_accum_dtype="fp32"`)
- `sm86` -> Triton FP16-PV path
- `sm89` -> INT8 QK + FP8 PV CUDA path (`fp32+fp16`)
- `sm90` -> Hopper-specific INT8 QK + FP8 PV CUDA path (`fp32+fp32`)
- `sm120/sm121` -> currently routes to SM89-style FP8 path with `per_warp` quant granularity

**Important practical note for B200 / SM100:** the auto-dispatch derives architecture strings like `sm{major}{minor}` via `get_cuda_arch_versions()` (`/workspace/SageAttention/sageattention/core.py:71`) and raises on unhandled architectures (`/workspace/SageAttention/sageattention/core.py:156`). There is no `sm100` branch in `sageattn()` in this repo snapshot, so on B200 you generally need to call a concrete entrypoint (e.g. `sageattn_qk_int8_pv_fp8_cuda(...)`) rather than relying on `sageattn(...)`.

### Build-time split into architecture-specific extensions

`/workspace/SageAttention/setup.py:174`

SageAttention builds:

- `sageattention._qattn_sm80`
- `sageattention._qattn_sm89`
- `sageattention._qattn_sm90`
- `sageattention._fused` (pre/post-process kernels: quant, mean subtraction, layout transforms)

Notable kernel families compiled into these extensions include:

- SM80: INT8-QK + FP16-PV kernels with `accum_f16`, `accum_f32`, `inst_buf`, and `fuse_v_mean` variants (`/workspace/SageAttention/csrc/qattn/attn_cuda_sm80.h:19`).
- SM89: INT8-QK + FP8-PV kernels with `fuse_v_scale`, `fuse_v_mean`, `accum_f16`/`accum_f32`, and `inst_buf` variants (`/workspace/SageAttention/csrc/qattn/attn_cuda_sm89.h:19`).
- SM90: Hopper INT8-QK + FP8-PV kernel family with `inst_buf` and `fuse_v_scale` variants (`/workspace/SageAttention/csrc/qattn/attn_cuda_sm90.h:19`).

The setup also supports newer architectures including `sm_100a`, `sm_120a`, `sm_121a` in gencode generation:

- `/workspace/SageAttention/setup.py:156`
- `/workspace/SageAttention/setup.py:167`

---

## Core Optimization Strategies in SageAttention2

## 1) INT8 Quantization for `QK^T`

### What Sage does

SageAttention quantizes `Q` and `K` to INT8 before entering the attention kernel.

Implemented in:

- `/workspace/SageAttention/sageattention/quant.py:22` (`per_block_int8`)
- `/workspace/SageAttention/sageattention/quant.py:105` (`per_warp_int8`)
- Triton variants:
  - `/workspace/SageAttention/sageattention/triton/quant_per_block.py:49`
  - `/workspace/SageAttention/sageattention/triton/quant_per_thread.py:154`
  - `/workspace/SageAttention/sageattention/triton/quant_per_block_varlen.py:60`

### Why it helps

- Reduces memory bandwidth for `Q` and `K`
- Enables low-precision tensor-core/WGMMA-friendly matmul paths for score computation
- Reduces cache/shared-memory footprint for score tiles

### How it is implemented

In `per_block_int8` (`quant.py`):

- Allocates `q_int8`, `k_int8`
- Computes scale tensors `q_scale`, `k_scale`
- Calls fused CUDA kernels:
  - `_fused.quant_per_block_int8_cuda(...)`
  - `_fused.quant_per_block_int8_fuse_sub_mean_cuda(...)` when `km` is provided

Relevant code:

- `/workspace/SageAttention/sageattention/quant.py:72`
- `/workspace/SageAttention/sageattention/quant.py:88`
- `/workspace/SageAttention/sageattention/quant.py:96`
- `/workspace/SageAttention/sageattention/quant.py:99`

At the CUDA-kernel level, the fused INT8 quantizer (`QuantInt8Kernel`) computes an amax per block, writes `scale = amax / 127`, and packs/stores INT8 outputs with vectorized stores. See `/workspace/SageAttention/csrc/fused/fused.cu:64` and the amax/scale/store logic at `/workspace/SageAttention/csrc/fused/fused.cu:147`.

### INT4 status (partially implemented)

SageAttention includes **INT4 quantization kernels and plumbing**, but it is not wired into the main public `sageattn_*` APIs in this repo snapshot:

- Triton per-thread int4 quant kernels exist: `/workspace/SageAttention/sageattention/triton/quant_per_thread.py:101`.
- Core CUDA attention helpers and enums support `DataType::kInt4`: `/workspace/SageAttention/csrc/qattn/attn_utils.cuh:41` and `/workspace/SageAttention/csrc/qattn/attn_utils.cuh:175`.
- The SM89 FP8 attention kernel is templated to allow `DTypeQK == int8 or int4`: `/workspace/SageAttention/csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh:56`.

---

## 2) Smoothing (Mean Subtraction) to Improve Quantization Accuracy

### What Sage does

SageAttention subtracts sequence means (especially on `K`) before quantization:

- `smooth_k=True` by default in major paths
- optional `smooth_v`

This reduces outliers / bias and improves quantization fidelity.

### Where it happens

Examples in `core.py`:

- Triton FP16 path: `/workspace/SageAttention/sageattention/core.py:279`
- SM80 CUDA path: `/workspace/SageAttention/sageattention/core.py:583`
- SM89 FP8 path: `/workspace/SageAttention/sageattention/core.py:772`
- SM90 Hopper path: `/workspace/SageAttention/sageattention/core.py:948`

### LSE correction / score correction

When `return_lse=True`, Sage computes a correction term from the removed mean and adds it back to the returned LSE:

- `/workspace/SageAttention/sageattention/core.py:291`
- `/workspace/SageAttention/sageattention/core.py:329`
- `/workspace/SageAttention/sageattention/core.py:595`
- `/workspace/SageAttention/sageattention/core.py:631`
- `/workspace/SageAttention/sageattention/core.py:784`
- `/workspace/SageAttention/sageattention/core.py:824`
- `/workspace/SageAttention/sageattention/core.py:960`
- `/workspace/SageAttention/sageattention/core.py:994`

### Why it helps vs FlashAttention baseline

Baseline FlashAttention generally keeps exact FP16/BF16 values in the core attention pipeline. Sage intentionally introduces quantization error, then compensates with smoothing/corrections so the low-precision path remains accurate enough.

### Varlen caveat

In the varlen API, `smooth_k` currently computes a mean over the full packed `k` tensor (across all sequences) and notes that computing per-sequence means would require a dedicated kernel:

- `/workspace/SageAttention/sageattention/core.py:432`

---

## 3) Quantization Granularity Tuning (`per_block`, `per_warp`, `per_thread`)

### What Sage does

SageAttention exposes multiple Q/K quantization granularities:

- `per_warp`
- `per_thread`

In `core.py` these are handled via `qk_quant_gran`:

- `/workspace/SageAttention/sageattention/core.py:457`
- `/workspace/SageAttention/sageattention/core.py:642`
- `/workspace/SageAttention/sageattention/core.py:835`

### Why it helps

This is a direct throughput vs accuracy tradeoff:

- Finer granularity -> better quantization fidelity, more scale metadata + overhead
- Coarser granularity -> lower overhead, potentially worse accuracy

Sage tunes this per architecture and kernel mode.

### Evidence in kernel launchers

SM89 launchers validate different scale tensor shapes depending on quant granularity:

- `/workspace/SageAttention/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu:131`
- `/workspace/SageAttention/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu:136`

Dispatch is quant-granularity-aware:

- `/workspace/SageAttention/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu:116`

---

## 4) FP8 Quantization for `V` (PV path) + Layout Transform for Kernel-Friendly Access

### What Sage does

SageAttention2 FP8 variants quantize `V` using per-channel FP8 quantization and prepare it in a kernel-friendly layout.

Implemented in:

- `/workspace/SageAttention/sageattention/quant.py:224` (`per_channel_fp8`)

This function:

1. Transposes `V`
2. Pads sequence length to a multiple of 64
3. Permutes sequence positions into a specific order
4. Quantizes to `torch.float8_e4m3fn`
5. Produces per-channel scales (and optional means)

Key code points:

- `/workspace/SageAttention/sageattention/quant.py:231`
- `/workspace/SageAttention/sageattention/quant.py:281`
- `/workspace/SageAttention/sageattention/quant.py:283`
- `/workspace/SageAttention/sageattention/quant.py:289`
- `/workspace/SageAttention/sageattention/quant.py:292`

### Why it helps

This is both:

- **Compression** (FP8 `V` instead of FP16/BF16)
- **Layout optimization** (pre-shaping data for efficient reads in the PV kernel)

Compared to baseline FlashAttention, Sage is explicitly restructuring `V` before the kernel to align with its custom low-precision pipeline.

### Kernel-side padding assumptions (SM89 and SM90)

The SM89 FP8 attention kernel contains an explicit assumption that `V` is padded, warning about illegal memory access / NaNs otherwise:

- `/workspace/SageAttention/csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh:259`

In the SM90 path, the Python wrapper pads `V` to a multiple of 128 before FP8 quantization (currently a workaround for `per_channel_fp8` limitations):

- `/workspace/SageAttention/sageattention/core.py:973`

---

## 5) Two-Level Accumulation (`inst_buf`) for Accuracy/Speed Tradeoff

### What Sage does

SageAttention uses hybrid accumulation schemes:

- `fp16+fp32`
- `fp32+fp32`
- `fp32+fp16` (SageAttention2++)

Documented in:

- `/workspace/SageAttention/sageattention/core.py:500`
- `/workspace/SageAttention/sageattention/core.py:685`
- `/workspace/SageAttention/bench/README.md:19`

The `+` modes correspond to **two-level accumulation**, where lower-precision accumulation is periodically merged into a higher-precision buffer.

### Where it shows up in kernel choices

Sage selects `*_inst_buf` kernels for these modes:

- SM80 FP16 path:
  - `/workspace/SageAttention/sageattention/core.py:624`
- SM89 FP8 path:
  - `/workspace/SageAttention/sageattention/core.py:817`
  - `/workspace/SageAttention/sageattention/core.py:819`
- SM90 FP8 Hopper path:
  - `/workspace/SageAttention/sageattention/core.py:989`

### Why it helps vs baseline FlashAttention

FlashAttention typically uses a fixed accumulator strategy for a given kernel. Sage exposes a wider set of tuned accumulation behaviors to better exploit low-precision tensor/WGMMA math while controlling numerical drift.

### PV scale tuning for `fp32+fp16`

In the FP8 path, Sage adjusts the `V` quantization scale range when using `pv_accum_dtype == 'fp32+fp16'` (SageAttention2++), setting `scale_max` to `2.25` instead of the FP8 E4M3 upper bound (`448.0`):

- `/workspace/SageAttention/sageattention/core.py:805`
- `/workspace/SageAttention/sageattention/core.py:807`

---

## 6) Fusing Quantization / Mean / Scale / Attention Operations

### What Sage does

SageAttention aggressively fuses preprocessing and attention-adjacent operations:

Fused preprocessing kernels (`_fused`):

- `quant_per_block_int8_fuse_sub_mean_cuda`
- `transpose_pad_permute_cuda`
- `mean_scale_fuse_quant_cuda`
- `scale_fuse_quant_cuda`

Used from:

- `/workspace/SageAttention/sageattention/quant.py:99`
- `/workspace/SageAttention/sageattention/quant.py:176`
- `/workspace/SageAttention/sageattention/quant.py:281`
- `/workspace/SageAttention/sageattention/quant.py:289`
- `/workspace/SageAttention/sageattention/quant.py:292`

Attention kernel variants explicitly fuse value-scale and value-mean handling:

- SM89 custom ops in `/workspace/SageAttention/sageattention/sm89_compile.py:5`
- `...fuse_v_scale_attn_inst_buf` in `/workspace/SageAttention/sageattention/sm89_compile.py:27`
- `...fuse_v_scale_fuse_v_mean_attn` in `/workspace/SageAttention/sageattention/sm89_compile.py:104`
- SM90 fused-v-scale custom op in `/workspace/SageAttention/sageattention/sm90_compile.py:55`

### Why it helps

Quantization/smoothing overhead can erase gains if done as separate kernels. Sage preserves net speedup by fusing as much preprocessing as possible into fewer launches / more cache-friendly stages.

### Softmax base-2 scaling and “fold dequant into `sm_scale`”

Several Sage kernels convert softmax scaling into base-2 exponentiation (`exp2`) by multiplying `sm_scale` by `log2(e)` inside the kernel:

- SM89: `/workspace/SageAttention/csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh:90`
- SM90: `/workspace/SageAttention/csrc/qattn/qk_int_sv_f8_cuda_sm90.cu:153`

For quantized `Q/K`, the SM89 kernel also folds per-tile dequantization factors into `sm_scale` (using `q_scale * k_scale[...]`), keeping the inner score pipeline in an “already-scaled” domain:

- `/workspace/SageAttention/csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh:252`
- `/workspace/SageAttention/csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh:258`

On the Python side, Sage multiplies the softmax scale by `1.44269504` (log2(e)) in the quantization helper to match the Triton attention kernel convention:

- `/workspace/SageAttention/sageattention/quant.py:94`

---

## 7) Architecture-Specific Kernel Tuning and Launch Configurations

### What Sage does

Sage uses different tiles and kernel variants per architecture.

#### Example: SM89 FP8 kernel launcher

`/workspace/SageAttention/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu`

Launch specialization includes:

- `CTA_Q = 128`
- `CTA_K = 64`
- `WARP_Q = 32`
- `WARP_K = 64`

Relevant lines:

- `/workspace/SageAttention/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu:119`
- `/workspace/SageAttention/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu:120`
- `/workspace/SageAttention/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu:121`
- `/workspace/SageAttention/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu:122`

Kernel launch setup:

- Dynamic shared memory sizing (`cudaFuncSetAttribute`)
- Tuned grid and block dimensions

Relevant lines:

- `/workspace/SageAttention/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu:145`
- `/workspace/SageAttention/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu:150`
- `/workspace/SageAttention/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu:152`
- `/workspace/SageAttention/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu:153`

#### Example: SM80 FP16 kernel launcher

`/workspace/SageAttention/csrc/qattn/qk_int_sv_f16_cuda_sm80.cu`

Multiple launch variants (causal/non-causal, accumulator variants) use:

- `CTA_Q = 128`
- `CTA_K = 64`
- `WARP_Q = 32` (or `16` for some head-dim-specific paths)
- `WARP_K = 64`

Relevant lines:

- `/workspace/SageAttention/csrc/qattn/qk_int_sv_f16_cuda_sm80.cu:789`
- `/workspace/SageAttention/csrc/qattn/qk_int_sv_f16_cuda_sm80.cu:790`
- `/workspace/SageAttention/csrc/qattn/qk_int_sv_f16_cuda_sm80.cu:791`
- `/workspace/SageAttention/csrc/qattn/qk_int_sv_f16_cuda_sm80.cu:792`
- `/workspace/SageAttention/csrc/qattn/qk_int_sv_f16_cuda_sm80.cu:1141`

### Why it helps vs baseline FlashAttention

Sage is not using one generic low-precision kernel. It maintains multiple highly specialized kernels and launch configurations to match precision mode + architecture constraints.

---

## 8) Hopper-Specific (SM90) Implementation with WGMMA/TMA + Sage Dataflow

### What Sage does

SageAttention provides a dedicated SM90 kernel path:

- Python API: `/workspace/SageAttention/sageattention/core.py:829`
- CUDA kernel: `/workspace/SageAttention/csrc/qattn/qk_int_sv_f8_cuda_sm90.cu`

This path uses Hopper-specific primitives:

- `wgmma` (warpgroup MMA)
- TMA / `cp.async.bulk.tensor`

Evidence:

- `/workspace/SageAttention/csrc/qattn/qk_int_sv_f8_cuda_sm90.cu:23`
- `/workspace/SageAttention/csrc/qattn/qk_int_sv_f8_cuda_sm90.cu:75`
- `/workspace/SageAttention/csrc/qattn/qk_int_sv_f8_cuda_sm90.cu:275`
- `/workspace/SageAttention/csrc/qattn/qk_int_sv_f8_cuda_sm90.cu:333`

### SM90-specific quantization / tiling differences in Python path

Sage uses different quantization tile settings on SM90:

- `BLKQ=64`
- `WARPQ=16`
- `BLKK=128`

Relevant lines:

- `/workspace/SageAttention/sageattention/core.py:966`
- `/workspace/SageAttention/sageattention/core.py:967`
- `/workspace/SageAttention/sageattention/core.py:968`
- `/workspace/SageAttention/sageattention/core.py:969`

It also pads `V` to a multiple of 128 before FP8 quantization:

- `/workspace/SageAttention/sageattention/core.py:973`
- `/workspace/SageAttention/sageattention/core.py:983`

### Why it matters

Compared to FA3/Hopper-style kernels, Sage’s key differentiator here is not the use of WGMMA/TMA alone (both can use Hopper primitives), but **how Sage feeds them**:

- INT8 `Q/K`
- FP8 `V`
- fused scale paths
- two-level accumulation (`inst_buf`)

---

## 9) Varlen Support to Reduce Padding Waste

### What Sage does

SageAttention provides a varlen API:

- `/workspace/SageAttention/sageattention/core.py:334`
- API listed in `/workspace/SageAttention/README.md:117`

It quantizes varlen Q/K using specialized Triton kernels:

- `/workspace/SageAttention/sageattention/core.py:439`
- `/workspace/SageAttention/sageattention/triton/quant_per_block_varlen.py:60`

### Why it helps

This is a performance strategy at the batch/system level: less wasted compute on padded tokens, especially for heterogeneous sequence lengths.

---

## 10) `torch.compile` / Graph-Friendly Integration

### What Sage does

Sage registers custom ops and fake implementations (`torch.library.custom_op`, `register_fake`) for compile-friendly graph integration:

- SM89 wrappers: `/workspace/SageAttention/sageattention/sm89_compile.py:5`, `/workspace/SageAttention/sageattention/sm89_compile.py:99`
- SM90 wrappers: `/workspace/SageAttention/sageattention/sm90_compile.py:5`, `/workspace/SageAttention/sageattention/sm90_compile.py:25`

README also explicitly calls out `torch.compile` support:

- `/workspace/SageAttention/README.md:30`

Several SageAttention entrypoints also include a runtime workaround `torch.cuda.set_device(v.device)` to avoid stability issues in distributed inference and to improve compatibility with `torch.compile` in non-fullgraph modes:

- `/workspace/SageAttention/sageattention/core.py:252`
- `/workspace/SageAttention/sageattention/core.py:548`

Finally, their examples show “plug-and-play” replacement of SDPA (`F.scaled_dot_product_attention = sageattn`) but note that model-specific masking behaviors may require selective replacement (e.g., keep masked/text attention on SDPA/FA, apply SageAttention to large mask-free self-attention):

- `/workspace/SageAttention/example/README.md:14`
- `/workspace/SageAttention/example/README.md:41`

### Why it helps

Kernel speedups can be diluted by framework overhead. This integration helps preserve gains in real inference pipelines.

---

## SageAttention3 (Blackwell) Optimizations Relevant to SM100/FA4 Comparison

SageAttention3 is a separate Blackwell-focused implementation under:

- `/workspace/SageAttention/sageattention3_blackwell/`

README:

- `/workspace/SageAttention/sageattention3_blackwell/README.md:5`

**Compatibility note:** the SageAttention3 CUDA extension in this repo snapshot checks for SM120/SM121 and errors otherwise:

- `/workspace/SageAttention/sageattention3_blackwell/sageattn3/blackwell/api.cu:222`

## 11) FP4 + FP8-Scale Microscaling (Blackwell)

### What SageAttention3 does

SageAttention3 preprocesses and quantizes to packed FP4 with FP8 scales:

- `scale_and_quant_fp4`
- `scale_and_quant_fp4_permute`
- `scale_and_quant_fp4_transpose`

Implemented in:

- `/workspace/SageAttention/sageattention3_blackwell/sageattn3/api.py:94`
- `/workspace/SageAttention/sageattention3_blackwell/sageattn3/api.py:102`
- `/workspace/SageAttention/sageattention3_blackwell/sageattn3/api.py:110`

### Preprocess and correction terms

In `preprocess_qkv(...)`, SageAttention3:

- subtracts mean from `K`
- pads `Q/K/V` to multiples of 128
- optionally computes per-block/group means for `Q`
- computes `delta_s = qm @ k^T` correction term

Implemented in:

- `/workspace/SageAttention/sageattention3_blackwell/sageattn3/api.py:75`
- `/workspace/SageAttention/sageattention3_blackwell/sageattn3/api.py:84`
- `/workspace/SageAttention/sageattention3_blackwell/sageattn3/api.py:87`
- `/workspace/SageAttention/sageattention3_blackwell/sageattn3/api.py:91`

### Why this matters for FA4/CuTe SM100

This is the most direct Sage-style optimization direction for SM100:

- microscaled low-precision Q/K/V representations
- explicit correction terms (`delta_s`)
- Blackwell-native kernels for the transformed dataflow

---

## 12) Fused Online Softmax + Probability Quantization (SageAttention3)

### What SageAttention3 does

SageAttention3’s Blackwell kernel fuses online softmax with quantization of intermediate probabilities / scales.

Key component:

- `SoftmaxFused::online_softmax_with_quant(...)`

Implemented in:

- `/workspace/SageAttention/sageattention3_blackwell/sageattn3/blackwell/softmax_fused.h:40`

This object maintains:

- row max
- row sum
- rescale factors

and quantizes data during the online softmax progression.

### Mainloop integration (Blackwell kernel)

The Blackwell mainloop:

- runs QK GEMM
- applies fused online softmax + quant
- quantizes/interleaves score fragments
- immediately feeds PV GEMMs
- rescales outputs incrementally

See:

- `/workspace/SageAttention/sageattention3_blackwell/sageattn3/blackwell/mainloop_tma_ws.h:750`
- `/workspace/SageAttention/sageattention3_blackwell/sageattn3/blackwell/mainloop_tma_ws.h:799`
- `/workspace/SageAttention/sageattention3_blackwell/sageattn3/blackwell/mainloop_tma_ws.h:806`
- `/workspace/SageAttention/sageattention3_blackwell/sageattn3/blackwell/mainloop_tma_ws.h:845`
- `/workspace/SageAttention/sageattention3_blackwell/sageattn3/blackwell/mainloop_tma_ws.h:861`
- `/workspace/SageAttention/sageattention3_blackwell/sageattn3/blackwell/mainloop_tma_ws.h:899`
- `/workspace/SageAttention/sageattention3_blackwell/sageattn3/blackwell/mainloop_tma_ws.h:901`

### Why it helps

This reduces intermediate bandwidth and kernel boundaries while preserving the online-softmax structure required for attention stability.

Concretely, SageAttention3 quantizes per-tile probability fragments into a packed FP4-like format (E2M1) and emits per-tile scale factors (UE4M3) during the mainloop:

- Probability quantization / scale packing: `/workspace/SageAttention/sageattention3_blackwell/sageattn3/blackwell/mainloop_tma_ws.h:750`
- Helpers that pack floats into UE4M3 / E2M1: `/workspace/SageAttention/sageattention3_blackwell/sageattn3/blackwell/utils.h:200`

---

## 13) Persistent Scheduling and Cluster Launch (Blackwell)

SageAttention3 Blackwell uses a persistent tile scheduler and clustered launch:

- `StaticPersistentTileScheduler` in `/workspace/SageAttention/sageattention3_blackwell/sageattn3/blackwell/launch.h:42`
- Grid/block/cluster launch in `/workspace/SageAttention/sageattention3_blackwell/sageattn3/blackwell/launch.h:89`
- `cutlass::launch_kernel_on_cluster(...)` in `/workspace/SageAttention/sageattention3_blackwell/sageattn3/blackwell/launch.h:93`

This is kernel engineering optimization on top of the low-precision Sage dataflow.

---

## Additional Kernel Engineering Building Blocks (SageAttention2)

SageAttention2’s CUDA kernels reuse many “high-end FA-class” building blocks, adapted for INT8/FP8 dataflows:

- **Permuted/swirled shared memory addressing** to improve bank utilization:
  - `/workspace/SageAttention/csrc/permuted_smem.cuh:39`
  - `/workspace/SageAttention/csrc/permuted_smem.cuh:86`
- **`cp.async`-based global→shared pipelining** with optional L2 prefetch:
  - `/workspace/SageAttention/csrc/cp_async.cuh:42`
  - `/workspace/SageAttention/csrc/cp_async.cuh:71`
- **Hopper SM90 TMA + mbarrier pipeline** (`cp.async.bulk.tensor.*`) plus WGMMA:
  - Tensor map encode: `/workspace/SageAttention/csrc/qattn/qk_int_sv_f8_cuda_sm90.cu:29`
  - Bulk tensor async load: `/workspace/SageAttention/csrc/qattn/qk_int_sv_f8_cuda_sm90.cu:75`
  - mbarrier init: `/workspace/SageAttention/csrc/qattn/qk_int_sv_f8_cuda_sm90.cu:56`

These overlap with FlashAttention’s general kernel engineering playbook; Sage’s main differentiation is that they are deployed in service of a different numeric formulation (quantization + smoothing + two-level accumulation).

## Comparison to Baseline FlashAttention (What Actually Changes)

## A) What is similar

SageAttention kernels still use the general FlashAttention-style principles:

- tiled score computation
- online softmax accumulation
- streaming over K/V tiles
- architecture-specific kernels (especially on Hopper/Blackwell)

## B) What Sage adds (the main optimization delta)

1. **Quantized representations in the main path**
   - INT8 `Q/K`
   - FP8 `V` (Sage2) / FP4+scales QKV (Sage3)
2. **Smoothing / correction math**
   - mean subtraction and correction terms for accuracy preservation
3. **Fused pre/post transforms**
   - quant + mean + scale + layout transforms integrated with attention kernels
4. **Two-level accumulation**
   - explicit hybrid accuracy/perf modes

In short: SageAttention changes the **numeric/dataflow formulation** first, then optimizes kernels around that formulation.

---

## Important Benchmarking Caveat

SageAttention’s benchmark docs note that published kernel TOPS usually exclude quantization and smoothing overhead:

- `/workspace/SageAttention/README.md:174`
- `/workspace/SageAttention/bench/README.md:63`

This means:

- Kernel-only gains may look larger than end-to-end gains
- Net speedup depends on how effectively preprocessing is fused/reused in the actual model pipeline

Also, Sage’s `bench_baseline.py` toggles which SDPA backend is used (`fa2`, `torch`, `xformers`) via PyTorch SDPA backend flags:

- `/workspace/SageAttention/bench/bench_baseline.py:21`

---

## Practical Implications for SM100 / FA4 Experiments in This Repo

If the goal is to make an FA4/CuTe SM100 kernel behave more like SageAttention, the highest-impact changes are typically:

1. **Add preprocessing/correction path**
   - `K` smoothing (mean subtraction)
   - optional `Q` group mean + `delta_s` correction
2. **Introduce quantized data representations**
   - start with `V`-side quantization + scales, then `Q/K`
3. **Fuse quant + scale handling into the SM100 forward kernel path**
4. **Add hybrid accumulation modes**
5. **Benchmark kernel-only and end-to-end separately**

For Blackwell specifically, SageAttention3 is the strongest reference, because it already implements a Blackwell-native low-precision attention pipeline.
