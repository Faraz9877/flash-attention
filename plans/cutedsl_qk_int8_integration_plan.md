# CuTeDSL SM100 Q/K INT8 Integration Plan (FlashAttention FA4)

This document is a concrete implementation plan for converting the SM100 CuTeDSL forward attention kernel path in `/workspace/flash-attention` from the current FP16/BF16 `Q/K/V` input flow to a **Q/K INT8-quantized** path (SageAttention-inspired), while keeping the first iteration low-risk:

- `Q/K`: INT8 + scales
- `V`: keep FP16/BF16 initially
- `PV`: keep existing FA4 path initially
- target kernel file: `flash_attn/cute/flash_fwd_sm100_qk_int8.py`

The goal is to reduce bandwidth and improve compute efficiency on the QK stage, while preserving the existing FA4/CuTe SM100 scheduling and softmax/PV pipeline as much as possible in phase 1.

---

## Scope and Non-Goals (Phase 1)

### In scope

- Add a CuTeDSL SM100 forward kernel variant that accepts **pre-quantized INT8 `Q/K`**.
- Use **per-block/per-warp scale tensors** (`q_scale`, `k_scale`) to dequantize scores logically (without materializing dequantized `Q/K`).
- Switch **QK MMA** to INT8 input / INT32 accumulation.
- Convert QK accumulators to FP32 for softmax and reuse existing softmax + PV flow.
- Reuse SageAttention quantization kernels as a preprocessing stage (already copied under `csrc/sageattention_quant/`).

### Out of scope (Phase 1)

- In-kernel quantization of Q/K inside the CuTe mainloop.
- V FP8 / FP4 quantization.
- Probability quantization / fused quantized softmax.
- Full Sage-style smoothing/correction/LSE compensation (optional future work).
- Backward kernel support.
- Full upstream `flash_attn.cute.interface` API integration (can start with experimental wrappers).

---

## Current Constraints in FA4/CuTe SM100 (What blocks Q/K INT8 today)

### 1) Frontend forces FP16/BF16 and same dtype for `q/k/v`

In `flash_attn/cute/interface.py`:

- `assert q.dtype in [torch.float16, torch.bfloat16]`
- `assert q.dtype == k.dtype == v.dtype`

This blocks:

- `q/k` INT8 inputs
- `v` staying FP16/BF16 while `q/k` are INT8
- passing scale tensors as first-class quantization metadata

### 2) SM100 forward kernel assumes shared dtype domains

In `flash_attn/cute/flash_fwd_sm100.py`, the kernel class stores:

- `self.q_dtype`, `self.k_dtype`, `self.v_dtype`, `self.o_dtype`

but then enforces:

- `self.q_dtype == self.k_dtype`
- `self.q_dtype == self.v_dtype`

This assumption propagates into:

- QK MMA operand dtype selection
- P/TMEM intermediate storage layouts
- softmax conversion path (recasts using `self.q_dtype`)
- staging heuristics based on `self.q_dtype.width`

### 3) P (probability/intermediate) layout uses Q dtype as a proxy

Several places implicitly assume the datatype used to store `P`-like intermediates is the same as `Q`:

- `tP_layout` creation
- `tSrP_r2t` recast in softmax
- `tilePlikeFP32` width transformations tied to `self.v_dtype` and QK layout expectations

With `Q/K = INT8`, this must be decoupled from QK input dtype.

### 4) Softmax path currently assumes QK accumulator flow is already the expected floating domain

The softmax stage loads QK accumulators and applies:

- mask / score_mod
- row max
- exp2 conversion

With QK INT8, the kernel must account for **dequantization scale** (`q_scale * k_scale`) before softmax normalization logic (or fold it into `softmax_scale`).

---

## High-Level Design Choice (Recommended)

Use a **hybrid Sage-inspired design**:

1. **Preprocess outside CuTe kernel**
   - Quantize `Q/K` to INT8 using SageAttention fused kernels
   - Produce `q_scale`, `k_scale`
   - Optional K smoothing later

2. **Keep FA4 SM100 kernel structure**
   - Reuse tile scheduling, TMA/UMMA pipeline, masking, softmax, PV, epilogue

3. **Change only QK representation and score scaling**
   - INT8 QK MMA -> INT32 accumulators
   - fold per-tile dequantization into score scaling (`softmax_scale_log2`)

This minimizes code churn while targeting the highest-impact bandwidth savings first.

---

## Required Changes by Area

## A. Python Frontend / Interface Changes (`flash_attn/cute/interface.py`)

### A1) Add an experimental INT8-QK entrypoint or mode

Do **not** overload the default entrypoint first. Add one of:

- a new function (e.g. `flash_attn_func_qk_int8(...)`)
- or an experimental flag path that dispatches to `flash_fwd_sm100_qk_int8.py`

This avoids destabilizing the default FA4 path.

### A2) Relax dtype validation rules for the experimental path

Current checks require:

- `q/k/v` all FP16/BF16 and same dtype

INT8 path needs:

- `q.dtype == k.dtype == torch.int8`
- `v.dtype in {torch.float16, torch.bfloat16}`
- `out.dtype` typically `v.dtype` or explicit output dtype

### A3) Add first-class quantization metadata inputs

Required additional inputs:

- `q_scale`
- `k_scale`
- quantization granularity enum / flags
- block sizes / warp grouping metadata (if needed for indexing)

Avoid relying on generic `aux_tensors` for core quantization metadata long-term. `aux_tensors` can be used only for early experiments.

### A4) Shape / alignment validation for quantized inputs

Add checks for:

- `q/k` INT8 alignment requirements (likely stricter/different from FP16/BF16)
- scale tensor shapes matching chosen granularity:
  - per-block
  - per-warp
  - per-thread (future)
- head dimension divisibility constraints for INT8 UMMA tile shapes

### A5) Plumb scale tensors into kernel launch signature

The experimental SM100 kernel will need `mQScale`, `mKScale` (or equivalent CuTe tensors) threaded through launch and JIT signature.

---

## B. Kernel Class Dtype Model Refactor (`flash_attn/cute/flash_fwd_sm100_qk_int8.py`)

This is the most important structural change.

### B1) Decouple dtype domains

Replace the implicit “Q/K/V share dtype” model with explicit dtype roles:

- `qk_input_dtype` (INT8 for experimental path)
- `v_input_dtype` (FP16/BF16 initially)
- `qk_acc_dtype` (INT32 for INT8 UMMA, then converted to FP32 in softmax)
- `softmax_work_dtype` (FP32)
- `p_store_dtype` (likely FP16/BF16 for phase 1, independent of QK input dtype)
- `pv_acc_dtype` (keep FP32 initially)
- `o_dtype`

In practice, this means:

- preserve `self.v_dtype`, `self.o_dtype`
- stop using `self.q_dtype` as a proxy for anything except Q operand representation
- add `self.p_dtype` (or equivalent)

### B2) Remove shared-dtype assertions

Current logic rejects any mismatch between `q/k/v`. Replace with:

- `q_dtype == k_dtype`
- `q_dtype` valid for the selected QK mode (`Int8` in experimental path)
- `v_dtype` valid for PV path (`Float16`/`BFloat16`)
- `o_dtype` compatible with output epilogue

### B3) Audit all `.width`-based sizing assumptions

`_setup_attributes()` and related staging logic use dtype width as a proxy for smem pressure. With Q/K INT8:

- Q/K smem demand drops
- V/P path remains unchanged
- optimal `kv_stage`, `q_stage`, etc. may shift

Start with conservative settings matching existing behavior, then retune.

---

## C. QK MMA Path Conversion to INT8 (Core Compute Change)

## C1) Switch QK tiled MMA operand type

In the QK path (`tiled_mma_qk` creation), use INT8 operand type instead of FP16/BF16.

The SM100 descriptor helpers already support this:

- `mma_sm100_desc.py` maps `cutlass.Int8` for A/B operand encoding
- `mma_sm100_desc.py` maps `cutlass.Int32` accumulator encoding

### C2) Use INT32 QK accumulators

For QK:

- operand type: INT8
- accumulator type: INT32

Then convert to FP32 before softmax math.

### C3) Validate Q/K operand layouts and major modes for INT8 UMMA

Current major-mode choices for QK are tuned for FP16/BF16 layouts. Confirm compatibility for INT8:

- operand major mode
- descriptor packing
- tile sizes (`mma_tiler_qk`)
- stride/alignment assumptions

This may require separate tile presets for the INT8 QK path.

---

## D. Score Dequantization Strategy (How to apply `q_scale * k_scale`)

This is the key numeric integration point.

### D1) Do NOT dequantize Q/K tensors explicitly

Avoid materializing FP16/BF16 Q/K after quantization; that loses the bandwidth/computation advantage.

### D2) Fold dequantization into score scaling

Use the SageAttention pattern:

- compute per-tile (or per-subtile) `dequant_scale = q_scale * k_scale[...]`
- multiply the original softmax scale by `dequant_scale`
- perform softmax on already-correctly-scaled scores

This keeps the score pipeline compact and avoids extra elementwise passes.

### D2.1) **Important: FA4 softmax scale is currently treated as a scalar**

In the current SM100 kernel, `SoftmaxSm100.create(...)` is called once per work tile with a **scalar** `softmax_scale_log2`. In baseline FP16/BF16, that’s correct because the only scaling is the global `1/sqrt(d)` (and maybe some constant conversions).

With Sage-style quantization, `dequant_scale = q_scale * k_scale[...]` generally varies by:

- `m_block` (Q tile along sequence)
- `n_block` (K tile along sequence)
- sometimes warp/subtile within a CTA (for per-warp granularity)

That means you cannot represent the full scaling as a single scalar per tile unless you intentionally choose a **constant scale scheme** (see D3.1).

So the INT8 plan must include one of these implementation choices:

- **Option A (lowest code churn, lower accuracy):** use a constant scale per (batch, head) (or per head) so `dequant_scale` is uniform across blocks; then you can keep `SoftmaxSm100.create(softmax_scale_log2, ...)` unchanged and just adjust the scalar once.
- **Option B (recommended, matches Sage):** keep per-block/per-warp scales, and refactor softmax usage so the effective scale can change per `n_block` iteration. Concretely:
  - change `SoftmaxSm100` to support an **updateable** scale (e.g. `softmax.set_scale_log2(scale_log2)`), or
  - bypass `SoftmaxSm100`’s internal scaling and explicitly apply `score *= (softmax_scale * dequant_scale)` in FP32 before `update_row_max(...)`, while ensuring exp2 uses the correct base conversion.
- **Option C (recommended for Phase 1; low churn + correct):** keep `SoftmaxSm100.scale_log2` fixed (e.g. `softmax_scale * LOG2_E`) and apply **per-block dequant scaling directly to the score fragment** before calling any `SoftmaxSm100` methods:
  - after loading the QK accumulator fragment, convert to FP32 and multiply by `dequant_scale`
  - then call `update_row_max(...)`, `scale_subtract_rowmax(...)`, and `apply_exp2_convert(...)` as usual

Option C works because the running `row_max/row_sum` are defined over the *true* logits domain; scaling each block’s logits into that true domain before updating softmax state preserves correctness even when `dequant_scale` changes per `n_block`.

If we don’t do this, we’ll silently compute the wrong logits while still producing numerically “reasonable” outputs, which is a high-risk failure mode.

### D3) Apply scaling at the correct granularity

Granularity must match how QK tiles accumulate:

- per-block: one scale for a larger tile region
- per-warp: one scale per warp tile (recommended first)
- per-thread: future/high-accuracy option

The indexing scheme must align with:

- CTA tile indices
- Q/K tile iteration (`n_block`, inner K-loop step)
- head/group dimensions

### D3.1) Efficiency note: scale lookup overhead is real

Per-block/per-warp scaling introduces extra global memory reads and indexing work in the softmax stage. To keep this efficient:

- stage the relevant `q_scale` / `k_scale` values into registers (or shared) once per `n_block` loop iteration
- avoid divergent indexing across threads (prefer warp-uniform scale values)
- prefer `FP16/BF16` scale storage only if accuracy allows; `FP32` is simplest/robust first

### D3.2) Varlen / paged-KV scale indexing must be defined explicitly

The kernel supports varlen (`cu_seqlens_*`), optional `seqused_*`, and paged KV (`page_table`). In these modes:

- `m_block`/`n_block` are not simple `block_idx * block_size` in a single contiguous sequence
- scale tensors must be indexed using the *logical token index within the sequence*, not the raw physical pointer order

So the plan needs an explicit convention, e.g.:

- `q_scale` indexed by `(batch, head, q_block_in_sequence)`
- `k_scale` indexed by `(batch_k, head_kv, k_block_in_sequence)` (or page-block index if paged)

Without a clear convention, correctness debugging becomes guesswork.

### D3.3) GQA/MQA scale conventions

FA4 supports GQA/MQA (`num_head` vs `num_head_kv`) and an optional packed-GQA layout. For INT8 QK:

- decide whether `q_scale` is per Q-head and `k_scale` per KV-head (typical), or both per Q-head
- ensure the kernel’s head-index math matches the chosen convention, especially in packed-GQA mode

### D4) Convert INT32 accumulators to FP32 before softmax operators

Softmax utilities operate in FP32. You need a conversion path:

- INT32 accumulator fragment -> FP32 fragment
- apply mask/score_mod and row-max/update in FP32

The exact point can be:

- immediately after TMEM load of score fragments
- or at the first softmax transform step

---

## E. Softmax / TMEM / P-Buffer Changes

## E1) Introduce `p_store_dtype` (do not reuse `q_dtype`)

Current code uses `self.q_dtype` when constructing/storing `P`-like intermediates. With Q/K INT8 this breaks.

Update:

- `tP_layout` creation to use `self.p_dtype`
- any recasts from FP32 softmax output into TMEM/smem storage to use `self.p_dtype`

### E2) Fix softmax output recast path

The softmax step currently recasts FP32 temporary storage using `self.q_dtype`. That must become `self.p_dtype` (or another explicit softmax-output storage dtype).

### E3) Re-audit `tilePlikeFP32` width conversions

The `tilePlikeFP32` transformations are tied to:

- QK tile width
- FP32 width
- V/P dtype width

These need validation once QK accumulator type and P storage dtype are no longer coupled to Q dtype.

### E4) Preserve mask / score_mod semantics

Any dequant scaling changes must preserve ordering and semantics relative to:

- custom `score_mod`
- causal/local masking
- sink/bias paths (if present)

The QK INT8 path should be numerically equivalent to baseline within expected quantization error.

### E5) Fix `rescale_threshold` and other dtype-conditional softmax behavior

Baseline code sets `rescale_threshold` based on `self.q_dtype.width` (e.g. FP16/BF16). Once `self.q_dtype` becomes INT8, this logic will choose the “non-FP16” branch even though the **softmax math is still FP32** after conversion.

The INT8 plan must explicitly decide what `rescale_threshold` policy should be for:

- INT8 QK accumulators converted to FP32 logits
- potentially larger dynamic range if scales are coarse

This is both correctness- and performance-relevant (rescaling triggers extra work and affects numerical stability).

---

## F. Shared Memory Layouts, TMA, and Copy Atoms

## F1) Q/K shared-memory layouts for INT8

Revalidate / specialize:

- `sQ_layout`
- `sK_layout`

for INT8 element width and packing. Points to check:

- alignment
- stage stride
- bank conflict patterns
- descriptor assumptions for UMMA loads

### F2) TMA descriptor / copy compatibility for INT8 operands

If TMA paths assume FP16/BF16 payload characteristics, update:

- descriptor construction
- alignment checks
- copy atoms / transaction shapes

If CuTe utilities already handle generic dtypes here, this may be minimal; verify, don’t assume.

### F3) TMEM / barrier choreography likely stays, but re-measure

The synchronization/barrier sequence can likely remain unchanged in phase 1, but:

- producer/consumer balance changes
- QK phase may complete sooner
- overlap opportunities may shift

Retuning can come after correctness.

---

## G. Kernel Launch Config / Tuning Changes (Performance Work, After Correctness)

## G1) Revisit tile shapes for INT8 QK

The best QK tile shapes for FP16/BF16 are not guaranteed optimal for INT8 UMMA.

Likely tuning axes:

- `mma_tiler_qk`
- CTA M/N tile sizes
- stage counts
- warpgroup role balance (MMA vs softmax/correction)

### G2) Rebalance staging heuristics

Because Q/K are smaller:

- more Q/K stages may fit
- or shared memory can be reallocated to improve V/P path or occupancy

Start with existing settings for stability, then benchmark.

### G3) Consider separate config tables for INT8-QK mode

Keep the default FP16/BF16 path untouched and add a separate configuration branch keyed by:

- arch = SM100
- qk mode = INT8
- head dim / head dim_v
- causal/local flags

---

## H. Quantization Preprocessing Integration (Using Copied SageAttention Sources)

## H1) Build a small FlashAttention-side extension from `csrc/sageattention_quant/`

Already copied:

- fused quantization kernels and pybind entrypoints
- supporting headers used by the fused implementation

Needed next (later, not in this file rename task):

- add a build target / extension wiring in `flash-attention` for the preprocessing module
- namespace the Python import to avoid collisions with SageAttention installs

### H2) Start with per-warp or per-block Q/K quantization

Recommended order:

1. **Per-warp INT8** (best balance of accuracy/overhead)
2. **Per-block INT8** (simpler shape/indexing fallback)
3. Per-thread (later)

### H3) Keep quantization as an explicit prepass initially

Do not quantize inside the CuTe kernel in phase 1. Reasons:

- significantly more invasive changes
- adds pipeline pressure and synchronization complexity
- makes correctness debugging harder

Prepass + native INT8 QK MMA already captures most of the QK bandwidth win.

### H4) Layout agreement between prepass and CuTe kernel

The SM100 kernel transposes tensor views early (e.g. `Q` to a `(s, d, h, b)`-like view, `K` similarly). A preprocessing quantization step must:

- either quantize in the original framework layout and rely on the same view-transpose (pointer-preserving) behavior, **or**
- quantize directly in the transposed view layout to simplify scale indexing and reduce ambiguity

The plan should choose one convention and stick to it so that `(m_block, n_block)` scale indexing is unambiguous across varlen/paged modes.

---

## I. Experimental API / Kernel Signature Proposal (Phase 1)

This is one practical shape for the experimental path. Exact tensor layout can be adjusted.

## I0) **Concrete scale tensor conventions (choose and enforce)**

This plan must define *exact* `q_scale/k_scale` shapes and indexing rules up front; otherwise dequant scaling bugs will dominate debugging time.

The lowest-friction approach is to **match SageAttention’s existing scale tensor conventions**, since we intend to reuse its quantization prepass:

- `per_block_int8` (SageAttention):  
  - `q_scale.shape == (B, Hq, ceil_div(Sq, BLKQ))`  
  - `k_scale.shape == (B, Hkv, ceil_div(Sk, BLKK))`
- `per_warp_int8` (SageAttention):  
  - `q_scale.shape == (B, Hq, ceil_div(Sq, BLKQ) * (BLKQ // WARPQ))`  
  - `k_scale.shape == (B, Hkv, ceil_div(Sk, BLKK))`

Concrete sources for these shapes:

- `/workspace/SageAttention/sageattention/quant.py` `per_block_int8` and `per_warp_int8`.

### I0.1) MVP recommendation for **CuTe integration** (not for final accuracy)

For the first working CuTe INT8-QK path, prefer:

- `Q`: **per-block** scale (`q_scale.shape == (B, Hq, q_blocks)`)
- `K`: **per-block** scale (`k_scale.shape == (B, Hkv, k_blocks)`)

Reason: `per-warp` `q_scale` requires correct warp/subtile-to-row mapping inside the SM100 kernel. That mapping is doable, but it is an additional correctness surface area that is independent of INT8 UMMA bring-up.

Once the per-block path is correct and stable, add `per_warp` Q scaling as phase 2.

### I0.2) Dense (non-varlen, non-paged) indexing rules

Assume logical layouts are the standard attention layouts (before any CuTe view transposes):

- `q_int8`: `(B, Sq, Hq, D)` (or `(B, Hq, Sq, D)` if we standardize on HND internally)
- `k_int8`: `(B, Sk, Hkv, D)`
- `v`: `(B, Sk, Hkv, Dv)`

Define:

- `q_block = floor(q_row / BLKQ)` where `q_row` is the **token index within the sequence** and `BLKQ` matches the Q-tile height used by the kernel (typically 128).
- `k_block = floor(k_col / BLKK)` where `k_col` is the K-token index within the sequence and `BLKK` matches the K-tile width for scaling purposes (choose 128 initially if the kernel’s `N` tile is 128).
- `kv_head = floor(q_head / (Hq / Hkv))` for standard GQA mapping (or whatever mapping FA4 uses in packed-GQA mode; it must be consistent).

Then in the QK loop for a given `(batch_idx, q_head, m_block, n_block)`:

- `q_scale_val = q_scale[batch_idx, q_head, q_block_idx]` where `q_block_idx` corresponds to the kernel’s `m_block` in **sequence-block units**
- `k_scale_val = k_scale[batch_idx, kv_head, k_block_idx]` where `k_block_idx` corresponds to the kernel’s `n_block` in **sequence-block units**
- `dequant_scale = q_scale_val * k_scale_val`

Important: the kernel currently passes `m_block=self.q_stage * m_block + stage` into softmax/mask logic. For INT8 scaling, define explicitly whether `q_block_idx` uses:

- the stage-adjusted `m_block` (recommended: match what masking/row indexing uses), or
- the base `work_tile` block index

and keep it consistent everywhere (masking, scale lookup, and any LSE indexing).

### I0.3) Varlen conventions (what must change)

Supporting varlen (`cu_seqlens_*` / `seqused_*`) requires a varlen-aware quantization prepass and scale indexing convention.

Define one of these and implement it consistently:

- **Varlen Option A (recommended):** store `q_scale/k_scale` in *block-index space per sequence*, i.e. `q_scale[batch, head, block_in_sequence]` where `block_in_sequence` starts at 0 for each sequence. Then the kernel uses its per-sequence `m_block`/`n_block` indices directly.
- **Varlen Option B:** store scales in global token-index space (over the packed `total_q/total_k`), and derive `q_block/k_block` using `cu_seqlens_*` offsets. This is more error-prone and adds indexing overhead.

MVP pragmatic recommendation:

- if we want to ship a working INT8-QK path quickly, **scope varlen out** for phase 1 and add it in phase 2 with Option A.

### I0.4) Paged-KV conventions (what must change)

Paged KV (`page_table`) breaks the “contiguous K blocks” assumption.

You need an explicit scale mapping that matches the physical K storage, e.g.:

- scales per page (or per page-block) and index via `page_table`, or
- preprocess a contiguous K for quantization (defeats the point for large caches), or
- write a paged-aware quantization kernel that emits scales aligned to the paged layout.

MVP pragmatic recommendation:

- scope paged-KV out of phase 1; add it later once the dense path is stable.

### Inputs

- `q_int8`: quantized Q
- `k_int8`: quantized K
- `v`: FP16/BF16 V
- `q_scale`: FP32 (or FP16/BF16 if validated) scale tensor
- `k_scale`: FP32 (or FP16/BF16 if validated) scale tensor
- existing FA4 metadata:
  - `cu_seqlens_q`, `cu_seqlens_k`
  - `seqused_q`, `seqused_k`
  - `page_table`
  - `window_size_left/right`
  - `learnable_sink`
  - masks / score mods

### Outputs

- `o` (same as baseline FA4)
- optional `lse`

### Mode metadata

- quantization granularity enum (`per_block`, `per_warp`)
- quant block sizes (`BLKQ`, `BLKK`) and/or warp partition metadata

---

## J. Correctness and Benchmarking Requirements

## J1) Always compare against baseline FA4 SM100

Use the existing A/B benchmark:

- `benchmarks/benchmark_cute_sm100_variants.py`

Required metrics:

- runtime
- TFLOPs
- `max_abs_err`
- `NRMSE`

### J2) Benchmark both kernel-only and end-to-end

Kernel-only gains can look good while pre-quantization overhead erases net speedup.

Measure:

1. **Kernel-only**
   - assumes quantized Q/K + scales already available
2. **End-to-end**
   - includes quantization prepass

### J3) Add numerical coverage matrix

At minimum:

- head dims: 64 / 128 (then others)
- seq lens: 1K / 2K / 4K / 8K
- causal and non-causal
- BF16 and FP16 V
- MHA / GQA representative shapes

---

## K. Phased Implementation Plan (Recommended Order)

## Phase 0 — Plumbing and scaffolding (no kernel math changes)

- [ ] Rename/repurpose experimental kernel file (`flash_fwd_sm100_qk_int8.py`) ✅ already done
- [ ] Add experimental entrypoint/wrapper for QK INT8 path
- [ ] Define scale tensor shapes and granularity metadata
- [ ] Build and expose Sage-derived quantization prepass module

## Phase 1 — Functional QK INT8 kernel path

- [ ] Relax interface dtype checks for experimental path
- [ ] Thread `q_scale/k_scale` into kernel signature
- [ ] Refactor dtype roles (`qk_input_dtype`, `v_dtype`, `p_dtype`, etc.)
- [ ] Switch QK MMA to INT8->INT32
- [ ] Convert score fragments to FP32 before softmax
- [ ] Fold dequant scales into score scaling
- [ ] Fix P storage/recast paths to use `p_dtype`
- [ ] Validate correctness vs baseline FA4

## Phase 2 — Performance tuning

- [ ] Tune tile shapes and stage counts for INT8 QK
- [ ] Tune quantization granularity (`per_block` vs `per_warp`)
- [ ] Reduce scale tensor overhead / improve indexing locality
- [ ] Rebalance overlap between QK MMA and softmax/PV phases

## Phase 3 — Sage-style accuracy/perf enhancements

- [ ] Add optional K smoothing (mean subtraction)
- [ ] Add correction/LSE compensation if needed
- [ ] Explore V low-precision paths (FP8/FP4-inspired)
- [ ] Explore probability quantization / more aggressive fusion

---

## L. Common Pitfalls / Failure Modes to Expect

- **Hidden dtype coupling**: a path still uses `self.q_dtype` for P/softmax/TMEM layout.
- **Scale misindexing**: wrong `q_scale/k_scale` tile mapping causes large error with no obvious crash.
- **Scale treated as constant**: per-block/per-warp scales exist, but softmax still uses a scalar `softmax_scale_log2`.
- **Descriptor mismatch**: INT8 UMMA descriptor encoding and actual layout/stride assumptions differ.
- **Alignment issues**: INT8 tensor alignment requirements differ from FP16/BF16 assumptions.
- **Benchmark bias**: measuring only kernel time and not the quantization prepass.
- **Cache contamination**: CuTe compile cache reusing a previously compiled variant during A/B testing.

### L1) Compile-cache keying must include INT8 mode

CuTe forward compilation caches typically key on shapes/options, and can accidentally reuse a compiled kernel if the key does not include:

- Q/K dtype (INT8 vs FP16/BF16)
- P storage dtype
- quant granularity mode

Even if the experimental benchmark clears caches, a real API integration must ensure the cache key is safe.

Concretely, `flash_attn/cute/interface.py` currently builds a `compile_key` around a single `dtype` and does not have separate `q_dtype/k_dtype/v_dtype` slots. The INT8-QK implementation must either:

- define a new entrypoint with a distinct compile cache, or
- expand `compile_key` to include at least:
  - `q_dtype`, `k_dtype`, `v_dtype`
  - `qk_mode` (fp16/bf16 vs int8)
  - `p_store_dtype`
  - quant granularity and block sizes (`BLKQ`, `BLKK`, and `WARPQ` if used)
  - scale dtypes (if not fixed to FP32)

### L2) Scale conventions must be tested explicitly

Before doing any performance tuning, add a small correctness harness that:

- runs a single `(B=1, Hq small, Sq small)` case
- prints/validates the chosen `(m_block, n_block) -> (q_scale_idx, k_scale_idx)` mapping
- tests at least one nontrivial GQA case (`Hq != Hkv`)

This catches indexing bugs that otherwise show up as “random” numerical error.

---

## M. Minimal First Milestone (Pragmatic Target)

The first milestone that is worth implementing and benchmarking is:

1. External Sage-derived Q/K quantization prepass (`per_warp` or `per_block`)
2. Experimental SM100 CuTe kernel variant with:
   - `Q/K` INT8 input
   - INT8->INT32 QK MMA
   - FP32 softmax
   - existing FP16/BF16 V + PV path
3. Dequant scaling folded into `softmax_scale`
4. A/B benchmark vs baseline FA4 using runtime + `max_abs_err` + `NRMSE`

This yields a clean baseline for deciding whether deeper Sage-style features (smoothing/correction/V quantization) are worth the additional complexity.
