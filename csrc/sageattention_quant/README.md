## SageAttention Quantization Kernel Snapshot

This directory contains a copied snapshot of the SageAttention fused quantization kernel sources from `/workspace/SageAttention/csrc` for planned INT8 QK integration experiments in FlashAttention CuTeDSL SM100 kernels.

Current status:

- Files are copied for reference and future build integration only.
- They are not wired into `flash-attention` build scripts yet.
- The copied `fused/` sources preserve SageAttention's original relative include structure (`../*.cuh`, `../*.h`).

Primary sources copied:

- `fused/fused.cu`
- `fused/fused.h`
- `fused/pybind.cpp`
- supporting utility headers used by the fused quantization implementation

Intended use:

- Reuse SageAttention's fused Q/K INT8 quantization (optionally with K mean subtraction) as a preprocessing stage for an experimental SM100 CuTeDSL attention path.
