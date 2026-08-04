# Triton Fused Preprocessing

## Operation contract

The Milestone 4 operation starts after resize/letterbox and host-to-device
transfer. Its input and output are:

```text
Input:  contiguous CUDA uint8 BGR [H, W, 3], values 0–255
Output: contiguous CUDA RGB [3, H, W], FP32 or FP16, values 0–1
```

This matches the corresponding Ultralytics transformations but deliberately
does not include resize, letterbox, batching, or transfer from CPU memory.

## Implementations

The trusted PyTorch reference performs `flip(-1)`, `permute(2, 0, 1)`, dtype
conversion, division by 255, and final contiguous materialization.

The Triton kernel assigns one program lane to one pixel. Each lane loads three
adjacent BGR bytes, multiplies them by `1/255`, and writes R, G, and B into
three separate output planes. A mask protects lanes beyond the image boundary
in the final block.

```text
input offset = pixel × 3

input:  [B0 G0 R0] [B1 G1 R1] ...
output: [R plane................]
        [G plane................]
        [B plane................]
```

The output tensor's allocated dtype controls whether Triton stores FP32 or
FP16 values.

## Correctness methodology

Correctness ran on a Modal NVIDIA L4 with PyTorch 2.13.0, CUDA 13.0, and
Triton 3.7.1. The matrix contained shapes 2×3, 5×7, 384×640, 640×640, and
641×639 in both FP32 and FP16. The small and irregular cases exercise channel
mapping and the final-block mask; realistic sizes verify full tensors.

Every Triton tensor was compared element-by-element with the PyTorch
reference. All 10 cases passed with zero maximum absolute difference and zero
mismatched elements.

## Performance methodology

CUDA events measure GPU execution. Each operation receives 30 warm-up and 200
measured iterations. Warm-up excludes Triton's first-use JIT compilation. The
PyTorch reference is measured before and after each Triton parameter sweep,
and those raw samples are combined to reduce simple order bias.

The benchmark includes steady-state output and intermediate allocations but
excludes Python wall time. Input tensors already reside on CUDA. Resize,
letterbox, CPU decode, host-to-device transfer, model inference, and
postprocessing are excluded.

Tested Triton configurations combine block sizes 128, 256, 512, and 1024 with
2, 4, and 8 warps. Raw measurements for every configuration are embedded in
`results/modal_l4_triton_preprocess_benchmark.json`.

## Result interpretation

Triton reduced median kernel time in all eight shape/dtype cases, with observed
speedups from 1.54× to 2.56×. At 640×640, the FP32 reference measured 0.0860 ms
and Triton 0.0543 ms; FP16 measured 0.0850 ms and approximately 0.054–0.055 ms.

The result demonstrates successful fusion, but the absolute 640×640 saving is
only about 0.032 ms. Against an approximately 13 ms end-to-end detector, that
component alone cannot produce a large pipeline speedup. Future integration
must measure whether transfer, resize, and framework boundaries dominate the
real preprocessing path.

Launch-parameter medians were usually separated by roughly one microsecond,
near the resolution/noise floor of these short kernels. Different cases chose
different nominal winners, so the data does not justify shape-specific tuning
yet. Block size 256 and four warps remain a simple robust default.
