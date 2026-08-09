# Standalone CUDA Preprocessing

## Milestone boundary

Milestone 5 uses a standalone CUDA executable:

```text
Python Modal orchestrator
    ├── writes an exact uint8 BGR-HWC fixture
    ├── computes the trusted PyTorch result on CUDA
    └── launches a raw nvcc-built executable
            ├── reads the fixture
            ├── runs a CUDA kernel
            ├── writes raw FP32/FP16 RGB-CHW output
            └── writes native CUDA-event samples
```

There is no PyTorch C++ extension, ATen, pybind, tensor-pointer sharing, or
YOLO integration here. Those belong to Milestone 6. The file boundary adds no
cost to the kernel measurement: input and output buffers are allocated before
CUDA events, and H2D/D2H copies are outside them.

## Contract

```text
Input:  one contiguous CUDA uint8 BGR [H, W, 3], values 0–255
Output: one contiguous CUDA RGB [3, H, W], FP32 or FP16, values 0–1
```

Resize, letterbox, batching, CPU decode, and host-to-device transfer remain
excluded.

## Files

- `csrc/preprocessing/standalone_preprocess.cu`
  - raw CUDA runtime harness
  - typed FP32/FP16 dispatch
  - CUDA error checks
  - preallocated device buffers
  - CUDA-event sampling
  - the validated naive one-thread-per-pixel kernel
- `src/kernelvision/preprocessing/cuda_standalone.py`
  - deterministic fixture definition
  - raw CUDA-output reader
- `scripts/modal_cuda_preprocessing.py`
  - Modal NVIDIA L4 build and correctness matrix
- `tests/test_cuda_standalone.py`
  - portable tests for the cross-process data protocol

## Deterministic correctness protocol

For flat input byte offset `i`:

```text
value(i) = (37 × i + 13) mod 256
```

Python writes those exact bytes to a headerless fixture. Both the trusted
PyTorch operation and the CUDA executable consume that fixture. CUDA writes a
headerless output whose declared shape and dtype are known by the orchestrator.
The comparison uses `rtol=0`, `atol=1e-7` for FP32, and `atol=5e-4` for FP16.

The initial matrix matches Milestone 4: `2×3`, `5×7`, `384×640`, `640×640`,
and `641×639`, each in FP32 and FP16. The irregular sizes exercise the final
partial block.

## Exercise 1 — naive one-thread-per-pixel kernel

Open `naive_bgr_hwc_to_rgb_chw()` in
`csrc/preprocessing/standalone_preprocess.cu`. Remove its `#error` line and
implement only the marked body.

For pixel `i` and `P = H × W`:

```text
input:  B = 3i       G = 3i + 1       R = 3i + 2
output: R = i        G = P + i        B = 2P + i
```

Required steps:

1. Compute `i = blockIdx.x * blockDim.x + threadIdx.x`.
2. Return if `i >= pixel_count`.
3. Load the three interleaved bytes.
4. Convert and multiply by `1.0f / 255.0f`.
5. Store three planar values through `convert_output<Output>()`.

Do not optimize, use shared memory, vectorize, or change the harness yet. The
first goal is a simple trusted CUDA baseline.

After implementing it, run:

```bash
modal run scripts/modal_cuda_preprocessing.py --block-size 256
```

Acceptance criterion: all 10 cases pass before any benchmark or optimized
kernel is added. The one-sample event files created during correctness are
plumbing checks, not performance results.

## Naive correctness result

The learner-written naive kernel compiled with CUDA 13.0 (`nvcc 13.0.48`) and
passed all 10 cases on a Modal NVIDIA L4. FP32 and FP16 both had zero maximum
absolute difference and zero mismatched values against the trusted PyTorch
reference. This establishes correctness only; the one-event correctness runs
are not benchmark results.

Report: `results/modal_l4_cuda_preprocess_correctness.json`

## Naive CUDA baseline benchmark

### Methodology

The controlled comparison ran in one Modal NVIDIA L4 environment with PyTorch
2.13.0, CUDA 13.0, Triton 3.7.1, and nvcc 13.0.48. Each shape/dtype case first
rechecked Triton and CUDA against the PyTorch reference. All eight cases passed
with zero mismatched values.

Input and output buffers are resident before timing. H2D, D2H, compilation,
context creation, subprocess startup, and outer wall time are excluded. Triton
and CUDA outputs are preallocated; PyTorch retains its normal multi-operation
reference behavior.

Because these operations are only a few microseconds, every event sample times
100 repeated launches and divides the interval by 100. Each implementation has
30 warmups and 200 samples in each of three position-balanced rounds:

```text
round 1: PyTorch → Triton → naive CUDA
round 2: Triton → naive CUDA → PyTorch
round 3: naive CUDA → PyTorch → Triton
```

This produces 600 samples per implementation/case and 14,400 saved CSV rows.
Block size 256 is the declared naive CUDA/Triton baseline, with four Triton
warps. No parameter search contributes to these headline results.

### Results

Combined medians:

| Shape | Dtype | PyTorch | Triton | Naive CUDA | CUDA vs PyTorch | CUDA vs Triton |
|---|---:|---:|---:|---:|---:|---:|
| 320×320 | FP32 | 0.05736 ms | 0.02644 ms | 0.00306 ms | 18.73× | 8.64× |
| 320×320 | FP16 | 0.05919 ms | 0.02646 ms | 0.00284 ms | 20.87× | 9.33× |
| 384×640 | FP32 | 0.05902 ms | 0.02712 ms | 0.00389 ms | 15.17× | 6.97× |
| 384×640 | FP16 | 0.05869 ms | 0.02668 ms | 0.00384 ms | 15.28× | 6.95× |
| 640×640 | FP32 | 0.06169 ms | 0.02642 ms | 0.00557 ms | 11.08× | 4.75× |
| 640×640 | FP16 | 0.05840 ms | 0.02663 ms | 0.00414 ms | 14.12× | 6.44× |
| 720×1280 | FP32 | 0.10287 ms | 0.02644 ms | 0.01111 ms | 9.26× | 2.38× |
| 720×1280 | FP16 | 0.06460 ms | 0.02643 ms | 0.00815 ms | 7.93× | 3.24× |

At 640×640 FP32, naive CUDA saves 0.05612 ms versus PyTorch and 0.02085 ms
versus Triton. Against an approximately 13 ms detector, even the PyTorch-to-
CUDA component saving represents only about 0.43% of end-to-end latency before
integration overhead.

### Interpretation limits

- This is not an end-to-end YOLO result.
- Native CUDA launches from C++, while Triton and PyTorch dispatch through
  Python/framework paths. CUDA-event intervals can retain GPU idle gaps caused
  by those different submission paths; device instructions alone do not
  explain the full difference.
- Repeated launches reuse the same input and output buffers. This is a
  warm-cache steady-state benchmark, not a fresh-frame DRAM-throughput test.
- The historical Milestone 4 Triton result used an allocating wrapper and one
  operation per event interval. The current preallocated, amplified Triton
  result has a different boundary and does not show a kernel optimization.
- Some sub-microsecond differences between native CUDA rounds are large in
  percentage terms because the complete native operation is only a few
  microseconds. Raw per-round samples remain available.
- The naive result establishes the next optimization control. It does not
  establish that a more complex CUDA kernel will be faster or worthwhile.

Reports:

- `results/modal_l4_cuda_preprocess_benchmark.json`
- `benchmarks/raw/modal_l4_cuda_preprocess_benchmark.csv`
