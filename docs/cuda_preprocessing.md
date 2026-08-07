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
