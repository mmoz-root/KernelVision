# Milestone 6: PyTorch CUDA Extension

## Goal

Milestone 5 proved that the preprocessing calculation can run quickly in a
standalone CUDA program. Milestone 6 makes that kernel callable from PyTorch
and then places it inside the real Ultralytics YOLO pipeline.

The operation remains:

```text
uint8 BGR-HWC → float RGB-CHW → normalization to [0, 1]
```

Resize and letterbox are still performed by Ultralytics. The extension replaces
only channel reordering, layout conversion, dtype conversion, and
normalization.

## Architecture

```text
CLI/config --preprocessor
        ↓
UltralyticsBackend
        ↓ predictor=custom DetectionPredictor subclass
Ultralytics letterbox
        ↓ NumPy BGR-HWC
torch.from_numpy → copy uint8 tensor to CUDA
        ↓
cuda_extension_preprocess
        ↓
PyTorch JIT extension loader
        ↓
pybind C++ binding → ATen validation/allocation → CUDA kernel
        ↓
normalized FP32/FP16 BCHW tensor → YOLO → normal postprocessing
```

Important files:

- `csrc/preprocessing/torch_extension.cpp` exposes the native entry point to
  Python and dispatches to CUDA.
- `csrc/preprocessing/torch_extension_kernel.cu` validates the tensor,
  allocates the output, selects FP32 or FP16, and launches on PyTorch's current
  CUDA stream.
- `src/kernelvision/preprocessing/cuda_extension.py` lazily compiles/loads the
  extension and provides friendly Python validation.
- `src/kernelvision/backends/cuda_extension_predictor.py` overrides only
  Ultralytics preprocessing while preserving its letterbox and postprocessing.
- `src/kernelvision/backends/ultralytics_backend.py` selects the standard or
  extension predictor.

## Why the current CUDA stream matters

PyTorch operations are asynchronous and ordered on CUDA streams. Launching on
PyTorch's current stream preserves the dependency order between the
host-to-device copy, this kernel, and model inference without forcing a global
synchronization. A hard-coded or unrelated stream could let model inference
read the output before preprocessing completes.

## Correctness gates

The extension first passed 10 component cases: five image shapes in FP32 and
FP16, all with maximum absolute difference 0 against the PyTorch reference.

The complete YOLO pipeline was then checked on `bus.jpg` with `yolov8n.pt` on a
Modal NVIDIA L4. Detections were paired by class and highest intersection over
union rather than list index.

| Precision | Detections | Max box difference | Max confidence difference | Minimum IoU |
|---|---:|---:|---:|---:|
| FP32 | 6 | 0 | 0 | 1.0 |
| FP16 | 6 | 0 | 0 | 1.0 |

Report: `results/modal_l4_cuda_extension_pipeline.json`.

## Component-boundary benchmark

The allocating public extension API was compared with the allocating PyTorch
and Triton APIs and the preallocated standalone CUDA boundary. At 640×640:

| Precision | PyTorch | Triton | Standalone CUDA | CUDA extension |
|---|---:|---:|---:|---:|
| FP32 | 0.047524 ms | 0.027238 ms | 0.005601 ms | 0.006431 ms |
| FP16 | 0.045690 ms | 0.026854 ms | 0.004116 ms | 0.006205 ms |

The extension was about 4.2× faster than the Triton public API and 7.4× faster
than the PyTorch reference at 640×640. This is an integration-boundary result,
not proof that the device instructions alone are faster: the CUDA arithmetic is
the same naive kernel carried forward from Milestone 5.

Reports:

- `results/modal_l4_cuda_extension_benchmark.json`
- `benchmarks/raw/modal_l4_cuda_extension_benchmark.csv`

## Complete-pipeline benchmark

For each precision, standard and extension preprocessing were measured in both
execution orders with 30 warmups and 200 samples per run. Extension compilation
was completed before controlled timing.

| Precision and order | Preprocess median change | End-to-end median change |
|---|---:|---:|
| FP32, baseline then extension | -22.46% | -3.14% |
| FP32, extension then baseline | -20.94% | +1.30% |
| FP16, baseline then extension | -20.83% | -4.05% |
| FP16, extension then baseline | -22.06% | -3.56% |

Across the four paired trials, the median preprocessing change was -21.50%
(about -0.462 ms). The median end-to-end change was -3.35% (about -0.659 ms),
but one trial regressed by 1.30%. Therefore the preprocessing-stage improvement
is consistent; the small whole-pipeline improvement is not fully separated
from run-to-run noise by this experiment.

Report: `results/modal_l4_cuda_extension_pipeline_benchmark.json`.

## Run the validation

```bash
modal run scripts/modal_cuda_extension_build.py
modal run scripts/modal_cuda_extension_correctness.py
modal run scripts/modal_cuda_extension_benchmark.py
modal run scripts/modal_cuda_extension_pipeline.py
modal run scripts/modal_cuda_extension_pipeline_benchmark.py
```

## Main lesson

A fast isolated kernel is necessary but not sufficient. Integration adds tensor
allocation, Python/C++ dispatch, a CPU-to-GPU copy, letterboxing, inference, and
postprocessing. Optimization claims must name the measured boundary: kernel,
public preprocessing API, preprocessing stage, backend call, or end-to-end.
