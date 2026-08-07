# KernelVision

**KernelVision** is an end-to-end GPU-optimization study of an object-detection inference pipeline.

Instead of treating a trained detector as a black box, the project measures and improves the complete path from input image to final bounding boxes:

```text
Image / video
    ↓
Preprocessing
    ↓
Model inference
    ↓
Postprocessing
    ↓
Annotated output + performance report
```

The project progressively compares implementations using **PyTorch, Triton, CUDA C++, ONNX, and TensorRT**.

> Status: Milestones 0–4 are implemented. Milestone 5 now has a standalone
> naive CUDA preprocessing kernel validated against PyTorch on a Modal L4;
> optimization and comparative benchmarking remain unfinished.

---

## Motivation

My previous computer-vision project focused on training and evaluating a YOLOv8 military-vehicle detector. KernelVision moves one layer deeper:

- Where does inference time actually go?
- How much time is spent outside the neural network?
- Can preprocessing be fused into one GPU kernel?
- How do Triton and CUDA implementations compare?
- How much does TensorRT improve model execution?
- Which optimizations improve end-to-end latency rather than only microbenchmarks?

The goal is to connect GPU programming concepts to a real ML system.

---

## Project Goals

The minimum viable project will include:

- A correct image and video detection pipeline
- Component-level and end-to-end benchmarking
- PyTorch FP32 and FP16 comparison
- Fused preprocessing in Triton
- The same preprocessing operation in CUDA
- A PyTorch custom CUDA operator
- ONNX export
- TensorRT FP16 inference
- Correctness validation
- Nsight profiling
- A reproducible benchmark report and demo

---

## Pipeline

```text
Input image or video
        ↓
Decode frame
        ↓
Resize / letterbox
        ↓
Normalize + HWC-to-CHW + dtype conversion
        ↓
Object-detection model
        ↓
Decode predictions
        ↓
Confidence filtering + NMS
        ↓
Draw detections
        ↓
Output video/image + latency report
```

The initial version uses standard framework operations. Custom kernels are introduced only after the baseline is correct and measurable.

---

## Implementations

### 1. PyTorch Baseline

- Standard preprocessing
- PyTorch/Ultralytics model execution
- Framework postprocessing
- FP32
- Batch size 1
- Synchronous execution

### 2. PyTorch FP16

- Same pipeline
- FP16 model execution where supported
- Output comparison against FP32

### 3. Triton Preprocessing

The first custom Triton kernel fuses:

```text
uint8 HWC image
    ↓
FP16/FP32 CHW tensor
    + normalization
```

Resize and letterbox fusion are later extensions.

### 4. CUDA Preprocessing

The same operation is implemented in CUDA:

- Naive CUDA version
- Optimized CUDA version
- PyTorch C++/CUDA extension
- Comparison with PyTorch and Triton

### 5. TensorRT Inference

- PyTorch model exported to ONNX
- TensorRT FP16 engine
- Model-only and end-to-end benchmark comparison
- Output validation against PyTorch

---

## Planned Repository Structure

```text
kernelvision/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
├── src/
│   └── kernelvision/
│       ├── cli.py
│       ├── pipeline.py
│       ├── backends/
│       ├── preprocessing/
│       ├── postprocessing/
│       ├── benchmarking/
│       ├── visualization/
│       └── utils/
├── csrc/
│   ├── preprocess_binding.cpp
│   └── preprocess_kernel.cu
├── scripts/
├── tests/
├── benchmarks/
├── results/
├── assets/
└── docs/
```

The structure will grow incrementally. Empty modules will not be created before they are needed.

---

## Development Roadmap

### Milestone 0 — Environment and Skeleton

- Minimal Python package
- Dependency setup
- Device and environment report
- Basic command-line interface

### Milestone 1 — Baseline Detection Pipeline

- Image inference
- Video inference
- Detection visualization
- Saved annotated outputs
- Configurable model, input size, confidence threshold, and device

### Milestone 2 — Benchmark Harness

Measure:

- Decode time
- Preprocessing time
- Host-to-device transfer time
- Model inference time
- Postprocessing time
- End-to-end latency
- Frames per second
- GPU memory where practical

Report:

- Mean
- Median
- P95
- Minimum and maximum
- Warm-up and measurement counts
- Hardware and software configuration

### Milestone 3 — PyTorch FP32 vs FP16

- [x] Correctness comparison
- [x] Model-only benchmark
- [x] End-to-end benchmark
- [x] Memory comparison

### Milestone 4 — Triton Fused Preprocessing

- [x] PyTorch reference
- [x] Fused BGR-to-RGB, HWC-to-CHW, normalization, and dtype conversion
- [x] Multiple image sizes
- [x] FP32 and FP16 correctness tests
- [x] Triton block-size and warp-count experiments
- [x] PyTorch versus Triton benchmark

### Milestone 5 — CUDA Preprocessing

- [x] Standalone raw-CUDA harness and deterministic correctness protocol
- [x] Naive CUDA implementation and Modal L4 correctness validation
- [ ] Optimized CUDA implementation
- [ ] Block-size and memory-access experiments
- [ ] PyTorch versus Triton versus CUDA benchmark

### Milestone 6 — PyTorch CUDA Extension

- C++ binding
- CUDA kernel
- Python wrapper
- Input validation
- Automated tests
- Pipeline integration

### Milestone 7 — ONNX and TensorRT

- ONNX export
- TensorRT FP16 engine
- TensorRT backend
- Output validation
- PyTorch and TensorRT comparison

### Milestone 8 — Final Demo and Report

- Annotated image/video demo
- Final benchmark tables and plots
- Architecture diagram
- Profiler findings
- Limitations and failed experiments
- Reproduction instructions

---

## Benchmark Methodology

Performance claims will follow these rules:

- Warm up GPU operations before measurement.
- Use CUDA events for isolated GPU operations when possible.
- Use wall-clock timing for end-to-end latency.
- Synchronize correctly around GPU timings.
- Report batch size, input shape, dtype, and GPU model.
- Separate preprocessing, transfer, inference, postprocessing, and end-to-end results.
- Prefer median and P95 latency over only the mean.
- Save raw timing samples.
- Exclude one-time compilation from steady-state benchmarks.
- State whether allocation and data transfer are included.
- Verify correctness before benchmarking.

Initial configurable defaults:

```text
Warm-up iterations: 30
Measured iterations: 200
Batch size: 1
Primary input size: 640 × 640
Dtypes: FP32 and FP16
```

## Milestone 3 Results — FP32 vs FP16

The final experiment used YOLOv8n, batch size 1, a 640 × 640 model input,
confidence 0.25, PyTorch 2.13.0, CUDA 13.0, Ultralytics 8.4.115, and one Modal
NVIDIA L4. Both precisions ran in the same container with 30 warm-up and 200
measured iterations per run. The order was reversed across two trials:
FP32→FP16, then FP16→FP32.

| Metric | FP32 | FP16 | Observed FP16 change |
|---|---:|---:|---:|
| Inference median, trial 1 | 5.663 ms | 5.627 ms | -0.63% |
| Inference median, trial 2 | 5.764 ms | 5.548 ms | -3.74% |
| End-to-end median, trial 1 | 13.814 ms | 12.901 ms | -6.61% |
| End-to-end median, trial 2 | 13.673 ms | 12.675 ms | -7.30% |
| Mean-based throughput, trial 1 | 72.231 FPS | 76.040 FPS | +5.27% |
| Mean-based throughput, trial 2 | 72.959 FPS | 77.569 FPS | +6.32% |
| Peak allocated GPU memory | 27.812 MB | 13.981 MB | -49.73% |

FP16 consistently improved latency, but its model-only speedup was modest for
this small batch-size-1 model. The larger end-to-end difference includes
framework and CPU-stage variation and should not be interpreted as pure GPU
compute acceleration.

On the `bus.jpg` correctness case, FP32 and FP16 both produced six detections
and all six matched by class with IoU ≥ 0.5. Mean matched IoU was 0.998210,
minimum matched IoU was 0.997358, the largest coordinate change was 0.467 px,
and the largest confidence change was 0.000737. This validates the chosen
sample, not dataset-wide detection accuracy or mAP.

Reports with raw samples:

- `results/modal_l4_fp32_vs_fp16_correctness.json`
- `results/modal_l4_fp32_vs_fp16_performance.json`

## Milestone 4 Results — Fused Triton Preprocessing

The fused operation converts one contiguous CUDA `uint8` BGR-HWC image into a
normalized, contiguous RGB-CHW FP32 or FP16 tensor. Resize, letterbox, batching,
and host-to-device transfer remain outside this kernel.

The Triton output matched the PyTorch reference exactly in all 10 correctness
cases: five shapes—including irregular dimensions—times two output dtypes.
Maximum absolute difference and mismatched-value count were both zero.

CUDA-event microbenchmarks used 30 warm-ups and 200 measured iterations per
operation on the same Modal NVIDIA L4:

| Input shape | Dtype | PyTorch median | Best Triton median | Speedup |
|---|---:|---:|---:|---:|
| 320 × 320 | FP32 | 0.0891 ms | 0.0532 ms | 1.67× |
| 320 × 320 | FP16 | 0.0901 ms | 0.0553 ms | 1.63× |
| 384 × 640 | FP32 | 0.0870 ms | 0.0553 ms | 1.57× |
| 384 × 640 | FP16 | 0.0870 ms | 0.0553 ms | 1.57× |
| 640 × 640 | FP32 | 0.0860 ms | 0.0543 ms | 1.58× |
| 640 × 640 | FP16 | 0.0850 ms | 0.0543 ms | 1.57× |
| 720 × 1280 | FP32 | 0.1388 ms | 0.0543 ms | 2.56× |
| 720 × 1280 | FP16 | 0.0850 ms | 0.0553 ms | 1.54× |

The kernel is faster because it replaces multiple framework operations with
one launch and one pass over the data. At the primary 640 × 640 size, however,
the absolute saving is only about 0.032 ms. Relative microbenchmark speedup
therefore does not imply a similarly large end-to-end improvement.

Block sizes 128–1024 and warp counts 2, 4, and 8 were tested. Most median
differences were around one microsecond, so no universal optimum was supported
by the data. `BLOCK_SIZE=256` and `num_warps=4` remain the stable default.

Detailed methodology and interpretation are in
[`docs/triton_preprocessing.md`](docs/triton_preprocessing.md). Reports:

- `results/modal_l4_triton_preprocess_correctness.json`
- `results/modal_l4_triton_preprocess_benchmark.json`

## Milestone 5 Progress — Naive CUDA Preprocessing

The learner-written standalone CUDA kernel compiled with nvcc 13.0.48 and
matched the PyTorch reference exactly in all 10 shape/dtype cases on a Modal
NVIDIA L4. This is correctness evidence only; naive CUDA has not yet been
benchmarked or compared with Triton.

- `docs/cuda_preprocessing.md`
- `results/modal_l4_cuda_preprocess_correctness.json`

---

## Correctness

Every optimized implementation must be compared against a trusted reference.

Preprocessing tests verify:

- Output shape
- Output dtype
- Channel ordering
- Numerical values
- Value range
- Device placement
- Regular and irregular dimensions

Model/backend validation compares:

- Box coordinates
- Class predictions
- Confidence scores
- Number of detections
- Tolerance-sensitive differences between precisions

No optimization is considered successful if it breaks expected output behavior.

---

## Profiling

### Nsight Systems

Used for:

- CPU/GPU timeline
- Kernel-launch gaps
- Synchronization
- Memory transfers
- Pipeline idle time

### Nsight Compute

Used for custom CUDA kernels:

- Memory throughput
- Compute utilization
- Registers per thread
- Occupancy
- Warp-stall reasons
- Branch divergence
- Cache behavior
- Shared-memory behavior

Profiling is driven by questions, not by collecting every available metric.

---

## Current Development Setup

KernelVision currently implements the baseline inference pipeline, benchmark
harness, FP32/FP16 comparison, and fused Triton preprocessing through
Milestone 4. Use Python 3.11 for local development:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
kernelvision --help
kernelvision environment
```

The local Apple Silicon environment is used for package development and
dependency-free tests. GPU implementation, correctness checks, profiling, and
benchmarks will run on a Modal-hosted NVIDIA L4. Future benchmark reports will
record the exact container, CUDA, PyTorch, and GPU configuration.

## Example Commands

### Install

The current installation instructions are documented in
[Current Development Setup](#current-development-setup). Additional ML
dependencies will be introduced with the baseline inference milestone.

### Run Image Inference

```bash
kernelvision image \
  --model yolov8n.pt \
  --image path/to/example.jpg \
  --device cpu \
  --precision fp32 \
  --img-size 640 \
  --conf 0.25 \
  --output results/example_annotated.jpg
```

Use `--device 0` for the first CUDA GPU in the future Modal L4 environment.
Local CPU runs validate behavior only and are not benchmark results.

### Reproduce the Modal L4 precision experiments

```bash
modal run scripts/modal_precision_comparison.py
modal run scripts/modal_precision_benchmark.py \
  --warmup 30 \
  --iterations 200

modal run scripts/modal_triton_preprocessing.py
modal run scripts/modal_triton_benchmark.py \
  --warmup 30 \
  --iterations 200

modal run scripts/modal_cuda_preprocessing.py --block-size 256
```

### Run Video Inference

```bash
kernelvision video \
  --model yolov8n.pt \
  --input assets/sample_videos/example.mp4 \
  --output results/example_annotated.mp4 \
  --device cpu \
  --conf 0.25 \
  --img-size 640
```

### Benchmark

```bash
kernelvision benchmark \
  --model yolov8n.pt \
  --image path/to/example.jpg \
  --device cpu \
  --warmup 30 \
  --iterations 200 \
  --json-out results/local_report.json \
  --csv-out benchmarks/raw/local_samples.csv
```

Local CPU benchmarks validate the harness but are not published performance
results. Run the reproducible GPU baseline through Modal:

```bash
modal run scripts/modal_benchmark.py \
  --warmup 30 \
  --iterations 200
```

See [Benchmark Methodology](docs/benchmark_methodology.md) and
[Modal NVIDIA L4 Execution](docs/modal_l4.md) for timing boundaries and setup.

### Run Tests

```bash
pytest -q
```

---

## Benchmark Results

No results are published yet.

Planned summary format:

| Pipeline | Preprocess | Inference | Postprocess | End-to-end | FPS |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | 2.247 ms | 7.431 ms | 1.341 ms | 18.497 ms | 53.469 |
| PyTorch FP16 | TBD | TBD | TBD | TBD | TBD |
| Triton preprocessing | TBD | TBD | TBD | TBD | TBD |
| CUDA preprocessing | TBD | TBD | TBD | TBD | TBD |
| TensorRT FP16 | TBD | TBD | TBD | TBD | TBD |
| Fully optimized | TBD | TBD | TBD | TBD | TBD |

The PyTorch FP32 row is the Milestone 2 Modal NVIDIA L4 baseline. Latencies are
medians; FPS is derived from mean end-to-end latency. The run used YOLOv8n,
batch size 1, a 640 × 640 model input, 30 warm-up iterations, and 200 measured
iterations. End-to-end timing includes image decode, the synchronized backend
call, and in-memory visualization; output-file writing is excluded.

- End-to-end P95: 20.380 ms
- Peak allocated GPU memory: 27.812 MB
- Raw/report data: [`benchmarks/raw/modal_l4_baseline.csv`](benchmarks/raw/modal_l4_baseline.csv)
  and [`results/modal_l4_baseline.json`](results/modal_l4_baseline.json)

Each published result will include:

- GPU model
- CUDA version
- PyTorch version
- TensorRT version
- Input size
- Batch size
- Precision
- Warm-up count
- Measurement count

---

## Scope Boundaries

The initial version does **not** require:

- Training a new detector
- Custom CUDA NMS
- INT8 calibration
- TensorRT plugins
- CUDA Graphs
- Multi-GPU execution
- Multiple video streams
- Production deployment
- A complex frontend

These are possible stretch goals after the main project is complete.

---

## Stretch Goals

- Fuse resize/letterbox into custom preprocessing
- GPU bounding-box decode
- Fused decode + confidence filtering
- Optimized NMS
- INT8 TensorRT inference
- CUDA streams
- Double-buffered frame processing
- CUDA Graph capture
- TensorRT custom plugin
- Multi-stream video processing
- Benchmark on multiple NVIDIA architectures

---

## Development Principles

1. Correctness before speed.
2. Measure before optimizing.
3. Change one important variable at a time.
4. Do not claim improvements without reproducible results.
5. Preserve raw benchmark data.
6. Document failed experiments.
7. Prefer a simple complete pipeline over an unfinished advanced one.
8. Keep PyTorch, Triton, and CUDA comparisons fair.
9. Use custom kernels to answer real pipeline bottlenecks.
10. Keep the repository understandable to another student or engineer.

---

## Expected Outcome

The intended final result is a portfolio project demonstrating:

- GPU programming
- ML inference systems
- Triton
- CUDA C++
- Python/C++ integration
- PyTorch custom operators
- TensorRT
- Profiling
- Benchmark design
- Computer vision
- Reproducible engineering

A future resume bullet will be based on measured results:

> Built an end-to-end GPU-optimized object-detection inference pipeline using PyTorch, Triton, CUDA, ONNX, and TensorRT; implemented fused image preprocessing as Triton and custom PyTorch CUDA operators, profiled component-level bottlenecks, and improved measured latency from **[baseline]** to **[optimized]** on **[GPU]** while preserving detection correctness.

---

## License

A license will be selected before public release.
