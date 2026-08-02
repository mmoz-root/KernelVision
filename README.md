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

> Status: work in progress. Benchmark numbers will be added only after reproducible measurements.

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

- Correctness comparison
- Model-only benchmark
- End-to-end benchmark
- Memory comparison

### Milestone 4 — Triton Fused Preprocessing

- PyTorch reference
- Triton HWC-to-CHW + normalization + dtype conversion
- Multiple image sizes
- Correctness tests
- Triton parameter experiments
- PyTorch versus Triton benchmark

### Milestone 5 — CUDA Preprocessing

- Naive CUDA implementation
- Optimized CUDA implementation
- Block-size and memory-access experiments
- PyTorch versus Triton versus CUDA benchmark

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

## Example Commands

These commands are placeholders and will be updated as the implementation lands.

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run Image Inference

```bash
python -m kernelvision.cli \
  --model path/to/model.pt \
  --image assets/sample_images/example.jpg \
  --device cuda \
  --img-size 640 \
  --conf 0.25 \
  --output results/example_annotated.jpg
```

### Run Video Inference

```bash
python scripts/run_video.py \
  --model path/to/model.pt \
  --input assets/sample_videos/example.mp4 \
  --output results/example_annotated.mp4 \
  --device cuda
```

### Benchmark

```bash
python scripts/benchmark_pipeline.py \
  --config configs/baseline.yaml
```

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
| PyTorch FP32 | TBD | TBD | TBD | TBD | TBD |
| PyTorch FP16 | TBD | TBD | TBD | TBD | TBD |
| Triton preprocessing | TBD | TBD | TBD | TBD | TBD |
| CUDA preprocessing | TBD | TBD | TBD | TBD | TBD |
| TensorRT FP16 | TBD | TBD | TBD | TBD | TBD |
| Fully optimized | TBD | TBD | TBD | TBD | TBD |

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
