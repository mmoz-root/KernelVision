# Development Log

## 2026-08-03 — Milestone 0 foundation

### Completed

- Added a minimal `src`-layout Python package.
- Added the `kernelvision` command-line entry point.
- Added an environment report that works before PyTorch is installed.
- Added portable automated tests for environment collection and formatting.
- Created a Python 3.11.9 local virtual environment and verified an editable
  package installation.
- Selected a Modal-hosted NVIDIA L4 as the consistent GPU target for CUDA work,
  profiling, and benchmark results.

### Validation

- `kernelvision --help` succeeds.
- `kernelvision --version` reports `0.1.0`.
- `kernelvision environment` reports the local platform and clearly identifies
  unavailable optional ML dependencies.
- `pytest -q` passes 2 tests locally.

### Incomplete

- No GPU measurements have been collected.

## 2026-08-03 — Milestone 1 baseline started

### Scope

- Implement one configurable image-inference path before adding video support.
- Validate functional correctness locally before running GPU work on the
  Modal-hosted NVIDIA L4.

### In progress

- Execute and inspect the full 30/200 Modal L4 baseline.

### Completed

- Added Ultralytics behind the optional `inference` dependency.
- Installed and verified the local CPU inference stack.
- Added a validated image-inference configuration.
- Added a lazily imported Ultralytics backend scaffold.
- Implemented the single-image Ultralytics prediction call.
- Added mocked tests that verify inference arguments and result-count handling
  without loading model weights.
- Added the `kernelvision image` command and image pipeline orchestration.
- Ran a real CPU smoke test with `yolov8n.pt` and Ultralytics' packaged
  `bus.jpg`; the annotated output contained plausible bus and person
  detections. This was a correctness check, not a benchmark.
- Added the `kernelvision video` command with explicit OpenCV decode and encode
  stages.
- Ran a real three-frame video smoke test and visually checked the first
  encoded frame. The output retained the source frame count, dimensions, and
  frame rate.
- `pytest -q` passes 7 tests locally.

## 2026-08-03 — Milestone 2 benchmark harness

### Completed

- Added device-aware CPU, MPS, and CUDA synchronization.
- Added raw per-iteration measurements for decode, framework preprocessing,
  inference, postprocessing, complete backend time, visualization, and
  end-to-end latency.
- Explicitly records host-to-device time as unavailable separately because it
  is included inside Ultralytics preprocessing.
- Added mean, median, linearly interpolated P95, minimum, and maximum.
- Added JSON summary/raw reports and raw CSV serialization.
- Added environment, input, iteration, precision, allocation-boundary, and
  peak CUDA-memory metadata.
- Added a `kernelvision benchmark` CLI command.
- Added a Modal 1.x app that requests one NVIDIA L4 and returns the same report
  schema.
- Validated the harness with a three-iteration local CPU diagnostic; its
  measurements were deliberately written to `/tmp` and are not project
  performance results.
- `pytest -q` passes 24 tests locally.

### Resolved remote validation issues

- The active Modal profile was present, but both configured Modal API endpoints
  initially returned HTTP 503 from the managed Codex execution environment.
- Isolated the connection failure to Modal client 1.5.3. Existing CUDA lab
  environments using Modal 1.2.6 resolved the same profile immediately, so the
  project remote dependency was pinned to 1.2.6.
- The first L4 container reached remote execution but OpenCV could not load
  `libGL.so.1` from Debian Slim. Added the minimal `libgl1` and `libglib2.0-0`
  runtime libraries to the Modal image.

### Modal L4 baseline result

- Completed 30 warm-up and 200 measured iterations on an NVIDIA L4.
- PyTorch 2.13.0 with CUDA 13.0, Ultralytics 8.4.115, YOLOv8n, batch size 1,
  and 640 × 640 model input.
- End-to-end latency: 18.497 ms median and 20.380 ms P95.
- Model inference latency: 7.431 ms median.
- Mean-latency-derived throughput: 53.469 FPS.
- Peak allocated GPU memory: 27.812 MB.
- End-to-end coefficient of variation: 4.87%.
- Saved 200 raw samples to `benchmarks/raw/modal_l4_baseline.csv` and the full
  report to `results/modal_l4_baseline.json`.

## 2026-08-04 — Milestone 3 FP32 versus FP16

### Completed

- Added explicit `fp32` and `fp16` precision configuration across the CLI,
  image/video pipeline, benchmark runner, and Ultralytics backend.
- Mapped FP32 to Ultralytics `quantize=32` and FP16 to `quantize=16`.
- Implemented box IoU and greedy one-to-one same-class detection matching.
- Added conversion from Ultralytics tensors into device-independent detection
  records and JSON comparison summaries.
- Added Modal L4 correctness and order-reversed paired performance workflows.
- Extended environment reports with NumPy and OpenCV versions.
- `pytest -q` passes 33 tests locally.

### Correctness result

- YOLOv8n produced six FP32 and six FP16 detections on `bus.jpg`.
- All six detections matched; there were no unmatched outputs.
- Mean matched IoU: 0.998210; minimum matched IoU: 0.997358.
- Maximum coordinate difference: 0.467 px.
- Maximum confidence difference: 0.000737.
- This is single-image numerical consistency evidence, not dataset-wide mAP.

### Paired Modal L4 performance result

- Used one pinned container with PyTorch 2.13.0, CUDA 13.0, Ultralytics
  8.4.115, NumPy 2.4.6, and OpenCV 5.0.0.93.
- Ran 30 warm-ups and 200 measured iterations for each precision in both
  FP32→FP16 and FP16→FP32 order.
- FP16 inference median improved by 0.63% and 3.74% across the two trials.
- FP16 end-to-end median improved by 6.61% and 7.30%.
- Mean-based throughput improved by 5.27% and 6.32%.
- Peak allocated GPU memory fell from 27.812 MB to 13.981 MB, a 49.73%
  reduction.

### Benchmarking lesson

The original Milestone 2 FP32 report was collected a day earlier and did not
record all CPU-side dependency versions. Comparing it directly with the first
FP16 run incorrectly suggested a much larger speedup. Regenerating the control
and then running both precisions in one order-reversed container showed that
the model-only FP16 gain is real but modest. Historical Milestone 2 results
remain preserved as the original baseline rather than being silently replaced.

## 2026-08-04 — Milestone 4 fused Triton preprocessing

### Completed

- Defined a model-preparation contract from contiguous CUDA `uint8` BGR-HWC
  input to normalized RGB-CHW FP32 or FP16 output.
- Added a validated PyTorch correctness reference.
- Implemented a one-pixel-per-lane Triton kernel that fuses BGR-to-RGB,
  HWC-to-CHW, normalization, and dtype conversion.
- Added a validated Python launch wrapper with configurable block size and warp
  count.
- Added Modal L4 correctness and CUDA-event microbenchmark workflows.
- Added Triton version reporting to reproducibility metadata.
- `pytest -q` passes 44 portable tests locally.

### Correctness result

- Tested five shapes in FP32 and FP16, including 5×7 and 641×639 boundary-mask
  cases.
- All 10 output tensors matched the PyTorch reference.
- Maximum absolute difference: 0.0.
- Mismatched values: 0.

### Performance result

- Used 30 warm-ups and 200 CUDA-event measurements per operation.
- Tested four image shapes, two output dtypes, four block sizes, and three warp
  counts on a Modal NVIDIA L4.
- Triton speedups ranged from 1.54× to 2.56× across the eight cases.
- At 640×640, FP32 improved from 0.0860 ms to 0.0543 ms (1.58×), while FP16
  improved from 0.0850 ms to approximately 0.054–0.055 ms (about 1.54–1.57×).
- The absolute 640×640 saving was about 0.032 ms.
- Parameter differences were generally around one microsecond; no universal
  best configuration emerged, so 256 elements and four warps remain default.

### Limitation

This is a GPU-only component microbenchmark. Input tensors already reside on
the GPU, and resize, letterbox, host-to-device transfer, Python wall time, model
inference, and postprocessing are excluded. Pipeline integration must be
benchmarked before making end-to-end speed claims.

## 2026-08-05 — Milestone 5 standalone CUDA preprocessing started

### Scope established

- Kept raw CUDA compilation and execution in a standalone executable so the
  PyTorch C++/CUDA extension boundary remains Milestone 6.
- Added a file-based deterministic input/output protocol shared with the
  trusted PyTorch reference.
- Added Modal NVIDIA L4 compilation and correctness orchestration for the same
  five shapes and two dtypes used in Milestone 4.
- Added native CUDA error checking, preallocated device buffers, and CUDA-event
  sampling plumbing around the naive kernel exercise.
- Added portable tests for deterministic fixture generation and FP32/FP16 raw
  output decoding.

### Learning checkpoint

The learner implemented `naive_bgr_hwc_to_rgb_chw()` as one thread per pixel,
with interleaved BGR loads, normalized planar RGB stores, FP32/FP16 dispatch,
and a final bounds check.

### Naive CUDA correctness result

- Compiled with CUDA 13.0 and nvcc 13.0.48 on a Modal NVIDIA L4.
- Passed five shapes in FP32 and FP16: 10/10 cases.
- Maximum absolute difference: 0.0.
- Mismatched values: 0.
- Saved `results/modal_l4_cuda_preprocess_correctness.json`.
- No benchmark claim has been made; optimization and controlled performance
  comparison remain unfinished.
