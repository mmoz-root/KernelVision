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
