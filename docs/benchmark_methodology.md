# Benchmark Methodology

## Baseline scope

The Milestone 2 harness benchmarks batch-size-1 Ultralytics image inference.
It records every measured iteration before calculating summary statistics.

The baseline uses:

- PyTorch/Ultralytics FP32 inference
- Configurable model input size (default: 640)
- Configurable warm-up count (default: 30)
- Configurable measured count (default: 200)
- A fixed confidence threshold (default: 0.25)

## Timing boundaries

The report contains the following millisecond measurements:

- `decode_ms`: OpenCV image decode from disk.
- `preprocess_ms`: Ultralytics preprocessing, including resize/letterbox,
  conversion, normalization, and host-to-device transfer when using CUDA.
- `host_to_device_ms`: `null` for this baseline because Ultralytics does not
  expose transfer separately from preprocessing.
- `inference_ms`: Ultralytics model-forward timing.
- `postprocess_ms`: Ultralytics prediction decode, filtering, and NMS timing.
- `backend_ms`: synchronized wall time around the complete backend call.
- `visualization_ms`: in-memory annotation drawing via `Results.plot()`.
- `end_to_end_ms`: decode, backend call, and visualization. Model loading and
  output-file writing are excluded.

Ultralytics uses accelerator synchronization around its internal stage timers.
KernelVision also synchronizes before and after timed accelerator-backed
regions. CPU operations do not synchronize.

## Warm-up and allocation

Model construction happens before warm-up. Warm-up iterations exercise both
prediction and visualization and are not included in raw samples. The current
baseline includes steady-state tensor allocation. On CUDA, peak allocated GPU
memory is reset after warm-up and recorded after measured iterations.

## Statistics

For every latency metric, KernelVision reports:

- Count
- Mean
- Median
- P95 using linear interpolation
- Minimum
- Maximum

Throughput is derived as `1000 / mean_end_to_end_ms`. Raw samples are saved to
CSV and are also embedded in the JSON report.

## Result interpretation

Local CPU runs validate the harness only. Publishable GPU results must come
from the Modal NVIDIA L4 workflow with 30 warm-up and 200 measured iterations.
The report records the exact GPU, Python, PyTorch, CUDA, Torchvision,
Ultralytics, input shape, model input size, batch size, precision, and timing
boundaries.

