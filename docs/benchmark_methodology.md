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

## FP32 versus FP16 experiment

Precision comparisons use `quantize=32` and `quantize=16` in Ultralytics
8.4.115. Separate model backend instances are required because Ultralytics
initializes its prediction backend precision on first use.

Correctness is checked before performance. FP32 detections are matched
one-to-one with FP16 detections of the same class, selecting candidate pairs
from highest to lowest IoU. The report records detection and match counts,
unmatched counts, mean and minimum matched IoU, maximum coordinate difference,
and maximum confidence difference.

The final performance comparison runs both precisions in one Modal L4
container. Each precision receives 30 warm-up and 200 measured iterations.
Two trials reverse execution order—FP32 then FP16, followed by FP16 then
FP32—to reveal simple order effects. All four reports retain their 200 raw
samples.

Model-only inference changes are the clearest evidence of precision effects.
End-to-end measurements additionally include image decode, preprocessing,
postprocessing, visualization, and normal CPU/runtime variation. A
single-image correctness comparison does not replace dataset-wide quality
evaluation such as mAP.
