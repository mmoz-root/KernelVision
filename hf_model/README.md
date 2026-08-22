---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
base_model: Ultralytics/YOLOv8
tags:
  - yolov8
  - onnx
  - fp16
  - tensorrt
  - gpu-optimization
  - computer-vision
---

# KernelVision YOLOv8n ONNX deployment graphs

This repository publishes the two validated ONNX graphs produced during
[KernelVision](https://github.com/mmoz-root/KernelVision), a correctness-first
study of GPU optimization in a YOLOv8n object-detection pipeline.

KernelVision did **not** train a new detector. Both graphs derive from the
official COCO-pretrained Ultralytics YOLOv8n model. They are published as
reproducible deployment artifacts so the project is not presented as owning a
new set of learned weights.

## Files

| File | Graph precision | Input | Raw output | Size |
|---|---|---|---|---:|
| `yolov8n_fp32.onnx` | FP32 | FP32 `[1, 3, 640, 640]` | FP32 `[1, 84, 8400]` | 12.2 MiB |
| `yolov8n_mixed_fp16.onnx` | ModelOpt mixed FP16/FP32 | FP16 `[1, 3, 640, 640]` | FP16 `[1, 84, 8400]` | 6.2 MiB |

Both graphs use ONNX opset 17 and a static batch size and image shape. The raw
output contains pre-NMS box and class predictions; confidence filtering,
non-maximum suppression (NMS), and coordinate scaling are not embedded in the
graphs.

## Provenance

```text
Ultralytics YOLOv8n PyTorch checkpoint
        ↓ export, opset 17
yolov8n_fp32.onnx
        ↓ NVIDIA ModelOpt AutoCast
yolov8n_mixed_fp16.onnx
        ↓ TensorRT 11.2.1.2 build on NVIDIA L4
TensorRT FP16 engine (not distributed)
```

The TensorRT engines are intentionally excluded because serialized engines are
tied to their build environment and target GPU stack.

## Correctness evidence

The FP32 ONNX graph was compared with the PyTorch reference over all `705,600`
raw output elements:

| Check | Result |
|---|---:|
| Shape | `[1, 84, 8400]` |
| All-close (`atol=1e-4`, `rtol=1e-4`) | Pass |
| Mean absolute difference | `2.981e-06` |
| P99 absolute difference | `6.866e-05` |
| Maximum absolute difference | `0.004059` |

The mixed graph was used to build the TensorRT FP16 candidate. Against the
PyTorch FP32 reference on the final application-level case, it produced five
matched detections with no unmatched boxes, mean matched IoU `0.997469`, and a
maximum coordinate difference of `0.546 px`.

## NVIDIA L4 benchmark context

These are results from the KernelVision benchmark harness, not hosted Hugging
Face inference measurements:

| Backend | Model-only median | Complete-pipeline median |
|---|---:|---:|
| PyTorch FP32 | `9.529 ms` | `13.600 ms` |
| TensorRT FP16 | `1.428 ms` | `9.664 ms` |
| Median speedup | `6.671×` | `1.407×` |

The complete pipeline includes image decode, preprocessing, host-to-device
transfer, model execution, NMS, and in-memory visualization. It excludes model
loading and output-file writing.

## Minimal ONNX Runtime example

The caller must supply an already letterboxed, RGB, normalized BCHW tensor:

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("yolov8n_fp32.onnx")
input_tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)
raw_output = session.run(["output0"], {"images": input_tensor})[0]
print(raw_output.shape)  # (1, 84, 8400)
```

For actual detections, apply the same letterbox metadata, confidence filtering,
NMS, and coordinate scaling described in the KernelVision source repository.

## Artifact integrity

| File | SHA-256 |
|---|---|
| `yolov8n_fp32.onnx` | `3db80127ae56dae98402da2b3bd11ef2214a61c1f30e69166148087e43f3bc5e` |
| `yolov8n_mixed_fp16.onnx` | `5d319e1d30f30d6e64bcfab3ca86a4d4844b1d8f740f0495d4d211fc0b069759` |

## Limitations

- The graphs accept only batch 1 at `640 × 640`.
- The model is the general COCO-pretrained YOLOv8n detector, not a
  KernelVision-trained model.
- Dataset-wide COCO mAP was not re-evaluated for these exports.
- The FP16 graph is a mixed-precision deployment input for TensorRT; reduced
  precision does not imply identical raw floating-point values.
- The reported performance covers one Modal-hosted NVIDIA L4 and should not be
  generalized to other hardware without measurement.

## Reproduction and license

- [Source, scripts, and raw benchmark records](https://github.com/mmoz-root/KernelVision)
- [Complete final report](https://github.com/mmoz-root/KernelVision/blob/main/docs/final_report.md)
- [Ultralytics YOLOv8 base model](https://huggingface.co/Ultralytics/YOLOv8)

The artifacts are distributed under AGPL-3.0 in accordance with the upstream
Ultralytics licensing terms. See the
[KernelVision third-party notices](THIRD_PARTY_NOTICES.md) for complete
attribution.
