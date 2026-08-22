# Third-party notices

KernelVision is an independent learning and benchmarking project. It is not
affiliated with or endorsed by Ultralytics, NVIDIA, Modal, PyTorch, or OpenAI.

## Ultralytics YOLOv8

This project uses the Ultralytics Python package and the pretrained YOLOv8n
model during reproduction and benchmarking. Ultralytics distributes its
open-source software and model artifacts under the GNU Affero General Public
License v3.0 (AGPL-3.0), subject to its published licensing terms:

- <https://github.com/ultralytics/ultralytics>
- <https://www.ultralytics.com/license>

The Ultralytics Python package, original pretrained PyTorch checkpoint, and
serialized TensorRT engines are not redistributed by KernelVision.
Reproduction scripts obtain or consume them separately. The companion
[Hugging Face model repository](https://huggingface.co/mmoz-root/kernelvision)
publishes two derived ONNX graphs under AGPL-3.0 with explicit provenance and
limitations.

## Ultralytics sample image

The final annotated bus demonstration is derived from `bus.jpg`, accessed via
the sample assets bundled with Ultralytics. The source asset repository is:

- <https://github.com/ultralytics/assets>

The annotated result is included only to document the benchmark output and
remains subject to applicable upstream terms.

## Other dependencies

KernelVision also depends on projects including PyTorch, Triton, ONNX,
ONNX Runtime, TensorRT, OpenCV, NumPy, Matplotlib, and Modal. Each dependency
is governed by its own license. Package version constraints are recorded in
`pyproject.toml`, while exact benchmark versions are recorded in the result
reports.

NVIDIA, CUDA, TensorRT, and related names are trademarks or registered
trademarks of NVIDIA Corporation. Other names may be trademarks of their
respective owners.
