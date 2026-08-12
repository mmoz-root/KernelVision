"""Ultralytics predictor that delegates fused preprocessing to CUDA."""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def get_cuda_extension_predictor_class() -> type[Any]:
    """Create the optional Ultralytics subclass only when requested."""
    try:
        import torch
        from ultralytics.models.yolo.detect import DetectionPredictor
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Ultralytics and PyTorch are required for CUDA-extension "
            "pipeline integration."
        ) from error

    from kernelvision.preprocessing import cuda_extension_preprocess

    class CudaExtensionDetectionPredictor(DetectionPredictor):
        """Replace only Ultralytics tensor formatting and normalization."""

        def preprocess(self, images: Any) -> Any:
            """Return normalized BCHW input using the CUDA extension."""
            if isinstance(images, torch.Tensor):
                return super().preprocess(images)

            # Milestone 6 integration exercise:
            # 1. Call self.pre_transform(images) to retain letterboxing.
            # 2. Convert each resulting NumPy BGR-HWC image to contiguous form.
            # 3. Wrap it with torch.from_numpy and move uint8 data to self.device.
            # 4. Select FP16 when self.model.fp16 is true, otherwise FP32.
            # 5. Call cuda_extension_preprocess for each image.
            # 6. Add a batch dimension for one image, or stack multiple outputs.
            import numpy as np

            transformed_images = self.pre_transform(images)
            output_dtype = (
                torch.float16 if self.model.fp16 else torch.float32
            )
            processed_images = []
            for image in transformed_images:
                cont_image = np.ascontiguousarray(image)
                cuda_image = torch.from_numpy(cont_image).to(device=self.device)
                processed_image = cuda_extension_preprocess(
                    cuda_image,
                    output_dtype=output_dtype,
                )
                processed_images.append(processed_image)
            return torch.stack(processed_images, dim=0)

    return CudaExtensionDetectionPredictor
