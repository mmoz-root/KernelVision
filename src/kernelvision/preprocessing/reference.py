"""Trusted PyTorch reference for the fused preprocessing operation."""

from __future__ import annotations

from typing import Any


def torch_reference_preprocess(
    image: Any,
    *,
    output_dtype: Any,
) -> Any:
    """Convert one uint8 BGR HWC image into normalized RGB CHW output.

    The output remains on the input tensor's device, is contiguous, and has
    either torch.float32 or torch.float16 dtype. Resize and letterbox are
    intentionally outside this operation.
    """
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyTorch is required for preprocessing. "
            "Install KernelVision with the 'inference' extra."
        ) from error

    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must have HWC shape [height, width, 3]")
    if image.dtype != torch.uint8:
        raise ValueError("image must have torch.uint8 dtype")
    if output_dtype not in (torch.float32, torch.float16):
        raise ValueError("output_dtype must be torch.float32 or torch.float16")

    rgb = image.flip(-1)

    chw = rgb.permute(2, 0, 1)

    converted = chw.to(dtype=output_dtype)

    return (converted / 255.0).contiguous()
