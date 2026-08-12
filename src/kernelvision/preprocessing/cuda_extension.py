from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def _load_cuda_extension() -> Any:
    try:
        from torch.utils.cpp_extension import load
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyTorch is required for CUDA extension preprocessing."
        ) from error

    project_root = Path(__file__).resolve().parents[3]

    return load(
        name="kernelvision_cuda_extension",
        sources=[
            str(project_root / "csrc/preprocessing/torch_extension.cpp"),
            str(project_root / "csrc/preprocessing/torch_extension_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=False,
    )


def cuda_extension_preprocess(
    image: Any,
    *,
    output_dtype: Any,
) -> Any:
    """Run fused preprocessing through the PyTorch CUDA extension."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyTorch is required for CUDA extension preprocessing."
        ) from error
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must have HWC shape [height, width, 3]")

    if image.dtype != torch.uint8:
        raise ValueError("image must have torch.uint8 dtype")

    if output_dtype not in (torch.float32, torch.float16):
        raise ValueError("output_dtype must be torch.float32 or torch.float16")

    if not image.is_cuda:
        raise ValueError("image must be on a CUDA device")

    if not image.is_contiguous():
        raise ValueError("image must be contiguous")

    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image height and width must be positive")

    output_fp16 = output_dtype == torch.float16
    extension = _load_cuda_extension()
    return extension.preprocess(image, output_fp16)
