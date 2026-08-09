"""Triton implementation of fused BGR-HWC preprocessing."""

from __future__ import annotations

from typing import Any

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _bgr_hwc_to_rgb_chw_kernel(
        input_pointer,
        output_pointer,
        pixel_count,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Process one image pixel per Triton program lane."""
        pixel_offsets = (
            tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
        active_pixels = pixel_offsets < pixel_count
        input_offsets = pixel_offsets * 3

        blue = tl.load(
            input_pointer + input_offsets,
            mask=active_pixels,
            other=0.0
        )
        green = tl.load(
            input_pointer + input_offsets + 1,
            mask=active_pixels,
            other=0.0
        )
        red = tl.load(
            input_pointer + input_offsets + 2,
            mask=active_pixels,
            other=0.0
        )

        scale = 1.0/255.0
        red = red * scale
        green = green * scale
        blue = blue * scale

        tl.store(
            output_pointer + pixel_offsets,
            red,
            mask=active_pixels
        )
        tl.store(
            output_pointer + pixel_count + pixel_offsets,
            green,
            mask=active_pixels
        )
        tl.store(
            output_pointer + 2*pixel_count + pixel_offsets,
            blue,
            mask=active_pixels
        )

        # output_pointer
        # │
        # ├─ [0 × pixels ... 1 × pixels) → Red
        # ├─ [1 × pixels ... 2 × pixels) → Green
        # └─ [2 × pixels ... 3 × pixels) → Blue

else:
    _bgr_hwc_to_rgb_chw_kernel = None


def _load_torch() -> Any:
    """Import PyTorch lazily so portable package imports still work."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyTorch is required for Triton preprocessing. "
            "Install KernelVision with the 'inference' extra."
        ) from error
    return torch


def _validate_triton_launch(
    image: Any,
    *,
    output_dtype: Any,
    block_size: int,
    num_warps: int,
    torch: Any,
) -> None:
    """Validate properties shared by allocating and preallocated launches."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must have HWC shape [height, width, 3]")
    if image.dtype != torch.uint8:
        raise ValueError("image must have torch.uint8 dtype")
    if output_dtype not in (torch.float32, torch.float16):
        raise ValueError("output_dtype must be torch.float32 or torch.float16")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")
    if not image.is_cuda:
        raise ValueError("image must be on a CUDA device")
    if not image.is_contiguous():
        raise ValueError("image must be contiguous")
    if triton is None or _bgr_hwc_to_rgb_chw_kernel is None:
        raise RuntimeError("Triton is not installed in this environment")


def _launch_triton_into(
    image: Any,
    output: Any,
    *,
    block_size: int,
    num_warps: int,
) -> Any:
    """Launch the Triton kernel into an already validated output tensor."""
    height, width, _ = image.shape
    pixel_count = height * width
    grid = (triton.cdiv(pixel_count, block_size),)
    _bgr_hwc_to_rgb_chw_kernel[grid](
        image,
        output,
        pixel_count,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


def triton_preprocess(
    image: Any,
    *,
    output_dtype: Any,
    block_size: int = 256,
    num_warps: int = 4,
) -> Any:
    """Allocate output and launch fused RGB-CHW preprocessing."""
    torch = _load_torch()
    _validate_triton_launch(
        image,
        output_dtype=output_dtype,
        block_size=block_size,
        num_warps=num_warps,
        torch=torch,
    )
    height, width, _ = image.shape
    output = torch.empty(
        (3, height, width),
        dtype=output_dtype,
        device=image.device,
    )
    return _launch_triton_into(
        image,
        output,
        block_size=block_size,
        num_warps=num_warps,
    )


def triton_preprocess_into(
    image: Any,
    output: Any,
    *,
    block_size: int = 256,
    num_warps: int = 4,
) -> Any:
    """Launch fused preprocessing into a preallocated output tensor."""
    torch = _load_torch()
    if not isinstance(output, torch.Tensor):
        raise TypeError("output must be a torch.Tensor")
    _validate_triton_launch(
        image,
        output_dtype=output.dtype,
        block_size=block_size,
        num_warps=num_warps,
        torch=torch,
    )
    height, width, _ = image.shape
    if output.shape != (3, height, width):
        raise ValueError("output must have CHW shape [3, height, width]")
    if output.device != image.device:
        raise ValueError("output must be on the same CUDA device as image")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    return _launch_triton_into(
        image,
        output,
        block_size=block_size,
        num_warps=num_warps,
    )
