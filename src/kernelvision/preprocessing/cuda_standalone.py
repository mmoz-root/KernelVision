"""Shared host-side helpers for standalone CUDA preprocessing experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any


_DETERMINISTIC_MULTIPLIER = 37
_DETERMINISTIC_OFFSET = 13


def deterministic_bgr_image(
    height: int,
    width: int,
    *,
    torch_module: Any,
    device: str = "cpu",
) -> Any:
    """Create the cross-language deterministic uint8 BGR-HWC test image.

    Byte ``i`` is ``(37 * i + 13) % 256``. Python writes these exact bytes
    to the fixture consumed by the CUDA executable, so no random-number-
    generator behavior crosses the Python/C++ process boundary.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    byte_count = height * width * 3
    offsets = torch_module.arange(
        byte_count,
        dtype=torch_module.int64,
        device=device,
    )
    values = (
        offsets * _DETERMINISTIC_MULTIPLIER + _DETERMINISTIC_OFFSET
    ) % 256
    return values.to(dtype=torch_module.uint8).reshape(height, width, 3)


def read_standalone_output(
    path: str | Path,
    *,
    height: int,
    width: int,
    dtype_name: str,
    torch_module: Any,
) -> Any:
    """Read a standalone CUDA raw output as a contiguous CPU CHW tensor."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    dtype_by_name = {
        "fp32": torch_module.float32,
        "fp16": torch_module.float16,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError("dtype_name must be 'fp32' or 'fp16'")

    data = bytearray(Path(path).read_bytes())
    dtype = dtype_by_name[dtype_name]
    element_bytes = 4 if dtype_name == "fp32" else 2
    expected_bytes = 3 * height * width * element_bytes
    if len(data) != expected_bytes:
        raise ValueError(
            f"raw output has {len(data)} bytes; expected {expected_bytes}"
        )

    return torch_module.frombuffer(data, dtype=dtype).clone().reshape(
        3,
        height,
        width,
    )
