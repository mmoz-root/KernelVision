"""Timing primitives for CPU and accelerator-backed operations."""

from __future__ import annotations


def synchronize_device(device: str) -> None:
    """Wait for outstanding work on the configured accelerator, if needed."""
    str_device = device.strip().lower()

    if str_device == "cpu":
        return

    import torch

    if str_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("mps unavailable")
        torch.mps.synchronize()
        return

    is_cuda = (
        str_device == "cuda"
        or str_device.startswith("cuda:")
        or str_device.isdigit()
    )

    if is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("cuda unavailable")
        torch.cuda.synchronize()
        return

    raise ValueError(f"unsupported device: {device}")
