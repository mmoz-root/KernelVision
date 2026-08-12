"""Portable tests for the lazy PyTorch CUDA-extension wrapper."""

from __future__ import annotations

import pytest

from kernelvision.preprocessing import cuda_extension_preprocess


torch = pytest.importorskip("torch")


def test_cuda_extension_wrapper_requires_tensor() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        cuda_extension_preprocess(
            object(),
            output_dtype=torch.float32,
        )


def test_cuda_extension_wrapper_rejects_wrong_shape() -> None:
    image = torch.zeros((2, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="HWC shape"):
        cuda_extension_preprocess(
            image,
            output_dtype=torch.float32,
        )


def test_cuda_extension_wrapper_rejects_wrong_dtype() -> None:
    image = torch.zeros((2, 3, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="torch.uint8"):
        cuda_extension_preprocess(
            image,
            output_dtype=torch.float32,
        )


def test_cuda_extension_wrapper_rejects_output_dtype() -> None:
    image = torch.zeros((2, 3, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="output_dtype"):
        cuda_extension_preprocess(
            image,
            output_dtype=torch.uint8,
        )


def test_cuda_extension_wrapper_requires_cuda_before_loading() -> None:
    image = torch.zeros((2, 3, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="CUDA device"):
        cuda_extension_preprocess(
            image,
            output_dtype=torch.float32,
        )
