"""Portable validation tests for the Triton preprocessing wrapper."""

import pytest

torch = pytest.importorskip("torch")

from kernelvision.preprocessing import (
    triton_preprocess,
    triton_preprocess_into,
)


def test_triton_wrapper_requires_cuda_input() -> None:
    image = torch.zeros((2, 3, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="CUDA"):
        triton_preprocess(image, output_dtype=torch.float32)


@pytest.mark.parametrize("block_size", [0, 100, 255])
def test_triton_wrapper_rejects_invalid_block_size(block_size: int) -> None:
    image = torch.zeros((2, 3, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="power of two"):
        triton_preprocess(
            image,
            output_dtype=torch.float32,
            block_size=block_size,
        )


def test_triton_into_requires_tensor_output() -> None:
    image = torch.zeros((2, 3, 3), dtype=torch.uint8)

    with pytest.raises(TypeError, match="output must be a torch.Tensor"):
        triton_preprocess_into(image, object())


def test_triton_into_requires_supported_output_dtype() -> None:
    image = torch.zeros((2, 3, 3), dtype=torch.uint8)
    output = torch.zeros((3, 2, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="output_dtype"):
        triton_preprocess_into(image, output)
