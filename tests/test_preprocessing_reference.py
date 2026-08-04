"""Tests for the trusted PyTorch preprocessing reference."""

import pytest

torch = pytest.importorskip("torch")

from kernelvision.preprocessing import torch_reference_preprocess


def test_reference_converts_bgr_hwc_to_normalized_rgb_chw() -> None:
    image = torch.tensor(
        [
            [[0, 10, 255], [30, 40, 50]],
            [[60, 70, 80], [90, 100, 110]],
        ],
        dtype=torch.uint8,
    )
    original = image.clone()
    expected = torch.tensor(
        [
            [[255, 50], [80, 110]],
            [[10, 40], [70, 100]],
            [[0, 30], [60, 90]],
        ],
        dtype=torch.float32,
    ) / 255.0

    output = torch_reference_preprocess(image, output_dtype=torch.float32)

    assert output.shape == (3, 2, 2)
    assert output.dtype == torch.float32
    assert output.device == image.device
    assert output.is_contiguous()
    assert torch.equal(image, original)
    assert torch.equal(output, expected)


def test_reference_supports_fp16_and_irregular_dimensions() -> None:
    image = torch.arange(5 * 7 * 3, dtype=torch.uint8).reshape(5, 7, 3)

    output = torch_reference_preprocess(image, output_dtype=torch.float16)

    assert output.shape == (3, 5, 7)
    assert output.dtype == torch.float16
    assert output.is_contiguous()
    assert output[0, 0, 0].item() == pytest.approx(
        image[0, 0, 2].item() / 255.0,
        abs=5e-4,
    )


@pytest.mark.parametrize(
    "image",
    [
        torch.zeros((2, 2, 3), dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.uint8),
        torch.zeros((2, 2, 4), dtype=torch.uint8),
    ],
)
def test_reference_rejects_invalid_inputs(image: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        torch_reference_preprocess(image, output_dtype=torch.float32)


def test_reference_rejects_non_floating_output_dtype() -> None:
    image = torch.zeros((2, 2, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="output_dtype"):
        torch_reference_preprocess(image, output_dtype=torch.int32)
