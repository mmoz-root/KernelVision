from __future__ import annotations

import struct

import pytest

from kernelvision.preprocessing import (
    deterministic_bgr_image,
    read_standalone_output,
)


def test_deterministic_bgr_image_uses_documented_byte_formula() -> None:
    torch = pytest.importorskip("torch")

    image = deterministic_bgr_image(
        1,
        3,
        torch_module=torch,
    )

    assert image.shape == (1, 3, 3)
    assert image.dtype == torch.uint8
    assert image.is_contiguous()
    assert image.reshape(-1).tolist() == [
        13,
        50,
        87,
        124,
        161,
        198,
        235,
        16,
        53,
    ]


def test_deterministic_bgr_image_rejects_nonpositive_shape() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="height and width must be positive"):
        deterministic_bgr_image(0, 4, torch_module=torch)


@pytest.mark.parametrize(
    ("dtype_name", "format_character", "values"),
    (
        ("fp32", "f", (0.0, 0.25, 0.5, 0.75, 1.0, 0.125)),
        ("fp16", "e", (0.0, 0.25, 0.5, 0.75, 1.0, 0.125)),
    ),
)
def test_read_standalone_output_restores_chw_tensor(
    tmp_path,
    dtype_name: str,
    format_character: str,
    values: tuple[float, ...],
) -> None:
    torch = pytest.importorskip("torch")
    output_path = tmp_path / "output.bin"
    output_path.write_bytes(
        struct.pack("<" + format_character * len(values), *values)
    )

    output = read_standalone_output(
        output_path,
        height=1,
        width=2,
        dtype_name=dtype_name,
        torch_module=torch,
    )

    expected_dtype = torch.float32 if dtype_name == "fp32" else torch.float16
    assert output.shape == (3, 1, 2)
    assert output.dtype == expected_dtype
    assert output.is_contiguous()
    assert output.reshape(-1).tolist() == list(values)


def test_read_standalone_output_rejects_wrong_byte_count(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    output_path = tmp_path / "short.bin"
    output_path.write_bytes(b"too short")

    with pytest.raises(ValueError, match="raw output has 9 bytes; expected 24"):
        read_standalone_output(
            output_path,
            height=1,
            width=2,
            dtype_name="fp32",
            torch_module=torch,
        )
