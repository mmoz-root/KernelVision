"""Portable tests for the custom Ultralytics preprocessing override."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from kernelvision.backends.cuda_extension_predictor import (
    get_cuda_extension_predictor_class,
)


def test_predictor_batches_extension_outputs(
    monkeypatch: Any,
) -> None:
    calls = []

    def fake_preprocess(image: Any, *, output_dtype: Any) -> Any:
        calls.append((image, output_dtype))
        return image[..., [2, 1, 0]].permute(2, 0, 1).to(output_dtype) / 255

    monkeypatch.setattr(
        "kernelvision.preprocessing.cuda_extension_preprocess",
        fake_preprocess,
    )
    get_cuda_extension_predictor_class.cache_clear()
    predictor_class = get_cuda_extension_predictor_class()
    predictor = object.__new__(predictor_class)
    predictor.pre_transform = lambda images: images
    predictor.model = SimpleNamespace(fp16=False)
    predictor.device = torch.device("cpu")
    images = [
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.full((2, 3, 3), 255, dtype=np.uint8),
    ]

    output = predictor.preprocess(images)

    assert output.shape == (2, 3, 2, 3)
    assert output.dtype == torch.float32
    assert len(calls) == 2
    assert all(call[0].is_contiguous() for call in calls)
    assert all(call[1] == torch.float32 for call in calls)
    get_cuda_extension_predictor_class.cache_clear()


def test_predictor_selects_fp16_from_model(
    monkeypatch: Any,
) -> None:
    selected_dtypes = []

    def fake_preprocess(image: Any, *, output_dtype: Any) -> Any:
        selected_dtypes.append(output_dtype)
        return image.permute(2, 0, 1).to(output_dtype)

    monkeypatch.setattr(
        "kernelvision.preprocessing.cuda_extension_preprocess",
        fake_preprocess,
    )
    get_cuda_extension_predictor_class.cache_clear()
    predictor_class = get_cuda_extension_predictor_class()
    predictor = object.__new__(predictor_class)
    predictor.pre_transform = lambda images: images
    predictor.model = SimpleNamespace(fp16=True)
    predictor.device = torch.device("cpu")

    output = predictor.preprocess(
        [np.zeros((2, 3, 3), dtype=np.uint8)]
    )

    assert output.dtype == torch.float16
    assert selected_dtypes == [torch.float16]
    get_cuda_extension_predictor_class.cache_clear()
