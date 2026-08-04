"""Tests for FP32-versus-FP16 numerical comparison helpers."""

import pytest

from kernelvision.precision_comparison import (
    BoundingBox,
    Detection,
    compare_precision_results,
    extract_detections,
    intersection_over_union,
    match_detections,
)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (BoundingBox(0, 0, 2, 2), BoundingBox(0, 0, 2, 2), 1.0),
        (BoundingBox(0, 0, 1, 1), BoundingBox(2, 2, 3, 3), 0.0),
        (BoundingBox(0, 0, 2, 2), BoundingBox(1, 1, 3, 3), 1.0 / 7.0),
    ],
)
def test_intersection_over_union(
    first: BoundingBox,
    second: BoundingBox,
    expected: float,
) -> None:
    assert intersection_over_union(first, second) == pytest.approx(expected)


def test_intersection_over_union_returns_zero_for_two_empty_boxes() -> None:
    empty = BoundingBox(1, 1, 1, 1)

    assert intersection_over_union(empty, empty) == 0.0


def _detection(
    coordinates: tuple[float, float, float, float],
    *,
    class_id: int,
) -> Detection:
    return Detection(BoundingBox(*coordinates), confidence=0.8, class_id=class_id)


def test_match_detections_uses_class_and_not_list_order() -> None:
    person = _detection((0, 0, 2, 2), class_id=0)
    car = _detection((10, 10, 12, 12), class_id=2)
    fp16_car = _detection((10, 10, 12, 12), class_id=2)
    fp16_person = _detection((0, 0, 2, 2), class_id=0)

    matches = match_detections([person, car], [fp16_car, fp16_person])

    assert [(match.fp32, match.fp16) for match in matches] == [
        (person, fp16_person),
        (car, fp16_car),
    ]
    assert all(match.iou == 1.0 for match in matches)


def test_match_detections_does_not_reuse_a_detection() -> None:
    exact = _detection((0, 0, 2, 2), class_id=0)
    overlapping = _detection((0, 0, 3, 3), class_id=0)
    only_fp16_detection = _detection((0, 0, 2, 2), class_id=0)

    matches = match_detections([exact, overlapping], [only_fp16_detection])

    assert len(matches) == 1
    assert matches[0].fp32 == exact
    assert matches[0].fp16 == only_fp16_detection


def test_match_detections_rejects_different_classes_and_low_iou() -> None:
    reference = _detection((0, 0, 2, 2), class_id=0)
    wrong_class = _detection((0, 0, 2, 2), class_id=1)
    far_away = _detection((10, 10, 12, 12), class_id=0)

    assert match_detections([reference], [wrong_class, far_away]) == ()


class FakeTensor:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list[object]:
        return self.values


class FakeBoxes:
    def __init__(
        self,
        xyxy: list[object],
        confidence: list[object],
        classes: list[object],
    ) -> None:
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(confidence)
        self.cls = FakeTensor(classes)


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None) -> None:
        self.boxes = boxes


def test_extract_detections_converts_tensor_values() -> None:
    result = FakeResult(FakeBoxes([[1.0, 2.0, 3.0, 4.0]], [0.75], [2.0]))

    assert extract_detections(result) == (
        Detection(BoundingBox(1.0, 2.0, 3.0, 4.0), 0.75, 2),
    )


def test_compare_precision_results_summarizes_matched_differences() -> None:
    fp32 = FakeResult(FakeBoxes([[0.0, 0.0, 10.0, 10.0]], [0.90], [0.0]))
    fp16 = FakeResult(FakeBoxes([[0.0, 0.0, 10.5, 10.0]], [0.88], [0.0]))

    report = compare_precision_results(fp32, fp16)

    assert report.fp32_detection_count == 1
    assert report.fp16_detection_count == 1
    assert report.matched_detection_count == 1
    assert report.unmatched_fp32_count == 0
    assert report.unmatched_fp16_count == 0
    assert report.mean_matched_iou == pytest.approx(10.0 / 10.5)
    assert report.minimum_matched_iou == pytest.approx(10.0 / 10.5)
    assert report.maximum_coordinate_difference_px == 0.5
    assert report.maximum_confidence_difference == pytest.approx(0.02)
    assert report.to_dict()["matches"][0]["fp32"]["class_id"] == 0
