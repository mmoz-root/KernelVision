"""Numerical comparison helpers for FP32 and FP16 detections."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """One axis-aligned detection box in x1, y1, x2, y2 form."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        """Return the box area, treating inverted dimensions as empty."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True, slots=True)
class Detection:
    """One model detection used in a precision comparison."""

    box: BoundingBox
    confidence: float
    class_id: int


@dataclass(frozen=True, slots=True)
class DetectionMatch:
    """One FP32 reference detection paired with an FP16 detection."""

    fp32: Detection
    fp16: Detection
    iou: float


@dataclass(frozen=True, slots=True)
class PrecisionComparisonReport:
    """Detection consistency measurements between FP32 and FP16."""

    fp32_detection_count: int
    fp16_detection_count: int
    matched_detection_count: int
    unmatched_fp32_count: int
    unmatched_fp16_count: int
    minimum_match_iou: float
    mean_matched_iou: float | None
    minimum_matched_iou: float | None
    maximum_coordinate_difference_px: float | None
    maximum_confidence_difference: float | None
    matches: tuple[DetectionMatch, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible comparison measurements."""
        return asdict(self)


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    """Return the intersection-over-union score for two boxes."""
    intersection_x1 = max(first.x1, second.x1)
    intersection_y1 = max(first.y1, second.y1)
    intersection_x2 = min(first.x2, second.x2)
    intersection_y2 = min(first.y2, second.y2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)

    intersection_area = intersection_width * intersection_height
    union_area = first.area + second.area - intersection_area

    if union_area == 0:
        return 0.0
    else:
        return intersection_area / union_area


def match_detections(
    fp32_detections: Sequence[Detection],
    fp16_detections: Sequence[Detection],
    *,
    minimum_iou: float = 0.5,
) -> tuple[DetectionMatch, ...]:
    """Greedily pair same-class detections from highest to lowest IoU."""
    candidates = []

    for fp32_index, fp32_detection in enumerate(fp32_detections):
        for fp16_index, fp16_detection in enumerate(fp16_detections):
            if fp32_detection.class_id != fp16_detection.class_id:
                continue

            iou = intersection_over_union(
                fp32_detection.box,
                fp16_detection.box,
            )

            if iou >= minimum_iou:
                candidates.append((iou, fp32_index, fp16_index))

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)

    used_fp32 = set()
    used_fp16 = set()
    matches = []

    for iou, fp32_index, fp16_index in candidates:
        if fp32_index in used_fp32 or fp16_index in used_fp16:
            continue

        used_fp32.add(fp32_index)
        used_fp16.add(fp16_index)
        matches.append(
            DetectionMatch(
                fp32=fp32_detections[fp32_index],
                fp16=fp16_detections[fp16_index],
                iou=iou,
            )
        )

    return tuple(matches)


def _to_list(value: Any) -> list[Any]:
    """Move a tensor-like value to CPU and convert it to a Python list."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if not hasattr(value, "tolist"):
        raise RuntimeError("detection output was not tensor-like")
    return value.tolist()


def extract_detections(result: Any) -> tuple[Detection, ...]:
    """Convert one Ultralytics result into device-independent detections."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ()

    coordinates = _to_list(boxes.xyxy)
    confidences = _to_list(boxes.conf)
    classes = _to_list(boxes.cls)
    if not len(coordinates) == len(confidences) == len(classes):
        raise RuntimeError("detection box, confidence, and class counts differ")

    detections = []
    for xyxy, confidence, class_id in zip(
        coordinates,
        confidences,
        classes,
        strict=True,
    ):
        if len(xyxy) != 4:
            raise RuntimeError("expected each detection box to contain four coordinates")
        detections.append(
            Detection(
                box=BoundingBox(*(float(value) for value in xyxy)),
                confidence=float(confidence),
                class_id=int(class_id),
            )
        )
    return tuple(detections)


def compare_precision_results(
    fp32_result: Any,
    fp16_result: Any,
    *,
    minimum_iou: float = 0.5,
) -> PrecisionComparisonReport:
    """Summarize numerical differences between two detection results."""
    if not 0.0 <= minimum_iou <= 1.0:
        raise ValueError("minimum_iou must be between 0.0 and 1.0")

    fp32_detections = extract_detections(fp32_result)
    fp16_detections = extract_detections(fp16_result)
    matches = match_detections(
        fp32_detections,
        fp16_detections,
        minimum_iou=minimum_iou,
    )

    if matches:
        matched_ious = [match.iou for match in matches]
        coordinate_differences = [
            abs(fp32_coordinate - fp16_coordinate)
            for match in matches
            for fp32_coordinate, fp16_coordinate in zip(
                asdict(match.fp32.box).values(),
                asdict(match.fp16.box).values(),
                strict=True,
            )
        ]
        confidence_differences = [
            abs(match.fp32.confidence - match.fp16.confidence)
            for match in matches
        ]
        mean_iou = sum(matched_ious) / len(matched_ious)
        minimum_matched_iou = min(matched_ious)
        maximum_coordinate_difference = max(coordinate_differences)
        maximum_confidence_difference = max(confidence_differences)
    else:
        mean_iou = None
        minimum_matched_iou = None
        maximum_coordinate_difference = None
        maximum_confidence_difference = None

    return PrecisionComparisonReport(
        fp32_detection_count=len(fp32_detections),
        fp16_detection_count=len(fp16_detections),
        matched_detection_count=len(matches),
        unmatched_fp32_count=len(fp32_detections) - len(matches),
        unmatched_fp16_count=len(fp16_detections) - len(matches),
        minimum_match_iou=minimum_iou,
        mean_matched_iou=mean_iou,
        minimum_matched_iou=minimum_matched_iou,
        maximum_coordinate_difference_px=maximum_coordinate_difference,
        maximum_confidence_difference=maximum_confidence_difference,
        matches=matches,
    )
