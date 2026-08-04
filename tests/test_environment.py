"""Tests for runtime environment reporting."""

from kernelvision.environment import collect_environment, format_environment


def test_collect_environment_includes_required_fields() -> None:
    """The report should expose the fields needed for reproducibility."""
    info = collect_environment()

    required_fields = {
        "Platform",
        "Architecture",
        "Python",
        "PyTorch",
        "Triton",
        "NumPy",
        "OpenCV",
        "CUDA available",
        "GPU",
    }
    assert required_fields <= info.keys()


def test_format_environment_outputs_one_entry_per_line() -> None:
    """Terminal formatting should be stable and easy to read."""
    info = {"Python": "3.11.9", "GPU": "none"}

    assert format_environment(info) == "Python: 3.11.9\nGPU: none"
