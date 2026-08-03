"""Runtime environment inspection for reproducible experiments."""

from __future__ import annotations

import platform
from importlib import metadata


def _package_version(package_name: str) -> str:
    """Return an installed package version without importing the package."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def collect_environment() -> dict[str, str]:
    """Collect software and accelerator information for the active runtime."""
    info = {
        "Platform": platform.platform(),
        "Architecture": platform.machine(),
        "Python": platform.python_version(),
        "PyTorch": _package_version("torch"),
        "Torchvision": _package_version("torchvision"),
        "Ultralytics": _package_version("ultralytics"),
    }

    if info["PyTorch"] == "not installed":
        info["PyTorch CUDA build"] = "unavailable"
        info["CUDA available"] = "unavailable"
        info["CUDA devices"] = "unavailable"
        info["GPU"] = "unavailable"
        info["MPS available"] = "unavailable"
        return info

    try:
        import torch
    except Exception as error:  # Report broken installations without hiding the cause.
        info["PyTorch import"] = f"failed: {type(error).__name__}: {error}"
        return info

    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    mps_backend = getattr(torch.backends, "mps", None)

    info["PyTorch CUDA build"] = str(torch.version.cuda or "none")
    info["CUDA available"] = str(cuda_available)
    info["CUDA devices"] = str(cuda_device_count)
    info["GPU"] = torch.cuda.get_device_name(0) if cuda_device_count else "none"
    info["MPS available"] = str(
        bool(mps_backend and mps_backend.is_available())
    )
    return info


def format_environment(info: dict[str, str]) -> str:
    """Format environment information for terminal output."""
    return "\n".join(f"{label}: {value}" for label, value in info.items())

