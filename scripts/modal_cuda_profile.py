"""Profile the naive CUDA preprocessing baseline on a Modal NVIDIA L4."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"
CUDA_DIR = PROJECT_ROOT / "csrc"

runtime_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("ultralytics==8.4.115")
    .env(
        {
            "PYTHONPATH": "/root/src",
            "YOLO_CONFIG_DIR": "/tmp/ultralytics",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "XDG_CACHE_HOME": "/tmp/cache",
        }
    )
    .add_local_dir(SOURCE_DIR, remote_path="/root/src")
    .add_local_dir(CUDA_DIR, remote_path="/root/csrc")
)

app = modal.App("kernelvision-naive-cuda-profile")


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a native command and capture its complete output."""
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


@app.function(image=runtime_image, gpu="L4", timeout=30 * 60)
def profile_l4(
    warmup_launches: int,
    sections: list[str],
    git_commit: str,
    git_worktree_dirty: bool,
) -> dict[str, Any]:
    """Correctness-gate and profile one selected naive CUDA launch."""
    import torch

    from kernelvision.benchmarking.ncu import parse_ncu_csv, parse_ncu_rules
    from kernelvision.environment import collect_environment
    from kernelvision.preprocessing import (
        deterministic_bgr_image,
        read_standalone_output,
        torch_reference_preprocess,
    )

    if warmup_launches < 1:
        raise ValueError("warmup_launches must be positive")
    if not sections:
        raise ValueError("at least one Nsight Compute section is required")

    height = 640
    width = 640
    dtype_name = "fp32"
    block_size = 256
    tolerance = 1e-7
    cuda_source = Path("/root/csrc/preprocessing/standalone_preprocess.cu")
    binary = Path("/tmp/kernelvision_cuda_profile_target")
    ncu_path = Path("/usr/local/cuda/bin/ncu")
    compile_command = [
        "nvcc",
        "-O3",
        "--std=c++17",
        "-lineinfo",
        str(cuda_source),
        "-o",
        str(binary),
    ]
    _run(compile_command)
    nvcc_version = _run(["nvcc", "--version"]).stdout.strip()
    ncu_version = _run([str(ncu_path), "--version"]).stdout.strip()

    cpu_image = deterministic_bgr_image(
        height,
        width,
        torch_module=torch,
        device="cpu",
    )
    input_bytes = cpu_image.numpy().tobytes(order="C")
    expected = torch_reference_preprocess(
        cpu_image.to(device="cuda"),
        output_dtype=torch.float32,
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        input_path = temporary / "input_640x640.bin"
        correctness_output = temporary / "correctness.bin"
        correctness_samples = temporary / "correctness.csv"
        profile_output = temporary / "profile_output.bin"
        profile_samples = temporary / "profile_samples.csv"
        input_path.write_bytes(input_bytes)

        target_arguments = [
            "--height",
            str(height),
            "--width",
            str(width),
            "--dtype",
            dtype_name,
            "--block-size",
            str(block_size),
            "--warmup",
            "1",
            "--iterations",
            "1",
            "--launches-per-sample",
            "1",
            "--input",
            str(input_path),
            "--output",
            str(correctness_output),
            "--samples",
            str(correctness_samples),
        ]
        _run([str(binary), *target_arguments])
        actual = read_standalone_output(
            correctness_output,
            height=height,
            width=width,
            dtype_name=dtype_name,
            torch_module=torch,
        ).to(device="cuda")
        difference = (actual.float() - expected.float()).abs()
        maximum_difference = float(difference.max().item())
        mean_difference = float(difference.mean().item())
        mismatched_values = int((difference > tolerance).sum().item())
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.0,
            atol=tolerance,
        )

        profile_target_arguments = [
            "--height",
            str(height),
            "--width",
            str(width),
            "--dtype",
            dtype_name,
            "--block-size",
            str(block_size),
            "--warmup",
            str(warmup_launches),
            "--iterations",
            "1",
            "--launches-per-sample",
            "1",
            "--input",
            str(input_path),
            "--output",
            str(profile_output),
            "--samples",
            str(profile_samples),
        ]
        ncu_command = [
            str(ncu_path),
            "--target-processes",
            "all",
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            "regex:naive_bgr_hwc_to_rgb_chw.*",
            "--launch-skip",
            str(warmup_launches),
            "--launch-count",
            "1",
            "--cache-control",
            "none",
            "--clock-control",
            "none",
            "--csv",
            "--page",
            "details",
        ]
        for section in sections:
            ncu_command.extend(("--section", section))
        ncu_command.extend((str(binary), *profile_target_arguments))
        ncu_result = _run(ncu_command, check=False)

    combined_output = "\n".join(
        part for part in (ncu_result.stdout, ncu_result.stderr) if part
    )
    metrics: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    parse_error = ""
    if ncu_result.returncode == 0:
        try:
            metrics = parse_ncu_csv(combined_output)
            rules = parse_ncu_rules(combined_output)
        except ValueError as error:
            parse_error = str(error)

    profiling_succeeded = (
        ncu_result.returncode == 0 and bool(metrics) and not parse_error
    )
    return {
        "schema_version": 1,
        "experiment": "naive CUDA Nsight Compute profile",
        "profiling_succeeded": profiling_succeeded,
        "correctness": {
            "passed": True,
            "tolerance": tolerance,
            "maximum_absolute_difference": maximum_difference,
            "mean_absolute_difference": mean_difference,
            "mismatched_values": mismatched_values,
        },
        "metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
            "git_worktree_dirty": git_worktree_dirty,
            "gpu": torch.cuda.get_device_name(),
            "height": height,
            "width": width,
            "dtype": dtype_name,
            "block_size": block_size,
            "warmup_launches_skipped": warmup_launches,
            "profiled_launches": 1,
            "sections": sections,
            "cache_control": "none",
            "clock_control": "none",
            "compile_command": compile_command,
            "ncu_command": ncu_command,
            "nvcc_version": nvcc_version,
            "ncu_version": ncu_version,
            "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "cuda_source_sha256": hashlib.sha256(
                cuda_source.read_bytes()
            ).hexdigest(),
            "environment": collect_environment(),
        },
        "metrics": metrics,
        "rules": rules,
        "diagnostics": {
            "ncu_returncode": ncu_result.returncode,
            "parse_error": parse_error,
            "stdout": ncu_result.stdout,
            "stderr": ncu_result.stderr,
        },
    }


@app.local_entrypoint()
def main(
    warmup_launches: int = 30,
    sections: str = (
        "LaunchStats,Occupancy,SpeedOfLight,MemoryWorkloadAnalysis,"
        "MemoryWorkloadAnalysis_Tables,WarpStateStats,SchedulerStats,"
        "InstructionStats"
    ),
    json_out: str = "results/modal_l4_cuda_profile.json",
) -> None:
    """Run focused profiling and save the complete metric report."""
    parsed_sections = [
        section.strip()
        for section in sections.split(",")
        if section.strip()
    ]
    git_commit = _run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
    ).stdout.strip()
    git_worktree_dirty = bool(
        _run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
        ).stdout.strip()
    )
    report = profile_l4.remote(
        warmup_launches,
        parsed_sections,
        git_commit,
        git_worktree_dirty,
    )
    report["metadata"]["orchestrator_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()

    output = Path(json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"profiling_succeeded={report['profiling_succeeded']}, "
        f"metrics={len(report['metrics'])}, "
        f"returncode={report['diagnostics']['ncu_returncode']}"
    )
    if report["diagnostics"]["parse_error"]:
        print(f"parse_error={report['diagnostics']['parse_error']}")
    print(f"Saved profiler report to {output}")
