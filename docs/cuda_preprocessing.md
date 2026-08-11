# Standalone CUDA Preprocessing

## Milestone boundary

Milestone 5 uses a standalone CUDA executable:

```text
Python Modal orchestrator
    ├── writes an exact uint8 BGR-HWC fixture
    ├── computes the trusted PyTorch result on CUDA
    └── launches a raw nvcc-built executable
            ├── reads the fixture
            ├── runs a CUDA kernel
            ├── writes raw FP32/FP16 RGB-CHW output
            └── writes native CUDA-event samples
```

There is no PyTorch C++ extension, ATen, pybind, tensor-pointer sharing, or
YOLO integration here. Those belong to Milestone 6. The file boundary adds no
cost to the kernel measurement: input and output buffers are allocated before
CUDA events, and H2D/D2H copies are outside them.

## Contract

```text
Input:  one contiguous CUDA uint8 BGR [H, W, 3], values 0–255
Output: one contiguous CUDA RGB [3, H, W], FP32 or FP16, values 0–1
```

Resize, letterbox, batching, CPU decode, and host-to-device transfer remain
excluded.

## Files

- `csrc/preprocessing/standalone_preprocess.cu`
  - raw CUDA runtime harness
  - typed FP32/FP16 dispatch
  - CUDA error checks
  - preallocated device buffers
  - CUDA-event sampling
  - the validated naive one-thread-per-pixel kernel
  - the shared-memory coalescing experiment
  - the warp-packed shuffle experiment
- `src/kernelvision/preprocessing/cuda_standalone.py`
  - deterministic fixture definition
  - raw CUDA-output reader
- `scripts/modal_cuda_preprocessing.py`
  - Modal NVIDIA L4 build and correctness matrix
- `tests/test_cuda_standalone.py`
  - portable tests for the cross-process data protocol

## Deterministic correctness protocol

For flat input byte offset `i`:

```text
value(i) = (37 × i + 13) mod 256
```

Python writes those exact bytes to a headerless fixture. Both the trusted
PyTorch operation and the CUDA executable consume that fixture. CUDA writes a
headerless output whose declared shape and dtype are known by the orchestrator.
The comparison uses `rtol=0`, `atol=1e-7` for FP32, and `atol=5e-4` for FP16.

The initial matrix matches Milestone 4: `2×3`, `5×7`, `384×640`, `640×640`,
and `641×639`, each in FP32 and FP16. The irregular sizes exercise the final
partial block.

## Exercise 1 — naive one-thread-per-pixel kernel

Open `naive_bgr_hwc_to_rgb_chw()` in
`csrc/preprocessing/standalone_preprocess.cu` and implement the marked body.

For pixel `i` and `P = H × W`:

```text
input:  B = 3i       G = 3i + 1       R = 3i + 2
output: R = i        G = P + i        B = 2P + i
```

Required steps:

1. Compute `i = blockIdx.x * blockDim.x + threadIdx.x`.
2. Return if `i >= pixel_count`.
3. Load the three interleaved bytes.
4. Convert and multiply by `1.0f / 255.0f`.
5. Store three planar values through `convert_output<Output>()`.

Do not optimize, use shared memory, vectorize, or change the harness yet. The
first goal is a simple trusted CUDA baseline.

After implementing it, run:

```bash
modal run scripts/modal_cuda_preprocessing.py --block-size 256
```

Acceptance criterion: all 10 cases pass before any benchmark or optimized
kernel is added. The one-sample event files created during correctness are
plumbing checks, not performance results.

## Naive correctness result

The learner-written naive kernel compiled with CUDA 13.0 (`nvcc 13.0.48`) and
passed all 10 cases on a Modal NVIDIA L4. FP32 and FP16 both had zero maximum
absolute difference and zero mismatched values against the trusted PyTorch
reference. This establishes correctness only; the one-event correctness runs
are not benchmark results.

Report: `results/modal_l4_cuda_preprocess_correctness.json`

## Naive CUDA baseline benchmark

### Methodology

The controlled comparison ran in one Modal NVIDIA L4 environment with PyTorch
2.13.0, CUDA 13.0, Triton 3.7.1, and nvcc 13.0.48. Each shape/dtype case first
rechecked Triton and CUDA against the PyTorch reference. All eight cases passed
with zero mismatched values.

Input and output buffers are resident before timing. H2D, D2H, compilation,
context creation, subprocess startup, and outer wall time are excluded. Triton
and CUDA outputs are preallocated; PyTorch retains its normal multi-operation
reference behavior.

Because these operations are only a few microseconds, every event sample times
100 repeated launches and divides the interval by 100. Each implementation has
30 warmups and 200 samples in each of three position-balanced rounds:

```text
round 1: PyTorch → Triton → naive CUDA
round 2: Triton → naive CUDA → PyTorch
round 3: naive CUDA → PyTorch → Triton
```

This produces 600 samples per implementation/case and 14,400 saved CSV rows.
Block size 256 is the declared naive CUDA/Triton baseline, with four Triton
warps. No parameter search contributes to these headline results.

### Results

Combined medians:

| Shape | Dtype | PyTorch | Triton | Naive CUDA | CUDA vs PyTorch | CUDA vs Triton |
|---|---:|---:|---:|---:|---:|---:|
| 320×320 | FP32 | 0.05736 ms | 0.02644 ms | 0.00306 ms | 18.73× | 8.64× |
| 320×320 | FP16 | 0.05919 ms | 0.02646 ms | 0.00284 ms | 20.87× | 9.33× |
| 384×640 | FP32 | 0.05902 ms | 0.02712 ms | 0.00389 ms | 15.17× | 6.97× |
| 384×640 | FP16 | 0.05869 ms | 0.02668 ms | 0.00384 ms | 15.28× | 6.95× |
| 640×640 | FP32 | 0.06169 ms | 0.02642 ms | 0.00557 ms | 11.08× | 4.75× |
| 640×640 | FP16 | 0.05840 ms | 0.02663 ms | 0.00414 ms | 14.12× | 6.44× |
| 720×1280 | FP32 | 0.10287 ms | 0.02644 ms | 0.01111 ms | 9.26× | 2.38× |
| 720×1280 | FP16 | 0.06460 ms | 0.02643 ms | 0.00815 ms | 7.93× | 3.24× |

At 640×640 FP32, naive CUDA saves 0.05612 ms versus PyTorch and 0.02085 ms
versus Triton. Against an approximately 13 ms detector, even the PyTorch-to-
CUDA component saving represents only about 0.43% of end-to-end latency before
integration overhead.

### Interpretation limits

- This is not an end-to-end YOLO result.
- Native CUDA launches from C++, while Triton and PyTorch dispatch through
  Python/framework paths. CUDA-event intervals can retain GPU idle gaps caused
  by those different submission paths; device instructions alone do not
  explain the full difference.
- Repeated launches reuse the same input and output buffers. This is a
  warm-cache steady-state benchmark, not a fresh-frame DRAM-throughput test.
- The historical Milestone 4 Triton result used an allocating wrapper and one
  operation per event interval. The current preallocated, amplified Triton
  result has a different boundary and does not show a kernel optimization.
- Some sub-microsecond differences between native CUDA rounds are large in
  percentage terms because the complete native operation is only a few
  microseconds. Raw per-round samples remain available.
- The naive result establishes the next optimization control. It does not
  establish that a more complex CUDA kernel will be faster or worthwhile.

Reports:

- `results/modal_l4_cuda_preprocess_benchmark.json`
- `benchmarks/raw/modal_l4_cuda_preprocess_benchmark.csv`

## Naive CUDA block-size experiment

### Question and controls

The exploratory experiment asked whether thread-block grouping materially
changes the unchanged naive kernel at 640×640 FP32. It varied only block size:
128, 256, 512, and 1024 threads. Pixel mapping, input, dtype, kernel body,
measurement amplification, and allocation/timing boundaries remained fixed.
Every configuration passed correctness with zero mismatched values.

Each configuration used 30 warmups, 100 launches per event interval, and 200
samples in each of four cyclic orders. Every block size occupied the first,
second, third, and fourth position exactly once. This produced 800 samples per
configuration and 3,200 raw CSV rows.

### Result

| Block size | Blocks | Warps/block | Median | Round-median span | Change vs 256 |
|---:|---:|---:|---:|---:|---:|
| 128 | 3,200 | 4 | 5.70368 µs | 0.08192 µs | 0.01024 µs slower |
| 256 | 1,600 | 8 | 5.69344 µs | 0.09808 µs | baseline |
| 512 | 800 | 16 | 5.65248 µs | 0.08192 µs | 0.04096 µs faster |
| 1024 | 400 | 32 | 5.76512 µs | 0.09120 µs | 0.07168 µs slower |

Block 512 was the nominal exploratory minimum, only 0.72% or 40.96 ns faster
than 256. That saving was smaller than the observed 98.08 ns round-median span,
and 512 beat 256 in only two of four rounds. The predeclared evidence rule was
therefore not met. No confirmation run was justified, and block size 256
remains the robust default.

This negative result is useful: changing launch grouping alone did not produce
a repeatable improvement large enough to justify retuning the baseline. The
next kernel change should be motivated by profiling rather than by choosing the
smallest noisy median.

Reports:

- `results/modal_l4_cuda_block_size_experiment.json`
- `benchmarks/raw/modal_l4_cuda_block_size_experiment.csv`

## Naive CUDA Nsight Compute profile

The unchanged 640×640 FP32, block-256 baseline was correctness-gated and then
profiled on a Modal NVIDIA L4 with Nsight Compute 2025.3.0. The executable ran
30 warmup launches; `--launch-skip 30 --launch-count 1` selected one subsequent
kernel launch. The focused report includes Launch Stats, Occupancy, Speed of
Light, Memory Workload Analysis and tables, Warp State, Scheduler, and
Instruction Stats. The CUDA source hash and complete profiler command are saved
with the results.

The selected launch remained exact against the PyTorch reference. Nsight
reported a 5.664 µs device duration, close to the approximately 5.5–5.7 µs
repeated-launch benchmark range. This agreement indicates that native host
submission gaps are not the dominant part of that amplified native CUDA
result. Profiler duration remains diagnostic and does not replace the
position-balanced timing report.

Key metrics were:

| Metric | Result |
|---|---:|
| Registers per thread | 16 |
| Static/dynamic shared memory | 0 B / 0 B |
| Theoretical / achieved occupancy | 100% / 73.04% |
| Achieved active warps per SM | 35.06 of 48 |
| L2 hit rate | 99.82% |
| L2 / DRAM throughput | 69.59% / 10.91% of peak |
| Compute throughput | 28.75% of peak |
| Scheduler cycles with no eligible warp | 74.35% |
| Active / eligible warps per scheduler | 8.76 / 0.49 |
| Warp cycles per issued instruction | 34.13 cycles |

The high L2 hit rate confirms that repeated buffers produce a warm-cache
profile rather than fresh-frame DRAM behavior. Low register use and zero
kernel shared memory allow 100% theoretical occupancy; the measured 73.04%
reflects an average over a short 4.60-wave launch, including its partial final
wave, rather than an obvious register or shared-memory limit.

The actionable profiler finding was the interleaved BGR load pattern. A warp's
per-channel byte loads use addresses `0, 3, 6, ...`, so Nsight reported only
10.7 useful bytes per transmitted 32-byte sector. Scheduler analysis reported
15.8 long-scoreboard cycles waiting on L1TEX dependencies, 46.2% of the average
34.1 cycles between issued instructions. Planar output stores are already
coalesced. The next isolated hypothesis is therefore to improve contiguous
input loading while preserving those output stores, not to add arithmetic,
reduce registers, or stage one-use pixels through shared memory without a
measured benefit.

Nsight rule speedup fields are heuristic opportunities, not measured or
additive predictions. Caches and clocks were deliberately left uncontrolled,
matching the warm-cache diagnostic boundary; the controlled CUDA-event
benchmark remains the performance authority.

Report:

- `results/modal_l4_cuda_profile.json`

## Profile-guided optimization experiments

The profile motivated two isolated attempts to replace the stride-three input
loads while preserving the already-coalesced planar output stores. Both
candidates retained the naive kernel as the control and passed the same
PyTorch-reference correctness gate in FP32 and FP16.

### Candidate 1 — cooperative shared-memory staging

Each block cooperatively copies its contiguous BGR bytes into a dynamic
shared-memory tile. After `__syncthreads()`, one thread reads each local pixel
and writes planar RGB. This changes warp global loads from `0, 3, 6, ...` to
contiguous byte ranges, but adds shared-memory writes, a block barrier, and
shared-memory reads.

All 10 correctness cases passed with zero maximum absolute difference. A
four-position balanced benchmark showed that the added work cost more than the
improved coalescing saved. At 640×640, the candidate was 41.9% slower in FP32
and 66.8% slower in FP16 than naive CUDA. It lost every round at that size.

Reports:

- `results/modal_l4_cuda_preprocess_coalesced_correctness.json`
- `results/modal_l4_cuda_optimization_benchmark.json`
- `benchmarks/raw/modal_l4_cuda_optimization_benchmark.csv`

### Candidate 2 — warp-packed loads and shuffles

The lower-overhead candidate assigns one warp to 32 pixels, or 96 BGR bytes.
Lanes 0–23 each load one adjacent 32-bit word. Every output lane locates its
three channel bytes, obtains the owning lane's word through `__shfl_sync`, and
extracts the byte with shift-and-mask operations. The final partial warp uses
a safe scalar fallback. This removes shared memory and the block barrier while
retaining contiguous 32-bit global loads and planar output stores.

All 10 correctness cases again passed with zero maximum absolute difference.
The final benchmark used 30 warmups, 200 samples per round, 100 launches per
event interval, and five cyclic orders. Each of PyTorch, Triton, naive CUDA,
shared-memory CUDA, and warp-packed CUDA occupied every execution position
once. This produced 1,000 samples per implementation/case and 40,000 raw rows.

CUDA medians from that single controlled run:

| Shape | Dtype | Naive | Shared staging | Warp packed |
|---|---:|---:|---:|---:|
| 320×320 | FP32 | 3.144 µs | 3.523 µs | 3.133 µs |
| 320×320 | FP16 | 3.072 µs | 3.154 µs | 3.062 µs |
| 384×640 | FP32 | 3.891 µs | 5.110 µs | 4.239 µs |
| 384×640 | FP16 | 3.308 µs | 4.454 µs | 3.799 µs |
| 640×640 | FP32 | 5.519 µs | 7.813 µs | 6.308 µs |
| 640×640 | FP16 | 4.065 µs | 6.799 µs | 5.181 µs |
| 720×1280 | FP32 | 11.223 µs | 16.210 µs | 12.943 µs |
| 720×1280 | FP16 | 7.982 µs | 14.367 µs | 11.274 µs |

At 320×320, warp packing was nominally 0.3% faster, only about 10 ns, and won
two of five FP32 rounds and three of five FP16 rounds. That is not meaningful
evidence of improvement. At every larger shape it lost all five rounds and was
8.9–41.2% slower than naive CUDA. Shuffle and bit-extraction work therefore
cost more than the packed loads saved under this warm-cache boundary.

### Milestone 5 conclusion

The profiler correctly identified inefficient per-instruction sector use, but
optimizing that metric did not reduce total kernel latency. The 99.82% L2 hit
rate made the naive loads inexpensive, while both alternatives added work.
The naive one-thread-per-pixel kernel remains the fastest robust standalone
implementation and is the implementation to carry into Milestone 6.

No extra Nsight run was performed for the losing candidates: the predeclared
performance gate reserves deeper profiling for a correct, meaningful timing
improvement. The negative results are retained because they demonstrate the
full profile → hypothesis → correctness → controlled measurement loop.

Final reports:

- `results/modal_l4_cuda_preprocess_warp_packed_correctness.json`
- `results/modal_l4_cuda_final_benchmark.json`
- `benchmarks/raw/modal_l4_cuda_final_benchmark.csv`
