# Modal NVIDIA L4 Execution

KernelVision uses one Modal-hosted NVIDIA L4 for all CUDA correctness checks,
profiling, and benchmark measurements.

Modal's current GPU API selects the device with `gpu="L4"`. KernelVision uses
`Image.add_local_dir` to expose the local `src/` tree inside the remote
container, following the Modal 1.x source-mounting API.

- [Modal GPU documentation](https://modal.com/docs/guide/gpu)
- [Modal image and local-source documentation](https://modal.com/docs/guide/images)

## Local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,inference,remote]"
modal setup
```

The `remote` extra currently pins Modal 1.2.6. On the development machine,
Modal 1.5.3 could not connect to either API endpoint, while 1.2.6 connected to
the same active profile immediately. Revisit this pin only after verifying a
newer client against the L4 workflow.

If Modal is already configured, verify the active profile:

```bash
modal profile list
```

## Short validation

Run this before spending time on the full benchmark:

```bash
modal run scripts/modal_benchmark.py \
  --warmup 2 \
  --iterations 5 \
  --json-out /tmp/kernelvision_l4_validation.json \
  --csv-out /tmp/kernelvision_l4_validation.csv
```

Confirm that the JSON metadata reports:

- `device: "0"`
- `CUDA available: "True"`
- `GPU` containing `L4`
- Five raw samples
- A non-null peak GPU-memory value

The short validation is not a publishable benchmark.

## Full Milestone 2 baseline

```bash
modal run scripts/modal_benchmark.py \
  --warmup 30 \
  --iterations 200 \
  --json-out results/modal_l4_baseline.json \
  --csv-out benchmarks/raw/modal_l4_baseline.csv
```

Do not copy the short validation or local CPU numbers into project result
tables. Only the full L4 run is eligible for the baseline report, and it should
still be checked for unexpected outliers or instability.

## Connectivity troubleshooting

If the client stalls, check [Modal's official status page](https://status.modal.com/)
and test from a normal terminal. During development on 2026-08-03, the managed
Codex execution environment and Modal 1.5.3 received HTTP 503 responses from
both configured Modal API endpoints. Pinning the project to the previously
working Modal 1.2.6 client resolved profile access. The first remote image also
required `libgl1` and `libglib2.0-0` for OpenCV on Debian Slim.
