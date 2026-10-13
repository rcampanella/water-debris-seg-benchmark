# GPU inventory — water debris benchmark

**Cluster:** Snellius (SURF). Re-run on a **GPU compute node** (or via `srun`) before Slurm training jobs.

---

## Partitions (Slurm)

| Partition | GPU resource | Typical node | Walltime (default max) | Benchmark use |
|---|---|---|---|---|
| `gpu_a100` | `gpu:a100:4` (4× **A100 40GB** per node) | `gcn*` | 5 days | EoMT/VidEoMT fit (`--gres=gpu:4`), eval (`--gres=gpu:1`) |
| `gpu_h100` | `gpu:h100:4` (4× H100 per node) | `gcn*` | 5 days | Optional faster fit; same scripts |
| `gpu_mig` | `gpu:a100_3g.20gb:8` | `gcn2–5` | 5 days | Smoke / single-GPU micro-runs only |
| `gpu_vis` | `gpu:a100:4` | `gcn*` | 1 day | Interactive viz; prefer batch for benchmark |

Login nodes (`int*`) have **no** GPU — do not treat login `nvidia-smi` as authoritative.

---

## Probe log

| When | Where | nvidia-smi | torch | Notes |
|---|---|---|---|---|
| 2026-10-05 | `int6` (login) | not available | not installed (base env) | Expected |
| 2026-10-05 | `gcn45` via `srun -p gpu_a100 --gres=gpu:1` | **1× NVIDIA A100-SXM4-40GB** | **2.1.2+cu121** alloc OK | Module `PyTorch/2.1.2-foss-2023a-CUDA-12.1.1`; full EoMT env (`torch==2.7`) via `setup_eomt_venv.sbatch` |

### Verified working (Slurm smoke)

| Host | GPU index | Name | VRAM | Partition |
|---|---|---|---|---|
| gcn45 | 0 | NVIDIA A100-SXM4-40GB | 40960 MiB | gpu_a100 |

---

## Working devices (policy)

Use Slurm `--gres=gpu:N`; do **not** hard-code `CUDA_VISIBLE_DEVICES` unless debugging. For multi-GPU fit, request `--gres=gpu:4` on `gpu_a100` (full node may be shared — add `--exclusive` only if policy allows).

| Slurm request | Use case |
|---|---|
| `--partition=gpu_a100 --gres=gpu:1` | Zero-shot eval, YOLO, mask→box jobs |
| `--partition=gpu_a100 --gres=gpu:4` | EoMT DDP fine-tune (batch 16 target) |
| `--partition=gpu_h100 --gres=gpu:1` | Optional eval when A100 queue is long |

---

## Excluded / unstable

| ID / host | Reason |
|---|---|
| *(none verified yet)* | Avoid nodes in `drained`, `draining`, `down`, `reboot` states when submitting — check `sinfo -p gpu_a100` |

**Removed from plan:** `elec-gpue002` exclusion applied to a **different** cluster; not used on Snellius.

---

## Failed devices

| Index | Error |
|---|---|
| — | — |

---

## Refresh commands

```bash
# Quick partition view
sinfo -o '%P %G %D %T %N' | grep -E 'PARTITION|gpu_'

# One-GPU probe (charges SBUs)
srun --partition=gpu_a100 --gres=gpu:1 --time=00:05:00 --mem=8G \
  bash -lc 'hostname; nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv'

# After loading PyTorch module / conda env on compute node
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count());
[print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

## Slurm template (eval job)

```bash
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH -J wdseg-eval
#SBATCH -o logs/%x-%j.out
```

---

## Gate status (with swarm)

| Gate | 2026-10-05 Snellius |
|---|---|
| ≥1 working CUDA device | **GO** (A100 verified on `gcn45`) |
| torch CUDA alloc (module smoke) | **GO** (`run_cuda_smoke.sh`) |
| torch CUDA alloc (EoMT venv) | **GO** — job **27590120**: `torch 2.7.0+cu126`, CUDA on A100 (`gcn47`) |
