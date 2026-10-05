# GPU inventory — water debris benchmark

Re-run on the **compute node** before Slurm jobs. Slurm templates must include `--exclude=elec-gpue002`.

## Probe (Stage 0 — login / dev node)

| When | nvidia-smi | torch | Notes |
|---|---|---|---|
| 2026-10-05 | not available | 2.13.0+cu130, `cuda_available=False` | No GPU on this host; re-probe on GPU partition |

## Working devices

| Index | Name | Slurm use | Probe |
|---|---|---|---|
| — | — | — | *None verified yet* |

## Excluded (policy)

| ID / host | Reason |
|---|---|
| `elec-gpue002` | Cluster convention — broken or unstable (see `docs/wm_states_demo_plan.md`) |

## Failed devices

| Index | Error |
|---|---|
| — | — |

## Commands to refresh

```bash
nvidia-smi -L
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count());
[print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```
