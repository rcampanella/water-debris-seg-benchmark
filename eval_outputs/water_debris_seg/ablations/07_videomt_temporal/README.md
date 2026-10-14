# Ablation 07 — VidEoMT temporal on LaRS sequences

**Question.** Does **online** VidEoMT beat per-frame EoMT on LaRS keyframes for PQ/mIoU and **temporal stability** (mask flicker, thing ID switches)?

**Expected outcome.** Similar or slightly better PQ vs EoMT still; clear win on ID stability for boats/buoys over 9-frame context.

**Hard rule.** VidEoMT **training** (if any) uses **LaRS sequences only**. Box sets: eval-only mask→box on stills (optional side table).

## Setup

| Item | Value |
|---|---|
| Data | LaRS sequence images + panoptic keyframes |
| Compare | EoMT still vs VidEoMT online |
| Metrics | PQ, flicker score, ID switch rate |
| Speed | FPS via `benchmark.py` |

## Results

| Method | PQ | Flicker | ID switches | FPS | Source |
|---|---|---|---|---|---|
| EoMT per-frame (val keyframes PQ) | ≈0.395 | — | — | — | ablation 03 validate |
| EoMT semantic seq flicker (ab02 ckpt) | — | **0.041** (1−mean consec. mIoU) | — | — | job **27711595**, W&B `g1bfz3nn`; 35 val clips |
| EoMT panoptic seq flicker (ab03 ckpt) | — | **0.229** | — | — | job **27715321**, W&B `5aoppsyb` |
| VidEoMT online (ViT-B) | VIS AP **0.000** @1k/2k/3k/4k (**bad run** — `SEM_SEG_HEAD.NUM_CLASSES=54` vs backbone 8) | — | — | — | **27760828** invalid; retrain after config fix |
| VidEoMT online (ViT-B) | VIS AP **0.000** (**bad run** — labels fixed, but ~92% of sampled windows had no labeled keyframe → no mask supervision) | — | — | — | **27769861** invalid |
| VidEoMT online (ViT-B), keyframe sampling + labeled-frame loss, 4k it | upstream VIS AP 0.000 (metric artefact, see below). **Per-keyframe** AP **0.015** / AP50 0.040 / APs 0.005 / APm 0.265 / APl 0.407 (`MAX_NUM`=10) | — | — | — | train **27772987**, W&B `lars_vitb_27772987`; `sources/videomt_rescore_keyframe_27772987.json` |
| ↳ same ckpt, eval-only `MAX_NUM`=50 | **Per-keyframe** AP **0.016** / AP50 0.041 / AP75 0.008 / APs 0.007 / APm 0.271 / APl 0.193; AR10 0.048 | — | — | — | eval **27775358** (27774515 hung on GPFS, node gcn43); `sources/videomt_rescore_keyframe_27775358.json` |
| VidEoMT seq flicker (27772987 preds, eval 27775358) | — | **0.612** (8-class thing mIoU proxy; vs EoMT semantic **0.041**) | — | — | `sources/videomt_seq_flicker_27775358_local.json`; sbatch **27776809** |
| VidEoMT 16k it (patched) | in-train keyframe AP **0.010** / APm **0.105** (`MAX_NUM`=10) | — | — | — | train **27776810** (~1h46m); mask/dice loss active; W&B `lars_vitb_16k_27776810` |
| ↳ eval-only `MAX_NUM`=50 | **Per-keyframe** AP **0.009** / AP50 0.026 / APs 0.007 / APm **0.116** / APl 0.060; AR10 0.044 | — | — | — | eval **27806956**; `sources/videomt_rescore_keyframe_27806956.json` |
| VidEoMT seq flicker (16k, eval 27806956) | — | **0.311** (vs 4k **0.612**, EoMT semantic **0.041**) | — | — | **27807251**; `sources/videomt_seq_flicker_27807251.json` |

**Flicker baseline detail:** Mean consecutive-frame semantic mIoU ≈ **0.959** on val `images_seq` clips tied to keyframes (`sources/eomt_semantic_seq_flicker_27711595.json`). Worst clips: `inhouse_seq_141` (0.148), `inhouse_seq_131` (0.107), `inhouse_seq_82` (0.084). VidEoMT must beat this proxy on the same clips.

## Verdict

**VidEoMT (2026-10-09).** With keyframe patches, training has real mask supervision. **4k** (27772987): per-keyframe AP **0.016**, flicker **0.612**. **16k** (27776810): mask/dice loss nonzero, but **per-keyframe AP 0.009** @50 did **not** improve vs 4k; **flicker 0.311** improved vs 4k yet remains far above EoMT semantic **0.041** (8-class thing proxy ≠ semantic 3-class). VidEoMT does not beat per-frame EoMT on stability or keyframe detection at this schedule. Not comparable to EoMT PQ **0.395** without same-unit instance AP or VidEoMT PQ.

Per-frame **LaRS semantic FT is already temporally stable** on most val clips (low flicker proxy); gains from VidEoMT may be modest on mIoU/PQ but still worth testing **thing ID consistency** once panoptic video training is wired.

*Prep complete* (job 27652852: 57541 sequence JPGs on scratch; VidEoMT repo cloned). **YTVIS export:** jobs **27715327** / **27715328**. **Train:** `ablation07_videomt_train.sbatch` → `run_videomt_train_lars.py` + `configs/ablation07_videomt_lars_vitb.yaml` (ViT-B online, init YT19 segmenter, 4k iters). **Retrain (SEM_SEG + `prepare_targets` patch):** job **27769861** (mask-index fix; canceled **27769704**) — still AP 0, see diagnosis below. **Retrain with keyframe patches:** job **27772987**.

## Zero mask-loss diagnosis (27760828 / 27769861)

`loss_mask` / `loss_dice` showed **0 for all 4000 iters** even after the class-id fixes. Root cause is **frame sampling vs keyframe-only labels**, not the export or loss code:

- 1028 / 1291 train clips are **10 frames with the keyframe at index 9**. Upstream `YTVISDatasetMapper.select_frames` (5 frames, range 2) picks `start = randrange(L − 5)`, so it covers frames 0–8 at most and **never samples index 9**. Overall only **~8.5%** of windows hold a labeled frame (measured 7–10% in the smoke).
- With batch 2, most iterations have **zero GT** → mask/dice loss exactly 0. D2 logs the **median over 20 iters**, so the log shows 0 throughout. The CE loss only learns "no-object" → VIS AP 0.
- A second problem: `VideoHungarianMatcher_Consistent` + `VideoSetCriterion` repeat the matching on every frame. On LaRS, unlabeled frames (`id == -1`) would be supervised as **empty mask / no-object**, which is wrong for "not annotated".

**Fix** (`water_debris_seg/videomt/keyframe_patch.py`, applied in `run_videomt_train_lars.py`):
1. `apply_keyframe_sampling_patch` — training windows always contain a randomly chosen labeled keyframe (keeps `reverse_agu`).
2. `apply_labeled_frames_loss_patch` — match on the full window (keeps query assignment consistent), then compute CE/mask/dice **only on labeled frames**, and only for tracks present in that frame.

**Verification** (`scripts/videomt_one_batch_loss_smoke.py`, job **27772527**, `sources/videomt_loss_smoke_27772527.log`): upstream had 0/8 batches with GT (`loss_mask` = 0). Patched: 500/500 windows labeled, 8/8 batches with GT, `loss_mask` ≈ 3.6–3.9, `loss_dice` ≈ 4.9–5.0 at init. About 7% of keyframe windows still lose all GT because tiny objects vanish after resize; that is expected.

## Why upstream VIS AP stays 0 on this export (eval-side)

Even after the loss fix, the in-training eval reported AP/AR **exactly 0** at every period. Two eval-side causes:

- **GT tracks are single-frame.** In `lars_ytvis_val.json` every one of the 512 GT tracks has exactly **one** labeled frame, because LaRS gives no instance IDs that link keyframes. Val clips hold **2–5 keyframes** (15 frames apart) and are otherwise unlabeled.
- **Upstream video IoU** (`ytvoseval.iou_seq`) adds every predicted mask on a GT-less frame to the union. The online model predicts a mask on every frame, so IoU is ≤ ~1/L per track → AP 0 by construction.

Oracle check (`scripts/rescore_videomt_ytvis_keyframes.py --oracle`): GT used as predictions on exactly its own frame scores AP **1.0** under both metrics. The same GT mask copied to all frames scores **0.0** upstream.

**Metric used from now on — per-keyframe VIS AP.** Each (video, labeled keyframe) is a 1-frame unit for GT and predictions (`split_ytvis_by_keyframe`), so it is instance-mask AP on keyframes from the video model, with the track's score. Applied in-training via `apply_keyframe_eval_patch` (launcher) and offline via `rescore_videomt_ytvis_keyframes.py`. It is **not** a temporal-ID metric. LaRS cannot provide GT ID switches, so temporal stability still relies on the flicker proxy (vs EoMT **0.041** / **0.229**).

Inference keeps `MODEL.BACKBONE.TEST.MAX_NUM` = **10** tracks per clip, while val clips have a median of 11 (max 61) per-keyframe GT instances. That caps recall (AR10 0.034), so the eval-only rerun uses 50 (`ablation07_videomt_eval.sbatch <ckpt> 50`).

## Prep log

| Check | Result | Source |
|---|---|---|
| Sequence images | OK | `sources/sequences_check_27652852.log` |
| VidEoMT clone | `/scratch-shared/.../third_party/videomt` | prep sbatch |
| Detectron2 + deps | **installed** in shared eomt venv (`install_videomt_deps.sh`); env smoke job **27715017** OK | `water-debris-seg/docs/VIDEOMT.md` |
| LaRS YTVIS JSON | **1291** train / **35** val videos; **6078** / **512** thing tracks | job **27715327** → `data/lars/ytvis/lars_ytvis_{split}.json` |
| D2 catalog smoke | `lars_ytvis_{train,val}` load OK | job **27715328** |
| Sequence inventory | **1557 clips**, **32354 frames** (train/val/test `images_seq`) | `sources/lars_sequences_inventory.json` |
| Val keyframe attributes | 198 images; scene/lighting/reflection/wave counts | `sources/lars_val_attributes.json` (job 27711595) |
| EoMT seq flicker JSON | 35 clips | `sources/eomt_semantic_seq_flicker_27711595.json` |

## Resources copied here

| Original | Copy in this folder | Present |
|---|---|---|
| — | sources/ | — |

## Code components

- [VidEoMT](https://github.com/tue-mps/videomt) `train_net_video.py`
- `scripts/install_videomt_deps.sh`, `export_lars_ytvis_json.py`, `lars_ytvis_load_smoke.py`, `videomt_one_batch_loss_smoke.py`, `lars_seq_videomt_flicker.py`, `inventory_lars_sequences.py`, `lars_seq_eomt_flicker.py`, `summarize_lars_attributes.py`, `ablation07_*` sbatch, `water_debris_seg/videomt/{lars_ytvis,prepare_targets_patch,keyframe_patch}.py`, `docs/VIDEOMT.md`

<!-- wandb-links:start (generated by scripts/wandb_audit.py; do not edit) -->

## W&B runs

Online runs for this ablation (fetched 2026-10-14); value-by-value comparison with the reported results: [`WANDB_AUDIT.md`](../../WANDB_AUDIT.md).

| Run | State | Slurm | Audit |
|---|---|---|---|
| [`eomt_seq_flicker_27711595`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/g1bfz3nn) | finished | 27711595 | ✅ online values match the reported results |
| [`eomt_panoptic_seq_flicker_27715321`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/5aoppsyb) | finished | 27715321 | ✅ online values match the reported results |
| [`videomt_lars_vitb_27716592`](https://wandb.ai/riccardo-campanella/videomt/runs/lars_vitb_27716592) | failed | 27716592 | failed attempt |
| [`videomt_lars_vitb_27718883`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_vitb_27718883) | failed | 27718883 | failed attempt |
| [`videomt_lars_vitb_27760828`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_vitb_27760828) | finished | 27760828 | ab07 documented bad run (NUM_CLASSES mismatch, AP 0.000) |
| [`videomt_lars_vitb_27769294`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_vitb_27769294) | crashed | 27769294 | crashed attempt |
| [`videomt_lars_vitb_27769704`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_vitb_27769704) | crashed | 27769704 | crashed attempt |
| [`videomt_lars_vitb_27769861`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_vitb_27769861) | finished | 27769861 | ab07 documented bad run (no mask supervision, AP 0.000) |
| [`videomt_lars_vitb_27772987`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_vitb_27772987) | finished | 27772987 | ab07 VidEoMT 4k training run behind AP 0.016 / flicker 0.612 |
| [`videomt_lars_eval_27775358`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/eval_27775358) | finished | 27775358 | ab07 eval of 4k ckpt (AP 0.016); W&B summary empty |
| [`videomt_lars_vitb_16k_27776810`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_vitb_16k_27776810) | finished | 27776810 | ab07 VidEoMT 16k training run (in-train AP 0.010) |
| [`videomt_seq_flicker_27776809`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/wzpe700z) | finished | 27776809 | ✅ online values match the reported results |
| [`videomt_lars_eval_27806956`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/eval_27806956) | finished | 27806956 | ab07 eval of 16k ckpt (AP 0.009); W&B summary empty |
| [`videomt_seq_flicker_27807251`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/r4ugnnd5) | finished | 27807251 | ✅ online values match the reported results |
| [`ab07_keyframe_ap_27772987`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/ab07_keyframe_ap_27772987) | finished | 27772987 | ✅ online values match the reported results |
| [`ab07_keyframe_ap_27775358`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/ab07_keyframe_ap_27775358) | finished | 27775358 | ✅ online values match the reported results |
| [`ab07_keyframe_ap_27806956`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/ab07_keyframe_ap_27806956) | finished | 27806956 | ✅ online values match the reported results |

<!-- wandb-links:end -->
