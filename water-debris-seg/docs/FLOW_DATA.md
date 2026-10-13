# FloW-Img on Snellius (ablation 04)

Official access: [ORCA FloW portal](https://orca-tech.cn/en/datasets/FloW/Introduction) (application required). GitHub pointer: [ORCAUBOAT/FloW-Dataset](https://github.com/ORCAUBOAT/FloW-Dataset).

## After download

1. Extract under scratch: `/scratch-shared/$USER/water-debris-seg/data/flow/`

   From NAS `FloW_IMG.zip` (test-only, ~250 MB unpacked):

   ```bash
   FLOW=/scratch-shared/$USER/water-debris-seg/data/flow
   unzip -q "$FLOW/sources/flow_img/FloW_IMG.zip" "FloW_IMG/test/*" -d "$FLOW"
   mv "$FLOW/FloW_IMG/test" "$FLOW/test" && rm -rf "$FLOW/FloW_IMG"
   touch "$FLOW/.flow_img_extracted"
   ```
2. Normalize to **test** split (800 images in paper; single class floating waste / bottle):

```
data/flow/test/
  images/          # *.jpg
  Annotations/     # Pascal VOC *.xml (LabelImg export from ORCA package)
```

Optional YOLO layout instead:

```
data/flow/test/images/
data/flow/test/labels/   # one .txt per image
```

3. Mark ready:

```bash
touch /scratch-shared/$USER/water-debris-seg/data/flow/.flow_img_extracted
```

4. Verify + build COCO GT:

```bash
python water-debris-seg/scripts/verify_flow.py
python water-debris-seg/scripts/masks_to_coco_boxes.py voc-split \
  --images data/flow/test/images \
  --annotations data/flow/test/Annotations \
  --out eval_outputs/water_debris_seg/ablations/04_flow_mask2box/sources/flow_test_gt_coco.json
```

5. Eval (EoMT mask→box, eval only):

```bash
sbatch water-debris-seg/scripts/snellius/ablation04_eomt_flow_zeroshot.sbatch
sbatch water-debris-seg/scripts/snellius/ablation04_eomt_flow_larsft.sbatch
```

## Benchmark use

- **Ablation 04:** zero-shot + LaRS panoptic FT → mAP / AP_S on **test** only.
- **Ablation 08:** optional YOLO train on FloW train split (not wired until train layout exists).
- **Ablation 09:** add FloW column to transfer table when 04 completes.
