# Water debris segmentation — decision log

Append-only. High-impact choices that affect benchmark comparability.

| Date | Decision | Rationale | Benchmark impact |
|---|---|---|---|
| 2026-10-05 | EoMT/VidEoMT mask training **LaRS only**; FloW/CANSURF/WSODD **eval-only** via mask→box mAP | Box sets lack pixel GT; rectangle/SAM GT would fake segmentation | All ablations 04–06; no `fit` on box splits |
| 2026-10-05 | Scaffold docs + `eval_outputs/water_debris_seg/` before data download | OpenFly-style gates: disk + GPU inventory first | Stage 0 complete; Stage 1 blocked until slack + GPU |
