// Builds docs/presentation/water_debris_benchmark_status.pptx
// Source of numbers: eval_outputs/water_debris_seg/REPORT.md and RESCORE.md (updated 2026-10-13, pycocotools)
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const THEME = {
  name: "Open Water",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "14232B", lt1: "FFFFFF", dk2: "0B3C49", lt2: "EEF4F5",
    accent1: "0B6E79", accent2: "F28C28", accent3: "7FB8BE",
    accent4: "5B6B73", accent5: "C9DDE0", accent6: "B23A48",
    hlink: "0B6E79", folHlink: "5B6B73",
  },
};
const H = THEME.colors; // hex for hex-only options

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "Water-surface debris segmentation — benchmark status";
pres.author = "rcampanella";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

const FOOT = "Water-debris seg benchmark · status as of 2026-10-13";

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.6, fontSize: 44, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.0, w: 11.7, h: 1.2, fontSize: 20, color: C.accent3, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 32, bold: true, color: C.text2, valign: "middle", align: "left" }, text: "" } },
    { text: { text: FOOT, options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent4, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
});

let n = 0;
const sec = (title) => pres.addSection({ title });
const add = (master, section) => pres.addSlide({ masterName: master, sectionTitle: section });
const box = (o) => ({ isTextBox: true, ...o, objectName: o.objectName || `obj${++n}` });
const shadow = () => ({ type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 90 });

// card with tinted background
function card(slide, x, y, w, h, fill = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { type: "none" }, rectRadius: 0.12, shadow: shadow(), objectName: `card${++n}` });
}
// circular badge (motif)
function badge(slide, x, y, d, label, fill, fontSize = 16) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" }, objectName: `badge${++n}` });
  slide.addText(label, box({ x, y, w: d, h: d, align: "center", valign: "middle", fontSize, bold: true, color: C.background1, margin: 0 }));
}

const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt", legendFontFace: "+mn-lt" };
const chartBase = () => ({
  ...chartText,
  catAxisLabelColor: H.accent4, valAxisLabelColor: H.accent4, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  dataLabelFontSize: 11, dataLabelColor: H.dk1, showValue: true, dataLabelPosition: "outEnd",
  valGridLine: { color: "DDE6E8", size: 0.75 }, catGridLine: { style: "none" },
  titleColor: H.dk2, titleFontSize: 14, showTitle: true,
  legendFontSize: 12, legendColor: H.dk1,
});

// ---------- 1. Title ----------
sec("Overview");
let s = add("TITLE_DARK", "Overview");
s.addText("Water-Surface Debris Segmentation Benchmark", { placeholder: "title" });
s.addText("Status update for the research team · results through 13 Oct 2026 · box metrics restated", { placeholder: "body" });
badge(s, 0.8, 1.0, 0.9, "v1", H.accent2, 20);
s.addNotes("Status update on the water-debris segmentation benchmark. Numbers come from eval_outputs/water_debris_seg/REPORT.md, last updated 2026-10-13. Box metrics were re-scored with pycocotools on 13 Oct; see the correction slide.");

// ---------- 2. Bottom line ----------
s = add("CONTENT", "Overview");
s.addText("Bottom line: v1 is essentially complete, and YOLO still wins on debris", { placeholder: "title" });
const stats = [
  ["8 / 8", "planned ablations done", "06 (WSODD) cancelled for scope; YOLO on FloW finished 12 Oct", C.accent1],
  ["0.395", "LaRS panoptic PQ", "after LaRS fine-tune, up from 0.298 zero-shot (same label space)", C.accent1],
  ["0.095 vs 0.857", "CANSURF mAP", "best EoMT mask→box vs in-domain YOLOv11n; FloW 0.082 vs 0.390", C.accent2],
  ["No gain", "from VidEoMT", "flicker and keyframe AP both worse than per-frame EoMT", C.accent6],
];
stats.forEach(([big, label, sub, col], i) => {
  const x = 0.6 + i * 3.1, y = 1.8, w = 2.85, h = 3.9;
  card(s, x, y, w, h);
  s.addText(big, box({ x: x + 0.25, y: y + 0.4, w: w - 0.5, h: 1.2, fontSize: big.length > 6 ? 34 : 48, bold: true, color: col, fontFace: THEME.headFontFace, margin: 0, valign: "bottom" }));
  s.addText(label, box({ x: x + 0.25, y: y + 1.75, w: w - 0.5, h: 0.5, fontSize: 18, bold: true, color: C.text1, margin: 0 }));
  s.addText(sub, box({ x: x + 0.25, y: y + 2.35, w: w - 0.5, h: 1.9, fontSize: 14, color: C.accent4, margin: 0, valign: "top" }));
});
s.addNotes("Four takeaways. Scope: all 8 runnable ablations have results; WSODD was dropped on 2026-10-07 because Baidu hosting is impractical. LaRS fine-tuning works for segmentation. Cross-dataset detection via mask-to-box stays 5 to 9 times below a YOLO trained in-domain (pycocotools numbers, re-scored 13 Oct). VidEoMT did not improve temporal stability or keyframe AP at the schedules tried.");

// ---------- 2b. Metric correction ----------
s = add("CONTENT", "Overview");
s.addText("Correction: box metrics restated after a scorer fix on 13 Oct", { placeholder: "title" });
card(s, 0.6, 1.55, 4.3, 5.05, C.background2);
badge(s, 0.85, 1.8, 0.8, "!", H.accent6, 24);
s.addText([
  { text: "What was wrong", options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } },
  { text: "\"mAP\" averaged only IoU 0.50 and 0.75", options: { bullet: true, breakLine: true } },
  { text: "AP_small never counted false positives", options: { bullet: true, breakLine: true } },
  { text: "Images with no predictions were skipped", options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
  { text: "Now: pycocotools, matches Ultralytics (0.390 vs 0.391 on FloW). Old files kept as *.legacy.json; audit table in RESCORE.md.", options: { fontSize: 13, color: C.accent4 } },
], box({ x: 0.85, y: 2.75, w: 3.85, h: 3.7, fontSize: 14, color: C.text1, margin: 0, valign: "top" }));
const corrRows = [
  ["Run", "mAP old → new", "AP_small old → new"],
  ["FloW · EoMT zero-shot", "0.086 → 0.034", "0.168 → 0.012"],
  ["FloW · EoMT semantic FT", "0.133 → 0.082", "0.425 → 0.044"],
  ["FloW · EoMT panoptic FT", "0.040 → 0.014", "0.130 → 0.007"],
  ["FloW · YOLOv11n (new run)", "0.566 → 0.390", "0.858 → 0.249"],
  ["CANSURF · EoMT zero-shot", "0.147 → 0.095", "n/a (19 small GT)"],
  ["CANSURF · EoMT semantic FT", "0.085 → 0.063", "0.632 → 0.001"],
  ["CANSURF · EoMT panoptic FT", "0.120 → 0.034", "0.133 → 0.001"],
  ["CANSURF · YOLOv11n", "0.861 → 0.857", "n/a (19 small GT)"],
];
s.addTable(corrRows.map((r, i) => r.map((t) => ({ text: t, options: { bold: i === 0, color: i === 0 ? C.background1 : C.text1, fill: { color: i === 0 ? C.text2 : (i % 2 ? C.background1 : C.background2) } } }))), {
  x: 5.2, y: 1.55, w: 7.5, colW: [3.3, 2.1, 2.1], fontSize: 13, rowH: 0.5, border: { type: "solid", pt: 0.5, color: H.accent5 }, valign: "middle", objectName: "corrTable",
});
s.addNotes("The repo's box scorer (eval_coco_bbox.py) was hand-written on 6 Oct instead of using pycocotools as the plan specified. The YOLO FloW run exposed it: 0.566 from our scorer vs 0.391 from Ultralytics. All FloW/CANSURF box numbers were re-scored from the saved prediction files, no model reruns. The ranking of methods is unchanged; the gap to YOLO is larger, and the earlier small-object result for semantic FT was entirely a scorer artefact. Every row is traceable in eval_outputs/water_debris_seg/RESCORE.md with file hashes.");

// ---------- 3. Setup ----------
sec("Setup");
s = add("CONTENT", "Setup");
s.addText("Two protocols: train masks on LaRS only, score box sets via mask→box", { placeholder: "title" });
// left: research question
card(s, 0.6, 1.55, 4.3, 5.0, C.background2);
s.addText([
  { text: "Mission", options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } },
  { text: "Nearshore USV camera: find surface debris for pickup without confusing boats, buoys, swimmers, water or sky.", options: { fontSize: 14, breakLine: true, paraSpaceAfter: 12 } },
  { text: "Research question", options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } },
  { text: "Can general mask transformers (EoMT, VidEoMT) match specialized detectors (YOLO) on debris, and does temporal modelling help on USV video?", options: { fontSize: 14 } },
], box({ x: 0.85, y: 1.8, w: 3.8, h: 4.5, valign: "top", color: C.text1, margin: 0 }));
// right: protocol flow
const flowY = [1.55, 4.15];
const rows = [
  { tag: "LaRS", desc: "Pixel panoptic + semantic GT, 9-frame sequences", train: "EoMT / VidEoMT: train + eval", metric: "PQ · mIoU · flicker", col: H.accent1 },
  { tag: "FloW\nCANSURF", desc: "Box-only GT (waste, cans)", train: "EoMT: eval only (masks → boxes)\nYOLO: train + eval", metric: "mAP · AP_small", col: H.accent2 },
];
rows.forEach((r, i) => {
  const y = flowY[i];
  card(s, 5.3, y, 7.4, 2.4, C.background1);
  badge(s, 5.55, y + 0.55, 1.3, r.tag, r.col, r.tag.includes("\n") ? 12 : 16);
  s.addText(r.desc, box({ x: 7.1, y: y + 0.25, w: 5.4, h: 0.5, fontSize: 15, bold: true, color: C.text1, margin: 0 }));
  s.addText(r.train, box({ x: 7.1, y: y + 0.8, w: 5.4, h: 0.8, fontSize: 14, color: C.text1, margin: 0, valign: "top" }));
  s.addText(r.metric, box({ x: 7.1, y: y + 1.7, w: 5.4, h: 0.45, fontSize: 14, italic: true, color: C.accent1, margin: 0 }));
});
s.addNotes("Hard rule: only LaRS has pixel ground truth, so it is the only place EoMT/VidEoMT masks are trained. Training on rectangle or SAM-filled masks from box datasets would just teach the model to paint boxes. So FloW and CANSURF are eval-only for the segmenters: masks become axis-aligned boxes and are scored with COCO mAP. YOLO can train on boxes because that is its native GT.");

// ---------- 4. Progress tracker ----------
s = add("CONTENT", "Setup");
s.addText("Progress: every planned ablation now has a result", { placeholder: "title" });
const abl = [
  ["01", "Zero-shot EoMT", "Done", "LaRS PQ + box bridge"],
  ["02", "LaRS semantic FT", "Done", "mIoU 98.0; attribute slices pending"],
  ["03", "LaRS panoptic FT", "Done", "PQ 0.395"],
  ["04", "FloW mask→box", "Done", "mAP ZS 0.034 · pano FT 0.014"],
  ["05", "CANSURF mask→box", "Done", "mAP ZS 0.095 · pano FT 0.034"],
  ["06", "WSODD distractors", "Cancelled", "Baidu hosting; out of scope v1"],
  ["07", "VidEoMT temporal", "Done", "4k + 16k schedules"],
  ["08", "YOLOv11 baselines", "Done", "mAP CANSURF 0.857 · FloW 0.390"],
  ["09", "Cross-dataset transfer", "Done", "semantic best EoMT on FloW (0.082)"],
];
const stCol = { Done: H.accent1, Cancelled: H.accent4, Partial: H.accent2 };
abl.forEach(([id, name, st, note], i) => {
  const col = i % 3, row = Math.floor(i / 3);
  const x = 0.6 + col * 4.1, y = 1.55 + row * 1.75, w = 3.85, h = 1.5;
  card(s, x, y, w, h);
  badge(s, x + 0.2, y + 0.3, 0.9, id, stCol[st], 18);
  s.addText(name, box({ x: x + 1.3, y: y + 0.15, w: w - 1.45, h: 0.45, fontSize: 15, bold: true, color: C.text1, margin: 0 }));
  s.addText(st, box({ x: x + 1.3, y: y + 0.6, w: w - 1.45, h: 0.3, fontSize: 12, bold: true, color: st === "Done" ? C.accent1 : st === "Partial" ? C.accent2 : C.accent4, margin: 0 }));
  s.addText(note, box({ x: x + 1.3, y: y + 0.9, w: w - 1.45, h: 0.5, fontSize: 12, color: C.accent4, margin: 0, valign: "top" }));
});
s.addNotes("Badge colour = status: teal done, orange partial, grey cancelled. YOLOv11 on FloW finished on 12 Oct (job 27942882), so every runnable ablation has a result. Ablation 02 attribute slices (glare, waves) are an optional extra.");

// ---------- 5. LaRS segmentation ----------
sec("Results");
s = add("CONTENT", "Results");
s.addText("On LaRS, zero-shot EoMT is already decent; fine-tuning adds +0.10 PQ", { placeholder: "title" });
const lars = [
  ["0.298", "PQ", "Zero-shot COCO EoMT", "predictions mapped into LaRS classes (was 0.001: bridge bug)", C.accent4],
  ["0.395", "PQ", "Panoptic FT, 11 classes", "native LaRS label space", C.accent1],
  ["98.0", "mIoU", "Semantic FT, 3 classes", "obstacle / water / sky", C.accent1],
];
lars.forEach(([big, unit, title, sub, col], i) => {
  const x = 0.6 + i * 4.1, w = 3.85;
  card(s, x, 1.6, w, 3.3);
  s.addText([{ text: big, options: { fontSize: 54, bold: true, color: col } }, { text: "  " + unit, options: { fontSize: 20, color: C.accent4 } }],
    box({ x: x + 0.3, y: 1.85, w: w - 0.6, h: 1.2, fontFace: THEME.headFontFace, margin: 0, valign: "bottom" }));
  s.addText(title, box({ x: x + 0.3, y: 3.3, w: w - 0.6, h: 0.5, fontSize: 17, bold: true, color: C.text1, margin: 0 }));
  s.addText(sub, box({ x: x + 0.3, y: 3.8, w: w - 0.6, h: 0.8, fontSize: 14, color: C.accent4, margin: 0, valign: "top" }));
});
card(s, 0.6, 5.25, 12.1, 1.35, C.background2);
s.addText([
  { text: "Read with care: ", options: { bold: true, color: C.accent6 } },
  { text: "boats are as good zero-shot as fine-tuned (PQ 0.39 vs 0.40); the gain comes from shoreline, swimmers and classes COCO lacks. The debris class Float is 0 for both, and the fine-tuned model finds only 14% of small objects." },
], box({ x: 0.9, y: 5.4, w: 11.5, h: 1.05, fontSize: 15, color: C.text1, margin: 0, valign: "middle" }));
s.addNotes("Ablations 01–03, all in native LaRS 11-class space. Zero-shot COCO predictions are mapped many-to-one into LaRS classes (configs/coco133_to_lars_pred_map.json, job 28011322); the same script reproduces the fine-tuned model's EoMT validate PQ (0.3946 vs 0.3945). The earlier 0.0014 came from a GT bridge that mapped boats to spoon and water to food-other. Per class, zero-shot to fine-tuned: boat 0.391 to 0.395, static obstacle 0.78 to 0.89, water 0.98 to 0.98, swimmer 0.06 to 0.20; row boats, buoy and float have no COCO class. Thing recall from qualitative/lars_val_things.csv: 388/1107 overall, 96/698 small.");

// ---------- 6. Box transfer mAP ----------
s = add("CONTENT", "Results");
s.addText("As a detector, LaRS-trained EoMT stays far below in-domain YOLO", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "FloW mAP", labels: ["Zero-shot", "Semantic FT", "Panoptic FT", "YOLOv11n"], values: [0.034, 0.082, 0.014, 0.390] },
  { name: "CANSURF mAP", labels: ["Zero-shot", "Semantic FT", "Panoptic FT", "YOLOv11n"], values: [0.095, 0.063, 0.034, 0.857] },
], {
  ...chartBase(), x: 0.6, y: 1.45, w: 8.0, h: 5.3, barDir: "col", barGapWidthPct: 60,
  title: "COCO mAP@[.50:.95], pycocotools (EoMT: mask→box; YOLO: native boxes)",
  chartColors: [H.accent1, H.accent2], showLegend: true, legendPos: "t",
  valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0", dataLabelFormatCode: "0.000",
});
card(s, 9.0, 1.6, 3.7, 5.0);
s.addText([
  { text: "What it says", options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } },
  { text: "Best EoMT is 0.095 on CANSURF and 0.082 on FloW: about 9× and 5× below YOLO.", options: { bullet: true, breakLine: true, paraSpaceAfter: 8 } },
  { text: "Semantic FT helps on FloW (recall 0.28 vs 0.09), but panoptic FT hurts versus zero-shot on both sets.", options: { bullet: true, breakLine: true, paraSpaceAfter: 8 } },
  { text: "FloW is hard even for YOLO (0.390): most FloW objects are small.", options: { bullet: true } },
], box({ x: 9.25, y: 1.85, w: 3.25, h: 4.5, fontSize: 14, color: C.text1, margin: 0, valign: "top" }));
s.addNotes("Ablations 04, 05, 08, 09, all scored with pycocotools on the same GT files. EoMT boxes carry a fixed score of 0.5, so AP cannot rank their hits above their misses and is pessimistic for EoMT; recall (AR100) is the fairer comparison: FloW 0.09/0.28/0.05 for zero-shot/semantic/panoptic vs 0.49 for YOLO. Semantic FT on CANSURF with a 256-pixel minimum-area filter reaches 0.078. CANSURF GT is the aligned version with 12 corrupt val images dropped.");

// ---------- 7. Small objects ----------
s = add("CONTENT", "Results");
s.addText("Small debris is the real failure: EoMT finds almost none on FloW", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "AP_small", labels: ["Zero-shot", "Semantic FT", "Panoptic FT", "YOLOv11n"], values: [0.012, 0.044, 0.007, 0.249] },
  { name: "Recall small (AR_small)", labels: ["Zero-shot", "Semantic FT", "Panoptic FT", "YOLOv11n"], values: [0.032, 0.158, 0.024, 0.388] },
], {
  ...chartBase(), x: 4.7, y: 1.45, w: 8.0, h: 5.3, barDir: "col", barGapWidthPct: 60,
  title: "FloW test, GT boxes < 32² px (1,199 of 2,024 boxes)",
  chartColors: [H.accent1, H.accent2], showLegend: true, legendPos: "t",
  valAxisMinVal: 0, valAxisMaxVal: 0.5, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0", dataLabelFormatCode: "0.000",
});
card(s, 0.6, 1.6, 3.7, 5.0);
badge(s, 0.9, 1.9, 0.8, "!", H.accent6, 24);
s.addText([
  { text: "Why FloW only", options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } },
  { text: "CANSURF has just 19 small boxes out of 11,526, so its small-object AP is noise. FloW is the small-object test.", options: { breakLine: true, paraSpaceAfter: 8 } },
  { text: "The earlier 0.63 small-object AP for semantic FT was a scorer defect, not a model effect.", options: { italic: true } },
], box({ x: 0.9, y: 2.9, w: 3.1, h: 3.5, fontSize: 14, color: C.text1, margin: 0, valign: "top" }));
s.addNotes("Far-field debris is the operational hard case. On FloW, 59 percent of GT boxes are smaller than 32x32 pixels. EoMT semantic FT recalls 16 percent of them, YOLO 39 percent at 640 px input; FloW images are 1280x720, so YOLO sees them at half resolution. A 1280 px YOLO run would test whether resolution is the bottleneck.");

// ---------- 8. Temporal ----------
s = add("CONTENT", "Results");
s.addText("VidEoMT does not beat per-frame EoMT on temporal stability", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Flicker", labels: ["EoMT semantic (ab02)", "EoMT panoptic (ab03)", "VidEoMT 16k", "VidEoMT 4k"], values: [0.041, 0.229, 0.311, 0.612] },
], {
  ...chartBase(), x: 0.6, y: 1.45, w: 7.6, h: 5.3, barDir: "bar", barGapWidthPct: 50,
  title: "Flicker = 1 − mean consecutive-frame mIoU (lower is better)",
  chartColors: [H.accent1], showLegend: false, catAxisOrientation: "maxMin",
  valAxisMinVal: 0, valAxisMaxVal: 0.7, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0", dataLabelFormatCode: "0.000",
});
const tstats = [
  ["0.016", "VidEoMT 4k keyframe AP", C.accent2],
  ["0.009", "VidEoMT 16k keyframe AP", C.accent6],
];
tstats.forEach(([big, label, col], i) => {
  const y = 1.6 + i * 1.75;
  card(s, 8.6, y, 4.1, 1.5);
  s.addText(big, box({ x: 8.85, y: y + 0.2, w: 1.7, h: 1.1, fontSize: 36, bold: true, color: col, fontFace: THEME.headFontFace, margin: 0, valign: "middle" }));
  s.addText(label, box({ x: 10.6, y: y + 0.2, w: 1.95, h: 1.1, fontSize: 14, color: C.text1, margin: 0, valign: "middle" }));
});
card(s, 8.6, 5.1, 4.1, 1.5, C.background2);
s.addText("Longer training halves flicker but lowers keyframe AP. Proxies differ: VidEoMT uses an 8-class thing mIoU, EoMT a 3-class semantic one.",
  box({ x: 8.85, y: 5.2, w: 3.6, h: 1.3, fontSize: 13, color: C.text1, margin: 0, valign: "middle" }));
s.addNotes("Ablation 07, scored on 35 LaRS val clips. Per-frame EoMT semantic is already very stable (mean consecutive mIoU ~0.959). LaRS has no ground-truth track IDs, so ID-switch metrics are not available; flicker is the stability proxy. VidEoMT keyframe AP is not directly comparable to EoMT PQ 0.395.");

// ---------- 9. Caveats ----------
sec("Wrap-up");
s = add("CONTENT", "Wrap-up");
s.addText("Caveats to keep in mind when quoting these numbers", { placeholder: "title" });
const cav = [
  ["Zero-shot PQ restated", "0.001 → 0.298 after a label-mapping bug; now same LaRS space as fine-tuned."],
  ["Fixed box score", "EoMT mask→box uses score 0.5; compare ranks only within one pipeline."],
  ["Small objects", "CANSURF has only 19 small GT boxes; make small-object claims on FloW."],
  ["Flicker proxies", "VidEoMT 8-class thing mIoU vs EoMT 3-class semantic mIoU."],
  ["Dataset scope", "WSODD out of v1; FloW needs ORCA approval; CANSURF drops 12 corrupt val images."],
  ["Scorer change", "Box numbers before 13 Oct are superseded; audit trail in RESCORE.md."],
];
cav.forEach(([h, t], i) => {
  const col = i % 2, row = Math.floor(i / 2);
  const x = 0.6 + col * 6.15, y = 1.55 + row * 1.7, w = 5.95, h2 = 1.45;
  card(s, x, y, w, h2);
  badge(s, x + 0.25, y + 0.3, 0.85, String(i + 1), H.accent1, 18);
  s.addText(h, box({ x: x + 1.35, y: y + 0.15, w: w - 1.55, h: 0.45, fontSize: 16, bold: true, color: C.text1, margin: 0 }));
  s.addText(t, box({ x: x + 1.35, y: y + 0.6, w: w - 1.55, h: 0.75, fontSize: 14, color: C.accent4, margin: 0, valign: "top" }));
});
s.addNotes("These are the comparability caveats from REPORT.md and the decision log. Fine-tuned checkpoints were copied from scratch to home on 13 Oct, so the purge risk is resolved.");

// ---------- 10. Next steps ----------
s = add("TITLE_DARK", "Wrap-up");
s.addText("Next steps and questions for the team", { placeholder: "title" });
const steps = [
  ["1", "Qualitative error analysis: error panels and mask overlays (in progress)"],
  ["2", "Decide whether VidEoMT merits a longer schedule or panoptic video training"],
  ["3", "YOLO on FloW at 1280 px: is input resolution the small-object bottleneck?"],
  ["4", "Discuss: is mask→box worth pursuing, or do we pair a segmenter with a detector?"],
];
steps.forEach(([k, t], i) => {
  const y = 4.15 + i * 0.75;
  badge(s, 0.8, y, 0.55, k, H.accent2, 14);
  s.addText(t, box({ x: 1.6, y, w: 10.9, h: 0.55, fontSize: 17, color: C.background1, margin: 0, valign: "middle" }));
});
s.addNotes("Proposed next steps. Item 4 is the main discussion point: the gap to in-domain YOLO is large, so the team should decide whether general mask transformers stay the primary detector or become a scene-context module alongside a specialised detector.");

(async () => {
  const out = path.join(__dirname, "water_debris_benchmark_status.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
