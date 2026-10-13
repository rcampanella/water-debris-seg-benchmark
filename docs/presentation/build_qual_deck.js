// Builds docs/presentation/water_debris_qualitative_review.pptx
// All numbers come from qual_assets/deck_data.json (written by qual_figures.py from the audited
// analysis outputs); every slide footer names its source files.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const A = path.join(__dirname, "qual_assets");
const D = JSON.parse(fs.readFileSync(path.join(A, "deck_data.json"), "utf8"));
const Q = "eval_outputs/water_debris_seg/qualitative";

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
const H = THEME.colors;
// error-status colours, identical to the figures (qual_render_panels.COL)
const ST = { tp: "2CA02C", merged: "C5B0D5", loc: "FFBB78", missed: "FF7F0E" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Water-surface debris detection — qualitative error review";
pres.author = "rcampanella";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.9, w: 11.7, h: 1.9, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.0, w: 11.7, h: 1.4, fontSize: 20, color: C.accent3, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.2, fontSize: 36, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.7, h: 1.2, fontSize: 18, color: C.accent3, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.95, fontSize: 28, bold: true, color: C.text2, valign: "middle", align: "left" }, text: "" } },
    { placeholder: { options: { name: "source", type: "body", x: 0.6, y: 6.98, w: 11.4, h: 0.4, fontSize: 10, color: C.accent4, valign: "top" }, text: "" } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
});

let n = 0;
const box = (o) => ({ isTextBox: true, ...o, objectName: o.objectName || `obj${++n}` });
const shadow = () => ({ type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 90 });
function card(s, x, y, w, h, fill = C.background2) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { type: "none" }, rectRadius: 0.1, shadow: shadow(), objectName: `card${++n}` });
}
function badge(s, x, y, d, label, fill, fs = 16) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" }, objectName: `badge${++n}` });
  s.addText(label, box({ x, y, w: d, h: d, align: "center", valign: "middle", fontSize: fs, bold: true, color: C.background1, margin: 0 }));
}
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
// place a figure inside a box, preserving aspect; returns the placed rectangle
function fig(s, name, x, y, maxW, maxH, alt) {
  const file = path.join(A, `${name}.png`);
  const { w, h } = pngSize(file);
  let W = maxW, Hh = maxW * h / w;
  if (Hh > maxH) { Hh = maxH; W = maxH * w / h; }
  const X = x + (maxW - W) / 2;
  s.addImage({ path: file, x: X, y, w: W, h: Hh, altText: alt, objectName: name });
  return { x: X, y, w: W, h: Hh };
}
const figSrc = (name) => D.figures[name].footer;
const content = (title, section, source) => {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  s.addText(source, { placeholder: "source" });
  return s;
};
const pct = (v) => `${Math.round(v * 100)}%`;
const f3 = (v) => v.toFixed(3);
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt", legendFontFace: "+mn-lt" };
const chartBase = () => ({
  ...chartText, catAxisLabelColor: H.accent4, valAxisLabelColor: H.accent4, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  dataLabelFontSize: 11, dataLabelColor: H.dk1, showValue: true, valGridLine: { color: "DDE6E8", size: 0.75 },
  catGridLine: { style: "none" }, titleColor: H.dk2, titleFontSize: 14, showTitle: true, legendFontSize: 12, legendColor: H.dk1,
});
const bullets = (items, o = {}) => items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8, ...o } }));
const O = D.outcome, M = D.box_metrics;

// ======================================================= Overview
pres.addSection({ title: "Overview" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Overview" });
s.addText("Where and why debris detection fails", { placeholder: "title" });
s.addText("Qualitative error review of the water-debris segmentation benchmark · inputs, outputs and failure mechanisms · 13 Oct 2026", { placeholder: "body" });
s.addNotes("Purpose: show, image by image, how each method fails, so the next design changes target the right mechanism. Every figure is drawn from the scored prediction files; every slide footer names its sources. Companion to the status deck (water_debris_benchmark_status.pptx).");

s = content("Four findings to discuss", "Overview", `Sources: ${D.sources.outcome}; ${D.sources.lars_things}; ${D.sources.lars_pq}`);
const sem = O.flow_eomt_semantic, csem = O.cansurf_eomt_semantic_aligned;
const sz = D.lars_things_by_size;
const findings = [
  ["1", "Two different failures", `Instance heads (zero-shot, panoptic FT) never fire on most debris: ${pct(O.flow_eomt_zeroshot.fn_missed)}–${pct(O.cansurf_eomt_panoptic.fn_missed)} of objects missed.`, H.accent6],
  ["2", "The semantic path merges", `It sees debris but fuses neighbours: ${pct(csem.fn_merged)} of CANSURF cans and ${pct(sem.fn_merged)} of FloW bottles end inside multi-object boxes.`, H.accent2],
  ["3", "Small objects are lost", `On LaRS, the panoptic model finds ${pct(sz.small.found / sz.small.n)} of small objects vs ${pct(sz.large.found / sz.large.n)} of large ones; ${pct(sz.small.n / (sz.small.n + sz.medium.n + sz.large.n))} of objects are small.`, H.accent1],
  ["4", "LaRS has almost no debris", `${D.lars_things_by_class.float.n} "Float" instances in LaRS val, none found; boats gain nothing from fine-tuning (PQ ${f3(D.lars_pq.zeroshot["Boat/ship"])} → ${f3(D.lars_pq.finetuned["Boat/ship"])}).`, H.accent4],
];
findings.forEach(([k, head, body, col], i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 1.5 + Math.floor(i / 2) * 2.7;
  card(s, x, y, 5.95, 2.45);
  badge(s, x + 0.3, y + 0.35, 0.8, k, col, 20);
  s.addText(head, box({ x: x + 1.35, y: y + 0.3, w: 4.4, h: 0.5, fontSize: 19, bold: true, color: C.text1, margin: 0 }));
  s.addText(body, box({ x: x + 1.35, y: y + 0.85, w: 4.4, h: 1.45, fontSize: 15, color: C.text1, margin: 0, valign: "top" }));
});
s.addNotes("The deck walks through these four in order, then turns them into design changes. Shares are of ground-truth objects at IoU 0.5, using pycocotools matching, so they reconcile with the metrics files.");

s = content("How to read this deck", "Overview", `Definitions: water-debris-seg/water_debris_seg/qualitative/errors.py · selection rules: ${Q}/selection.json · index: ${Q}/INDEX.md, MASKS.md`);
card(s, 0.6, 1.45, 6.0, 5.3);
s.addText([
  { text: "How images were chosen", options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } },
  ...bullets([
    "6 uniform-random images per dataset (seed 0) as an unbiased reference",
    "Plus the worst images per error type and method, by fixed rules",
    "Plus images where YOLO and EoMT disagree most, both ways",
    "Every pick records its rule and counts in selection.json",
  ]),
], box({ x: 0.85, y: 1.65, w: 5.5, h: 2.9, fontSize: 15, color: C.text1, margin: 0, valign: "top" }));
s.addText([
  { text: "Operating point and audit", options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } },
  ...bullets([
    "YOLO boxes with score ≥ 0.25; EoMT boxes all kept (fixed score 0.5)",
    `Dumped masks reproduce the scored boxes on ${D.consistency}/${D.consistency_total} image-model pairs`,
  ]),
], box({ x: 0.85, y: 4.6, w: 5.5, h: 2.0, fontSize: 15, color: C.text1, margin: 0, valign: "top" }));
const tax = [
  ["Colour", "Ground-truth object outcome", "Detection outcome"],
  ["green", "found (IoU ≥ 0.5)", "true positive"],
  ["lilac / purple", "merged into a multi-object box", "box covers ≥ 2 objects"],
  ["light orange / pink", "poorly localised (IoU 0.1–0.5)", "poorly localised"],
  ["orange / red", "missed (IoU < 0.1)", "background (overlaps nothing)"],
];
const taxFill = [H.dk2, "D9F0D9", "ECE4F3", "FFF0DE", "FFE3D0"];
s.addTable(tax.map((r, i) => r.map((t) => ({ text: t, options: { bold: i === 0, color: i === 0 ? C.background1 : C.text1, fill: { color: taxFill[i] } } }))), {
  x: 6.9, y: 1.45, w: 5.8, colW: [1.6, 2.2, 2.0], fontSize: 13, rowH: 0.62, border: { type: "solid", pt: 0.5, color: H.accent5 }, valign: "middle", objectName: "taxTable",
});
s.addText("Missed and merged objects are drawn dashed; detections solid. \"Fragmented\" (one object split into several boxes) is under 1% everywhere and is grouped with poorly localised.", box({ x: 6.9, y: 4.75, w: 5.8, h: 1.2, fontSize: 14, color: C.accent4, margin: 0, valign: "top" }));
s.addNotes("Matching is pycocotools' own (COCOeval evalImgs, IoU 0.5, at most 100 detections per image), so 'found' equals recall at IoU 0.5 in the metrics files. Merge is checked before duplicate and localisation, so an object swallowed by a larger box is counted as merged, not as poorly localised.");

s = content("Where in the pipeline each failure is created", "Overview", "Pipeline: water-debris-seg/scripts/eval_eomt_cansurf_mask2box.py (panoptic), eval_eomt_cansurf_semantic_mask2box.py (semantic) · box sets are eval-only for EoMT");
const steps = [
  ["Image", "FloW / CANSURF still"],
  ["EoMT", "zero-shot COCO, or LaRS fine-tuned"],
  ["Mask", "object segments (panoptic) or 'obstacle' pixels (semantic)"],
  ["Boxes", "one box per segment, or per connected blob"],
  ["Match", "vs GT boxes, IoU 0.5"],
];
steps.forEach(([h, t], i) => {
  const x = 0.6 + i * 2.5;
  card(s, x, 1.6, 2.15, 1.7, i === 2 || i === 3 ? C.background2 : C.background1);
  s.addText(h, box({ x: x + 0.15, y: 1.75, w: 1.85, h: 0.45, fontSize: 18, bold: true, color: C.text2, margin: 0 }));
  s.addText(t, box({ x: x + 0.15, y: 2.2, w: 1.85, h: 1.0, fontSize: 13, color: C.text1, margin: 0, valign: "top" }));
  if (i < steps.length - 1) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.18, y: 2.3, w: 0.28, h: 0.3, fill: { color: H.accent3 }, line: { type: "none" }, objectName: `arrow${i}` });
});
const where = [
  ["1", "Not recognised", "Instance heads output no segment for debris (no COCO class, ~no LaRS floats). Created at step 2.", H.accent6, 0.6 + 2.5],
  ["2", "Merged", "Semantic mask has no instance boundaries; touching objects form one blob. Created at step 4.", H.accent2, 0.6 + 3 * 2.5],
  ["3", "Shoreline boxes", "'Obstacle' includes static shore, so the shoreline becomes one huge box. Created at step 3.", H.accent1, 0.6 + 2 * 2.5],
];
where.forEach(([k, h, t, col], i) => {
  const y = 3.75 + i * 1.05;
  badge(s, 0.75, y + 0.1, 0.6, k, col, 16);
  s.addText([{ text: `${h}: `, options: { bold: true } }, { text: t }], box({ x: 1.55, y, w: 10.9, h: 0.85, fontSize: 15, color: C.text1, margin: 0, valign: "middle" }));
});
s.addNotes("The hard rule (train EoMT masks on LaRS only; box datasets are eval-only) is why debris never gets direct mask supervision. Failure 1 is a recognition problem, failure 2 an instance-separation problem, failure 3 a class-definition problem. They need different fixes.");

// ======================================================= Box datasets
pres.addSection({ title: "Box datasets" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Box datasets" });
s.addText("FloW and CANSURF: what happens to each object", { placeholder: "title" });
s.addText("Outcome shares, an unbiased sample, then one slide per failure mechanism", { placeholder: "body" });

s = content("What happens to every ground-truth object", "Box datasets", `Source: ${D.sources.outcome} (IoU 0.5; YOLO score ≥ 0.25). FloW test 2,024 objects; CANSURF val 11,526 objects`);
const order = { flow_test: ["flow_eomt_zeroshot", "flow_eomt_panoptic", "flow_eomt_semantic", "flow_yolo11n"],
  cansurf_val_aligned: ["cansurf_eomt_zeroshot", "cansurf_eomt_panoptic", "cansurf_eomt_semantic_aligned", "cansurf_yolo11n"] };
[["flow_test", "FloW (inland bottles)"], ["cansurf_val_aligned", "CANSURF (cans)"]].forEach(([ds, label], i) => {
  const runs = order[ds];
  const labels = runs.map((r) => O[r].method);
  const series = [
    { name: "found", values: runs.map((r) => O[r].tp) },
    { name: "merged", values: runs.map((r) => O[r].fn_merged) },
    { name: "poorly localised", values: runs.map((r) => O[r].fn_loc + O[r].fn_frag) },
    { name: "missed", values: runs.map((r) => O[r].fn_missed) },
  ].map((x) => ({ ...x, labels }));
  s.addChart(pres.charts.BAR, series, {
    ...chartBase(), x: 0.5 + i * 6.25, y: 1.4, w: 6.1, h: 4.35, barDir: "bar", barGrouping: "stacked", barGapWidthPct: 45,
    title: label, chartColors: [ST.tp, ST.merged, ST.loc, ST.missed], showLegend: i === 0, legendPos: "b",
    dataLabelPosition: "ctr", dataLabelFormatCode: '[>=0.07]0%;[<0.07]"";', catAxisOrientation: "maxMin",
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0%", objectName: `outcome_${ds}`,
  });
});
card(s, 0.6, 5.9, 12.1, 0.95, C.background2);
s.addText([
  { text: "Read across: ", options: { bold: true } },
  { text: `zero-shot and panoptic FT mostly miss (orange); semantic FT mostly finds or merges (green + lilac), missing only ${pct(sem.fn_missed)} on FloW and ${pct(csem.fn_missed)} on CANSURF. YOLO finds ${pct(O.flow_yolo11n.tp)} and ${pct(O.cansurf_yolo11n.tp)}.` },
], box({ x: 0.85, y: 5.95, w: 11.6, h: 0.85, fontSize: 15, color: C.text1, margin: 0, valign: "middle" }));
s.addNotes("This chart is the map for the rest of the section. Two EoMT variants fail at recognition; the semantic variant fails at separation. Merged objects count as misses for mAP, which is why semantic FT's mAP looks so low despite its coverage.");

s = content("An unbiased sample: what a typical FloW image looks like", "Box datasets", `${figSrc("fig_random_flow")} · picked uniformly at random (seed 0), not by error · bottom row = zoom (yellow box)`);
let r = fig(s, "fig_random_flow", 0.6, 1.3, 12.1, 4.75, "FloW test image 000042 with zoom on the far bottles: ground truth and four methods' boxes coloured by error type");
s.addText([
  { text: "What it shows: ", options: { bold: true } },
  { text: "YOLO finds all 7 bottles. Zero-shot and panoptic EoMT leave most bottles unmarked (dashed orange). Semantic FT finds 4 but adds a box along the whole shoreline (red line): failure 3 shows up even in a random image." },
], box({ x: 0.6, y: r.y + r.h + 0.15, w: 12.1, h: 6.9 - (r.y + r.h + 0.15), fontSize: 15, color: C.text1, margin: 0, valign: "top" }));
s.addNotes("Five more random FloW images and six random CANSURF images are in qualitative/panels/*/0[1-6]_random_*.png; they show the same pattern. Starting from a random image guards against a gallery of worst cases.");

s = content("Failure 1: instance heads never fire on small floating bottles", "Box datasets", `${figSrc("fig_missed_flow")} · rule: most small objects never detected by zero-shot EoMT · bottom row = zoom (yellow box)`);
r = fig(s, "fig_missed_flow", 0.6, 1.3, 12.1, 4.4, "FloW 000657 with zoom: zero-shot and panoptic EoMT miss most bottles; YOLO finds 10 of 15");
s.addText([
  { text: "What it shows: ", options: { bold: true } },
  { text: `zero-shot finds 2 of 15 and panoptic FT 1 of 15; the cluster in the zoom is entirely orange (never detected). YOLO finds 10, but merges some neighbours too (purple). FloW-wide, these two EoMT variants miss ${pct(O.flow_eomt_zeroshot.fn_missed)} and ${pct(O.flow_eomt_panoptic.fn_missed)} of objects.` },
], box({ x: 0.6, y: r.y + r.h + 0.15, w: 12.1, h: 6.9 - (r.y + r.h + 0.15), fontSize: 15, color: C.text1, margin: 0, valign: "top" }));
s.addNotes("Mechanism: the COCO head has no 'floating waste' class, and LaRS fine-tuning saw only 3 floats in val (and few in train). There is no class for these masks to be assigned to, so they are dropped before boxing.");

s = content("Failure 2: the semantic path finds bottles but fuses each row into one box", "Box datasets", `${figSrc("fig_merge_flow_zoom")} · rule: most objects merged by semantic FT`);
r = fig(s, "fig_merge_flow_zoom", 0.6, 1.35, 9.1, 5.5, "FloW 000212 with zoom: semantic FT boxes span whole rows of bottles");
card(s, 9.85, 1.35, 2.85, 5.45);
s.addText([
  { text: "What it shows", options: { bold: true, fontSize: 17, color: C.text2, breakLine: true } },
  { text: `Semantic FT: 3 found, 13 missed, almost all purple (one box per row of bottles).`, options: { breakLine: true, paraSpaceAfter: 10 } },
  { text: `The same image in YOLO: 11 found.`, options: { breakLine: true, paraSpaceAfter: 10 } },
  { text: `FloW-wide: ${pct(sem.fn_merged)} of bottles merged; on CANSURF ${pct(csem.fn_merged)}.`, options: {} },
], box({ x: 10.05, y: 1.55, w: 2.5, h: 5.1, fontSize: 15, color: C.text1, margin: 0, valign: "top" }));
s.addNotes("Merged objects count as misses for mAP even though the model has detected them as 'obstacle'. This is the main reason semantic FT has the best EoMT recall but still a low mAP.");

s = content("Why it merges: the mask has no object boundaries", "Box datasets", `${figSrc("fig_merge_flow_mask")}`);
r = fig(s, "fig_merge_flow_mask", 0.6, 1.35, 12.1, 3.75, "FloW 000212: semantic obstacle mask in yellow with derived boxes, next to YOLO");
s.addText([
  ...bullets([
    "The yellow 'obstacle' mask covers the bottles correctly, but touching bottles form one connected blob, and one blob becomes one box.",
    "The same mask also covers buildings and shore above the waterline: one image-wide box (red line at the horizon).",
    "Fix options: split blobs (distance transform / watershed), or add an instance head trained on LaRS objects; drop blobs above the water boundary.",
  ]),
], box({ x: 0.6, y: r.y + r.h + 0.25, w: 12.1, h: 6.85 - (r.y + r.h + 0.25), fontSize: 15, color: C.text1, margin: 0, valign: "top" }));
s.addNotes("These are the exact masks behind the scored numbers: boxes re-derived from them match the prediction file for this image. Mask file: qualitative/masks/flow_test/flow_eomt_semantic/000212.npz.");

s = content("On CANSURF the whole scene can become one 'obstacle'", "Box datasets", `${figSrc("fig_merge_cansurf_mask")}`);
r = fig(s, "fig_merge_cansurf_mask", 0.6, 1.3, 12.1, 3.75, "CANSURF 375_dog: zero-shot masks, semantic mask covering the full image, YOLO finds all 13 cans");
s.addText([
  ...bullets([
    "Semantic FT labels the entire frame (cans and fabric) as obstacle: 13 cans become one image-sized box.",
    "Zero-shot EoMT produces no object mask at all here (13 of 13 missed); YOLO finds all 13.",
    "This is a close-up of cans on a pet bed, not a water scene; CANSURF val mixes such scenes with water footage.",
  ]),
], box({ x: 0.6, y: r.y + r.h + 0.25, w: 12.1, h: 6.85 - (r.y + r.h + 0.25), fontSize: 15, color: C.text1, margin: 0, valign: "top" }));
s.addNotes(`CANSURF-wide, ${pct(csem.fn_merged)} of cans are merged by semantic FT and only ${pct(csem.fn_missed)} are missed outright. The semantic model was trained to call anything that is not water or sky an obstacle, so a frame with no water is all obstacle.`);

s = content("Failure 3: shoreline and vegetation become large false boxes", "Box datasets", `${figSrc("fig_shore_flow_mask")} · rule: most background false positives for semantic FT`);
r = fig(s, "fig_shore_flow_mask", 0.6, 1.35, 12.1, 3.75, "FloW 000778: semantic obstacle mask covers wall and plants, producing red background boxes");
s.addText([
  ...bullets([
    "The wall, reeds and floating plants are all 'obstacle' in LaRS terms, so they produce red background boxes; the single bottle is found.",
    `Background boxes are ${pct(D.det.flow_eomt_semantic.fp_bkg / (D.det.flow_eomt_semantic.fp_bkg + D.det.flow_eomt_semantic.fp_loc + D.det.flow_eomt_semantic.fp_merge + D.det.flow_eomt_semantic.fp_dup))} of semantic-FT false positives on FloW (${D.det.flow_eomt_semantic.fp_bkg.toLocaleString("en-US")} boxes).`,
    "Fix options: keep only blobs whose base touches water; a learned water-boundary or horizon prior.",
  ]),
], box({ x: 0.6, y: r.y + r.h + 0.25, w: 12.1, h: 6.85 - (r.y + r.h + 0.25), fontSize: 15, color: C.text1, margin: 0, valign: "top" }));
s.addNotes("In LaRS, static obstacle is the right label for this region: for a USV it is something not to hit. For debris pickup it is noise. The class definition, not the model, is what produces these boxes.");

s = content("Found but loose: recall at IoU 0.5 is much closer to YOLO than mAP suggests", "Box datasets", `Sources: ${D.sources.box_metrics}; ${D.sources.outcome}. FloW test`);
const fr = ["flow_eomt_zeroshot", "flow_eomt_panoptic", "flow_eomt_semantic", "flow_yolo11n"];
const fl = fr.map((x) => O[x].method);
s.addChart(pres.charts.BAR, [
  { name: "found at IoU 0.5", labels: fl, values: fr.map((x) => O[x].recall50) },
  { name: "AP@0.75", labels: fl, values: fr.map((x) => M[x].AP75) },
  { name: "mAP@[.5:.95]", labels: fl, values: fr.map((x) => M[x].mAP) },
], {
  ...chartBase(), x: 0.6, y: 1.4, w: 8.2, h: 5.4, barDir: "col", barGapWidthPct: 55, title: "FloW: coverage vs box quality",
  chartColors: [H.accent1, H.accent3, H.accent2], showLegend: true, legendPos: "t", dataLabelPosition: "outEnd",
  dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0", objectName: "looseChart",
});
card(s, 9.1, 1.55, 3.6, 5.1);
s.addText([
  { text: "What it says", options: { bold: true, fontSize: 17, color: C.text2, breakLine: true } },
  { text: `Semantic FT finds ${pct(O.flow_eomt_semantic.recall50)} of bottles vs YOLO ${pct(O.flow_yolo11n.recall50)}, a 1.4× gap.`, options: { breakLine: true, paraSpaceAfter: 10 } },
  { text: `On mAP the gap is ${(M.flow_yolo11n.mAP / M.flow_eomt_semantic.mAP).toFixed(1)}×: merged boxes, loose boxes (AP@0.75 ${f3(M.flow_eomt_semantic.AP75)}) and a fixed score of 0.5 that cannot rank detections.`, options: { breakLine: true, paraSpaceAfter: 10 } },
  { text: "Separation and scoring are the cheap wins.", options: { bold: true } },
], box({ x: 9.35, y: 1.75, w: 3.15, h: 4.8, fontSize: 15, color: C.text1, margin: 0, valign: "top" }));
s.addNotes("Recall at IoU 0.5 here is at the operating point (YOLO score at least 0.25). mAP and AP@0.75 are from RESCORE.json, scored by pycocotools on all predictions.");

s = content("The ceiling fails too: YOLO misses all four cans in a pool scene", "Box datasets", `${figSrc("fig_yolo_fail_cansurf")} · rule: lowest YOLO recall (≥ 2 objects)`);
r = fig(s, "fig_yolo_fail_cansurf", 0.6, 1.35, 12.1, 3.75, "CANSURF pool frame: YOLO misses 4 of 4 cans, EoMT semantic finds all 4");
s.addText([
  ...bullets([
    "Indoor pool with a waterfall: YOLO detects none of the 4 cans; EoMT semantic FT finds all 4 (plus one false box).",
    "All three lowest-recall YOLO images come from this one video (IMG_3619): an out-of-distribution scene for a CANSURF-trained detector.",
    "One video, so anecdotal: it suggests EoMT's water/obstacle prior can hold where an in-domain detector breaks. Worth a targeted out-of-distribution test.",
  ]),
], box({ x: 0.6, y: r.y + r.h + 0.25, w: 12.1, h: 6.85 - (r.y + r.h + 0.25), fontSize: 15, color: C.text1, margin: 0, valign: "top" }));
s.addNotes("This is the counterexample to the headline gap, and it matters for a USV: the deployment domain will not match any training set exactly. Panels 34 to 36 in qualitative/panels/cansurf_val_aligned/ show the same video.");

// ======================================================= LaRS
pres.addSection({ title: "LaRS pixel level" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "LaRS pixel level" });
s.addText("LaRS: what fine-tuning learned, and what it did not", { placeholder: "title" });
s.addText("Pixel ground truth: per-class PQ, small objects, conditions, and the debris class", { placeholder: "body" });

s = content("Fine-tuning helps shoreline and COCO-absent classes, not boats", "LaRS pixel level", `Source: ${D.sources.lars_pq} (job 28011322; both in native LaRS 11-class space)`);
const cls = ["Water", "Sky", "Static Obstacle", "Boat/ship", "Row boats", "Paddle board", "Swimmer", "Buoy", "Animal", "Other", "Float"];
s.addChart(pres.charts.BAR, [
  { name: `zero-shot (PQ ${f3(D.lars_pq.all.zeroshot)})`, labels: cls, values: cls.map((c) => D.lars_pq.zeroshot[c]) },
  { name: `LaRS fine-tuned (PQ ${f3(D.lars_pq.all.finetuned)})`, labels: cls, values: cls.map((c) => D.lars_pq.finetuned[c]) },
], {
  ...chartBase(), x: 0.6, y: 1.4, w: 8.6, h: 5.4, barDir: "bar", barGapWidthPct: 40, title: "Panoptic quality per LaRS class",
  chartColors: [H.accent3, H.accent1], showLegend: true, legendPos: "t", dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
  catAxisOrientation: "maxMin", valAxisMinVal: 0, valAxisMaxVal: 1.1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0.0", dataLabelFontSize: 10, objectName: "pqChart",
});
card(s, 9.5, 1.55, 3.2, 5.1);
s.addText([
  { text: "What it says", options: { bold: true, fontSize: 17, color: C.text2, breakLine: true } },
  { text: "Water and sky are solved zero-shot.", options: { breakLine: true, paraSpaceAfter: 8 } },
  { text: `Boat PQ is unchanged (${f3(D.lars_pq.zeroshot["Boat/ship"])} → ${f3(D.lars_pq.finetuned["Boat/ship"])}).`, options: { breakLine: true, paraSpaceAfter: 8 } },
  { text: "Gains: shoreline, swimmers, and row boats / buoys that COCO has no class for.", options: { breakLine: true, paraSpaceAfter: 8 } },
  { text: "Float stays at 0.", options: { bold: true } },
], box({ x: 9.75, y: 1.75, w: 2.8, h: 4.8, fontSize: 15, color: C.text1, margin: 0, valign: "top" }));
s.addNotes("Zero-shot COCO predictions are mapped many-to-one into LaRS classes (configs/coco133_to_lars_pred_map.json); row boats, buoy and float have no COCO counterpart, so zero-shot is 0 there by construction. The scoring script reproduces EoMT's own validate PQ for the fine-tuned model (0.3946 vs 0.3945). The earlier zero-shot 0.0014 came from a label-bridge bug.");

s = content("Small objects: the fine-tuned model finds 1 in 7", "LaRS pixel level", `Sources: ${D.sources.lars_things} (object found = mask IoU > 0.5) · ${figSrc("fig_lars_small")}`);
const sizes = ["small", "medium", "large"];
r = fig(s, "fig_lars_small", 0.6, 1.3, 12.1, 3.6, "LaRS inhouse_seq_191_00399: semantic errors and zoomed panoptic objects vs 21 small GT objects, none found");
const yb = r.y + r.h + 0.15;
s.addChart(pres.charts.BAR, [{
  name: "found", labels: sizes.map((z) => `${z} (${sz[z].n})`), values: sizes.map((z) => sz[z].found / sz[z].n),
}], {
  ...chartBase(), x: 0.6, y: yb, w: 4.8, h: 6.9 - yb, barDir: "bar", barGapWidthPct: 40, title: "Panoptic FT: LaRS objects found, by size",
  chartColors: [H.accent1], showLegend: false, dataLabelPosition: "outEnd", dataLabelFormatCode: "0%", catAxisOrientation: "maxMin",
  valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0%", titleFontSize: 12, objectName: "sizeChart",
});
s.addText([
  { text: "Example: ", options: { bold: true } },
  { text: `${D.lars_small_example.objects} distant objects on a hazy horizon; the panoptic model finds ${D.lars_small_example.found_panoptic}. Yet the semantic model labels ${pct(D.lars_small_example.obstacle_px_share)} of their pixels as obstacle (${D.lars_small_example.objects_touched} of ${D.lars_small_example.objects} touched): the signal is there at pixel level, but no object masks come out. Small = under 32² px; ${pct(sz.small.n / (sz.small.n + sz.medium.n + sz.large.n))} of LaRS val objects are small.` },
], box({ x: 5.7, y: yb + 0.05, w: 7.0, h: 6.85 - yb, fontSize: 14, color: C.text1, margin: 0, valign: "top" }));
s.addNotes("PQ 0.395 is carried by stuff classes (water, sky, shore around 0.9). Object recall is 388 of 1,107 overall. Debris in FloW is 59 percent small, so this is the size regime that matters for pickup.");

s = content("Conditions: dusk, fog and wakes hurt most, but the samples are small", "LaRS pixel level", `Source: ${D.sources.lars_conditions} (LaRS image attributes) · ${figSrc("fig_lars_glitter")}`);
const condOrder = ["all images", "dusk", "night", "fog", "wakes", "glitter", "extra dark", "extra bright"];
const condRows = [["Condition", "Images", "Obstacle IoU (semantic)", "Objects found (panoptic)"]].concat(
  condOrder.map((k) => { const c = D.lars_conditions[k]; return [k, String(c.n), f3(c.obstacle_iou), `${c.found}/${c.things} (${pct(c.found / c.things)})`]; }));
s.addTable(condRows.map((row, i) => row.map((t, j) => ({ text: t, options: { bold: i === 0 || (i === 1), color: i === 0 ? C.background1 : C.text1, fill: { color: i === 0 ? H.dk2 : i % 2 ? "FFFFFF" : H.lt2 }, align: j === 0 ? "left" : "center" } }))), {
  x: 0.6, y: 1.4, w: 6.3, colW: [1.7, 1.0, 1.8, 1.8], fontSize: 13, rowH: 0.48, border: { type: "solid", pt: 0.5, color: H.accent5 }, valign: "middle", objectName: "condTable",
});
s.addText("Strata of 5–26 images: treat differences as hypotheses to test, not effects.", box({ x: 0.6, y: 5.85, w: 6.3, h: 0.6, fontSize: 14, italic: true, color: C.accent4, margin: 0 }));
r = fig(s, "fig_lars_glitter", 7.2, 1.3, 5.5, 4.55, "LaRS glitter example at dusk: semantic errors on the horizon line, few objects found");
s.addText("Glitter at dusk: semantic errors cluster on the horizon line (red); the panoptic model finds 2 of 10 small objects on the sun path (bottom, zoomed).", box({ x: 7.2, y: r.y + r.h + 0.1, w: 5.5, h: 6.9 - (r.y + r.h + 0.1), fontSize: 14, color: C.text1, margin: 0, valign: "top" }));
s.addNotes("Per-image values are in lars_val_per_image.csv; the worst image for each attribute is rendered in qualitative/mask_panels/lars_val/.");

s = content("LaRS barely contains debris, so LaRS-only training cannot learn it", "LaRS pixel level", `Source: ${D.sources.lars_things} · object found = panoptic FT mask IoU > 0.5`);
const cl = D.lars_things_by_class;
const clOrder = ["boat/ship", "buoy", "other", "row boat", "paddle board", "animal", "swimmer", "float"];
const clRows = [["LaRS val class", "Objects", "Found", "Share"]].concat(clOrder.map((k) => [k, String(cl[k].n), String(cl[k].found), pct(cl[k].found / cl[k].n)]));
s.addTable(clRows.map((row, i) => row.map((t, j) => ({ text: t, options: { bold: i === 0 || row[0] === "float", color: i === 0 ? C.background1 : row[0] === "float" ? C.accent6 : C.text1, fill: { color: i === 0 ? H.dk2 : i % 2 ? "FFFFFF" : H.lt2 }, align: j === 0 ? "left" : "center" } }))), {
  x: 0.6, y: 1.4, w: 6.3, colW: [2.4, 1.3, 1.3, 1.3], fontSize: 14, rowH: 0.5, border: { type: "solid", pt: 0.5, color: H.accent5 }, valign: "middle", objectName: "classTable",
});
card(s, 7.2, 1.4, 5.5, 4.55);
s.addText([
  { text: "Implication for the hard rule", options: { bold: true, fontSize: 17, color: C.text2, breakLine: true } },
  ...bullets([
    `Only ${cl.float.n} 'Float' objects in LaRS val; ${pct(cl["boat/ship"].n / clOrder.reduce((a, k) => a + cl[k].n, 0))} of objects are boats.`,
    "'Other' (the closest thing to waste) is found 8% of the time.",
    "Training masks only on LaRS optimises for boats; debris shape and appearance are never supervised.",
    "Options: pixel-label a FloW/CANSURF subset (SAM-assisted, human-checked), or relax the rule for an instance head.",
  ]),
], box({ x: 7.45, y: 1.6, w: 5.05, h: 4.2, fontSize: 15, color: C.text1, margin: 0, valign: "top" }));
s.addNotes("This is the root cause behind failure 1. The hard rule was right to forbid training masks on rectangles, but it leaves debris with no mask supervision at all. A small, carefully checked pixel-labelled debris set would remove the constraint without faking segmentation.");

// ======================================================= Caveats
pres.addSection({ title: "Caveats" });
s = content("Data and evaluation caveats behind these images", "Caveats", "Sources: RESCORE.md; decision log 2026-10-13; qualitative/README.md (finding 7); data/flow and data/cansurf file listings");
const cav = [
  ["CANSURF is not all water", "Val mixes water footage with synthetic sprites and indoor close-ups; augmentations repeat a source up to 9×. Train/val overlap is negligible."],
  [`CANSURF has ${D.gt_sizes.cansurf_val_aligned.small} small objects`, `Out of ${D.gt_sizes.cansurf_val_aligned.n.toLocaleString("en-US")}. Small-object claims rest on FloW (${D.gt_sizes.flow_test.small.toLocaleString("en-US")} of ${D.gt_sizes.flow_test.n.toLocaleString("en-US")} small).`],
  ["Fixed EoMT score", "Every EoMT box scores 0.5, so AP cannot rank its detections and is pessimistic; recall is the fairer EoMT signal until real mask scores are exported."],
  ["Two metric bugs fixed on 13 Oct", "Box scorer (now pycocotools) and the LaRS→COCO label bridge (zero-shot PQ 0.0014 → 0.298). Old values kept for audit; tests now guard both."],
  ["YOLO FloW at half resolution", "1280×720 images trained at 640 px for parity with CANSURF; a 1280 px run may raise the small-object ceiling."],
  ["Selection is rule-based", "Worst-case panels are chosen by fixed rules and shown next to random samples; counts on every chart cover all images, not the gallery."],
];
cav.forEach(([h, t], i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 1.4 + Math.floor(i / 2) * 1.8;
  card(s, x, y, 5.95, 1.6);
  badge(s, x + 0.25, y + 0.4, 0.75, String(i + 1), H.accent1, 18);
  s.addText(h, box({ x: x + 1.25, y: y + 0.15, w: 4.5, h: 0.4, fontSize: 16, bold: true, color: C.text1, margin: 0 }));
  s.addText(t, box({ x: x + 1.25, y: y + 0.55, w: 4.5, h: 0.95, fontSize: 13, color: C.accent4, margin: 0, valign: "top" }));
});
s.addNotes("Caveat 4 is the one reviewers should know about: both bugs lived in untracked, agent-written evaluation code. Both are fixed, cross-checked against reference implementations, and guarded by tests in water-debris-seg/tests/.");

// ======================================================= Implications
pres.addSection({ title: "Implications" });
s = content("From failure mode to design change", "Implications", "Evidence columns refer to the slides above; counts from error_summary.csv and lars_val_things.csv");
const dc = [
  ["Failure", "Evidence", "Proposed change", "Metric to watch"],
  ["Merged objects (semantic)", `${pct(csem.fn_merged)} CANSURF, ${pct(sem.fn_merged)} FloW merged`, "Split blobs (distance transform / watershed); or class-agnostic instance head trained on LaRS objects", "share merged; FloW mAP"],
  ["Shoreline boxes", "Image-wide red boxes, even in random samples", "Keep blobs touching water; water-boundary / horizon prior", "background FPs per image"],
  ["Not recognised (instance heads)", `${pct(O.flow_eomt_zeroshot.fn_missed)} FloW missed zero-shot`, "Pixel-label a small debris set (FloW/CANSURF); add a debris class", "FloW recall, AP_small"],
  ["Small objects", `${pct(sz.small.found / sz.small.n)} of small LaRS objects found`, "Higher input resolution or tiling; test YOLO at 1280 px too", "AP_small (FloW), small recall"],
  ["Unrankable boxes", "Fixed score 0.5", "Export mask confidence as box score", "AP vs recall gap"],
];
s.addTable(dc.map((row, i) => row.map((t, j) => ({ text: t, options: { bold: i === 0 || j === 0, color: i === 0 ? C.background1 : C.text1, fill: { color: i === 0 ? H.dk2 : i % 2 ? "FFFFFF" : H.lt2 } } }))), {
  x: 0.6, y: 1.4, w: 12.1, colW: [2.4, 2.8, 4.5, 2.4], fontSize: 13, rowH: 0.8, border: { type: "solid", pt: 0.5, color: H.accent5 }, valign: "middle", objectName: "designTable",
});
s.addNotes("Ordered by expected gain per unit of effort: blob splitting and the water-contact filter are post-processing changes that need no retraining and can be evaluated on the saved masks directly.");

s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Implications" });
s.addText("Proposed next experiments and questions", { placeholder: "title" });
const nx = [
  ["1", "Post-process the saved semantic masks: split blobs + water-contact filter (no retraining)"],
  ["2", "Export real mask scores; re-score EoMT so AP reflects ranking"],
  ["3", "YOLO on FloW at 1280 px; EoMT with tiling for small objects"],
  ["Q", "Should we relax the LaRS-only rule for a small, checked debris pixel set?"],
  ["Q", "Is the target a detector, or a scene model (water/shore/obstacle) that gates a detector?"],
];
nx.forEach(([k, t], i) => {
  const y = 4.35 + i * 0.58;
  badge(s, 0.8, y, 0.45, k, k === "Q" ? H.accent6 : H.accent2, 13);
  s.addText(t, box({ x: 1.45, y: y - 0.04, w: 11.0, h: 0.52, fontSize: 16, color: C.background1, margin: 0, valign: "middle" }));
});
s.addNotes("Experiments 1 and 2 can run on existing outputs within a day. The two questions decide the direction of v2: the pool example shows the scene prior has real value even where it loses on mAP.");

// ======================================================= Appendix
pres.addSection({ title: "Appendix" });
s = content("Appendix: provenance and how to regenerate", "Appendix", "All paths relative to the repository root unless absolute");
s.addTable([
  ["What", "Where"],
  ["Error tables per run", `${Q}/errors/<run_id>/{detections,gt,summary}.csv/json (summary has pred + GT sha256)`],
  ["Image selection rules", `${Q}/selection.json`],
  ["All panels (78 box, 92 mask)", `${Q}/INDEX.md, MASKS.md`],
  ["Saved masks + consistency", `${Q}/masks/<dataset>/<run_id>/*.npz, consistency.json (${D.consistency}/${D.consistency_total} identical)`],
  ["Box metrics audit", "eval_outputs/water_debris_seg/RESCORE.md / .json"],
  ["LaRS PQ (both models)", "ablations/01_zeroshot_lars/sources/pq_*_28011322.json"],
  ["Slide figures + numbers", "docs/presentation/qual_figures.py → qual_assets/ (deck_data.json lists each figure's sources)"],
].map((row, i) => row.map((t) => ({ text: t, options: { bold: i === 0, color: i === 0 ? C.background1 : C.text1, fill: { color: i === 0 ? H.dk2 : i % 2 ? "FFFFFF" : H.lt2 } } }))), {
  x: 0.6, y: 1.4, w: 12.1, colW: [3.0, 9.1], fontSize: 12, rowH: 0.45, border: { type: "solid", pt: 0.5, color: H.accent5 }, valign: "middle", objectName: "provTable",
});
s.addText([
  { text: "Regenerate: ", options: { bold: true } },
  { text: "rescore_box_predictions.py → qual_error_analysis.py → qual_render_panels.py → qual_dump_masks.sbatch (GPU) → qual_render_masks.py → docs/presentation/qual_figures.py → build_qual_deck.js", options: { fontFace: "Courier New", fontSize: 12 } },
], box({ x: 0.6, y: 5.35, w: 12.1, h: 0.9, fontSize: 14, color: C.text1, margin: 0, valign: "top" }));
s.addNotes("Every figure in this deck is listed in qual_assets/deck_data.json with its image path, prediction files and sha256 prefixes.");

(async () => {
  const out = path.join(__dirname, "water_debris_qualitative_review.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
