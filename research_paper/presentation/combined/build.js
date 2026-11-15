// Build the combined deck: node build.js [out.pptx]  ->  research_analyses.pptx
// Cover, then four parts in this order, each after a section divider:
//   1 consensus      ../consensus/      s_intro.js, s_results.js  + assets/numbers.json, fig/
//   2 cross-dataset  ../cross_dataset/  s_intro.js, s_results.js, s_end.js  + assets/facts.json, analysis_report/figures/
//   3 intensity      ../intensity/      slides.js + data.js (output/intensity_analysis.json), analysis_report/figures/
//   4 venues         ../venues/         slides.js (docs/venues.md)
// Nothing is copied: every part runs its own slide modules with its own lib.js (style, page numbers, font-size
// tracking), so the slides look exactly as in the separate decks. All four libs draw on the same pptxgenjs object;
// none of them uses a slide master, so there is nothing to merge. Each lib.js takes the page counters reached so far
// (init(pptx, { slideNo, appendixNo })), so page numbers run on across the parts: 1, 2, 3, ... for the slides in the
// show, A1, A2, ... for the hidden appendix slides, which stay at the end of their own part.
// Rebuild the parts first when their data change (each part's rebuild.sh), then run this build.
const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const V = require("../venues/lib");              // cover, dividers and the venue part
const LC = require("../consensus/lib");
const LX = require("../cross_dataset/lib");
const LI = require("../intensity/lib");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Chen Lequn";
pptx.title = "SER benchmark: three analyses and target venues";
pptx.subject = "Model consensus, cross-dataset differences, emotion intensity (public datasets only) and target venues";

// Run one block of slides with a lib, continuing the page counters of the blocks before it
let at = { slideNo: 0, appendixNo: 0 };
function run(L, build) {
  const from = { ...at };
  L.init(pptx, at);
  build(L);
  at = { slideNo: L.slideNo, appendixNo: L.appendixNo };
  return { first: from.slideNo + 1, last: at.slideNo, shown: at.slideNo - from.slideNo,
    hidden: at.appendixNo - from.appendixNo, a0: from.appendixNo + 1, a1: at.appendixNo };
}
const range = (r) => (r.hidden ? `A${r.a0}–A${r.a1}` : "none");

// ---- Cover (its contents table is added once the page ranges are known)
const { C, MX, CW } = V;
let cover;
run(V, (L) => {
  cover = L.blank();
  cover.addShape(pptx.ShapeType.rect, { x: MX, y: 0.75, w: 0.09, h: 2.55, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
  L.text(cover, "Speech Emotion Recognition Benchmark:\nThree Analyses and Target Venues", MX + 0.4, 0.65, 11.8, 1.45, { fontSize: 40, bold: true, ls: 0.95 });
  L.text(cover, "Model consensus  ·  cross-dataset differences  ·  emotion intensity  ·  where to submit", MX + 0.4, 2.2, 11.8, 0.5, { fontSize: 24 });
  L.text(cover, "Chen Lequn  ·  research_paper  ·  November 2026", MX + 0.4, 2.85, 11.8, 0.42, { fontSize: 20 });
});

// ---- Section divider: dark navy slide, white text, no page number (like the title slides)
const PARTS = [
  { title: "Model consensus", sub: "When several models agree on a clip, how often is the agreed label right?",
    note: "This part asks whether agreement between models can select reliable pseudo-labels." },
  { title: "Cross-dataset differences", sub: "Language, conversational versus acted speech, and clip duration",
    note: "This part asks why the same models do well on some datasets and poorly on others: language, conversational versus acted speech, or clip duration." },
  { title: "Emotion intensity", sub: "Is a stronger emotional expression easier to classify?",
    note: "This part asks whether clips with a stronger emotion label are classified correctly more often." },
  { title: "Target venues", sub: "Shortlist (ACL, EMNLP, Interspeech, IEEE ASLI 2027) and additional candidates",
    note: "This part summarises the venue shortlist of research_paper/docs/venues.md: ACL 2027, EMNLP 2027, Interspeech 2027 and IEEE ASLI 2027, then the additional candidates checked on 14 November 2026 (NeurIPS 2027, ICASSP 2028, ACM MM 2027, AAAI-28, EACL) and the review windows." },
];
function divider(k) {
  let s;
  run(V, (L) => {
    s = L.blank({ background: C.ink });
    s.addShape(pptx.ShapeType.rect, { x: MX, y: 2.05, w: 0.09, h: 2.75, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
    L.text(s, `PART ${k + 1} OF ${PARTS.length}`, MX + 0.4, 2.0, 11.8, 0.42, { fontSize: 20, bold: true, color: C.white, extra: { charSpacing: 2 } });
    L.text(s, PARTS[k].title, MX + 0.4, 2.5, 11.8, 0.85, { fontSize: 40, bold: true, color: C.white });
    L.text(s, PARTS[k].sub, MX + 0.4, 3.45, 11.8, 0.5, { fontSize: 24, color: C.white });
  });
  return s;
}
function label(s, k, r) {                        // page range on the divider, its notes
  const app = r.hidden ? `; its ${r.hidden} hidden appendix slides (${range(r)}) follow at the end of the part` : "";
  V.text(s, `Slides ${r.first}–${r.last}${r.hidden ? `  ·  hidden appendix ${range(r)} at the end of this part` : ""}`, MX + 0.4, 4.3, 11.8, 0.42, { fontSize: 20, color: C.white });
  V.notes(s, `Part ${k + 1} of ${PARTS.length}: ${PARTS[k].title.toLowerCase()}. ${PARTS[k].note} It runs from slide ${r.first} to slide ${r.last}${app}.`);
}

// ---- Part 1: consensus (same steps as ../consensus/build.js)
const d1 = divider(0);
const r1 = run(LC, (L) => {
  const N = JSON.parse(fs.readFileSync(path.join(ROOT, "consensus", "assets", "numbers.json"), "utf8"));
  require("../consensus/s_intro")(L, N);
  require("../consensus/s_results")(L, N);
});

// ---- Part 2: cross-dataset (same helpers as ../cross_dataset/build.js; keep them in sync)
const d2 = divider(1);
const warnings = [];
const r2 = run(LX, (L) => {
  const D = JSON.parse(fs.readFileSync(path.join(ROOT, "cross_dataset", "assets", "facts.json"), "utf8"));
  const F = (k) => { if (!(k in D.facts)) throw new Error(`unknown fact ${k}`); return D.facts[k].text; };
  const Vf = (k) => { if (!(k in D.facts)) throw new Error(`unknown fact ${k}`); return D.facts[k].value; };
  const list = (a) => (a.length <= 1 ? a.join("") : `${a.slice(0, -1).join(", ")} and ${a[a.length - 1]}`);
  const signed = (v, d = 1) => { if (v === null || v === undefined) return "—"; let t = (v >= 0 ? "+" : "") + v.toFixed(d); if (Number(t) === 0) t = (0).toFixed(d); return t.replace("-", "−"); };
  const claim = (ok, message) => { if (!ok) warnings.push(message); };
  const XH = { D, F, V: Vf, list, signed, claim, gemmaPending: !Vf("gemma_complete") };
  require("../cross_dataset/s_intro")(L, XH);
  require("../cross_dataset/s_results")(L, XH);
  require("../cross_dataset/s_end")(L, XH);
});

// ---- Part 3: intensity (same steps as ../intensity/build.js)
const d3 = divider(2);
const intensityData = require("../intensity/data");
require("../intensity/write_facts")(intensityData);
const r3 = run(LI, (L) => require("../intensity/slides")(L, intensityData));

// ---- Part 4: venues (same steps as ../venues/build.js)
const d4 = divider(3);
const r4 = run(V, (L) => require("../venues/slides")(L));

// ---- Page ranges on the dividers and the cover
const R = [r1, r2, r3, r4];
[d1, d2, d3, d4].forEach((s, k) => label(s, k, R[k]));
V.table(cover, ["Part", "Content", "Slides", "Hidden appendix"], [
  ["1", "Model consensus as a pseudo-label filter"],
  ["2", "Cross-dataset differences: language, setting, clip duration"],
  ["3", "Emotion intensity: are stronger emotions easier to classify?"],
  ["4", "Target venues: shortlist and additional candidates"],
].map(([n, t], k) => [{ text: n, bold: true, align: "center" }, t, { text: `${R[k].first}–${R[k].last}`, align: "center" }, { text: range(R[k]), align: "center" }]),
{ y: 3.65, colW: [0.9, 7.1, 1.7, 2.43], fontSize: 20 });
V.text(cover, "Complete benchmark: 10 models across 16 public corpora. Analyses use saved predictions.", MX, 6.45, CW, 0.42, { fontSize: 20 });
V.notes(cover, `This deck puts the three analyses of our speech emotion recognition benchmark and the venue shortlist into one presentation. Part 1 asks whether agreement between models can select reliable pseudo-labels. Part 2 asks why the same models do well on some datasets and poorly on others. Part 3 asks whether clips with a stronger emotion label are classified correctly more often. Part 4 summarises where we could submit the paper: the shortlist, the additional candidates and the review windows.
Each part starts with a divider and keeps its own title slide; the hidden appendix slides of each part come at the end of that part and are skipped in the slide show. Page numbers run on across the whole deck.`);

// ---- Checks: every lib's font-size guard
let ok = true;
for (const [name, L] of [["consensus", LC], ["cross-dataset", LX], ["intensity", LI], ["cover, dividers, venues", V]]) {
  process.stdout.write(`${name}: `);
  ok = L.checkSizes() && ok;
}
if (!ok) { console.error("font size check failed"); process.exit(1); }
warnings.forEach((w) => console.log(`  CHECK WORDING (cross-dataset part; slide numbers of its own deck): ${w}`));

// pptxgenjs writes one <a:pPr> per text run, so a paragraph with several runs (bold + normal text) gets repeated
// paragraph properties, which the OOXML schema does not allow (LibreOffice then drops the bullet).
// Keep only the first <a:pPr> of every paragraph.
function cleanParagraphs(xml) {
  const pPr = /<a:pPr\b[^>]*\/>|<a:pPr\b[^>]*>[\s\S]*?<\/a:pPr>/g;
  let fixed = 0;
  const out = xml.replace(/<a:p>([\s\S]*?)<\/a:p>/g, (m, body) => {
    let seen = false;
    const b = body.replace(pPr, (q) => { if (seen) { fixed += 1; return ""; } seen = true; return q; });
    return `<a:p>${b}</a:p>`;
  });
  return [out, fixed];
}

// Second guard on the written slides themselves: smallest font anywhere >= 16 pt, in tables >= 18 pt
function xmlSizes(name, xml) {
  const sizes = (x) => [...x.matchAll(/<a:(?:rPr|defRPr|endParaRPr)\b[^>]*\bsz="(\d+)"/g)].map((m) => Number(m[1]) / 100);
  const tables = [...xml.matchAll(/<a:tbl>[\s\S]*?<\/a:tbl>/g)].map((m) => m[0]);
  return { name, all: Math.min(...sizes(xml)), table: Math.min(...tables.flatMap(sizes)) };
}

const out = process.argv[2] || "research_analyses.pptx";
(async () => {
  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(await pptx.write({ outputType: "nodebuffer" }));
  let total = 0;
  const found = [];
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/(slides|notesSlides)\/[^/]+\.xml$/.test(n))) {
    const [xml, fixed] = cleanParagraphs(await zip.file(name).async("string"));
    total += fixed;
    zip.file(name, xml);
    if (name.startsWith("ppt/slides/")) found.push(xmlSizes(name, xml));
  }
  const tooSmall = found.filter((f) => f.all < 16 || f.table < 18);
  const minAll = Math.min(...found.map((f) => f.all)), minTable = Math.min(...found.map((f) => f.table));
  console.log(`slide XML: smallest text ${minAll} pt, smallest table text ${minTable} pt over ${found.length} slides`);
  tooSmall.forEach((f) => console.log(`  TOO SMALL in ${f.name}: ${f.all} pt (table ${f.table} pt)`));
  if (tooSmall.length) { console.error("font size check failed"); process.exit(1); }
  // pptxgenjs lists a slide master per slide in [Content_Types].xml but writes only one: drop the entries without a part
  const types = await zip.file("[Content_Types].xml").async("string");
  zip.file("[Content_Types].xml", types.replace(/<Override PartName="\/([^"]+)"[^>]*\/>/g, (m, part) => (zip.file(part) ? m : "")));
  const buf = await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" });
  fs.writeFileSync(path.resolve(__dirname, out), buf);
  const parts = ["consensus", "cross-dataset", "intensity", "venues"].map((n, k) => `${n} ${R[k].first}–${R[k].last} + ${range(R[k])}`).join("; ");
  const hidden = R.reduce((a, r) => a + r.hidden, 0);
  console.log("wrote", out, `(${at.slideNo + hidden} slides: ${at.slideNo} shown + ${hidden} hidden; ${parts}; removed ${total} repeated paragraph properties)`);
})();
