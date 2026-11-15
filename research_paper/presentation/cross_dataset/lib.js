// Shared style and helpers for the cross-dataset deck (pptxgenjs, 16:9 = 13.333 x 7.5 in).
// Adapted from preparation_tools/presentation/lib.js (same colours, fonts, title layout and notes) with the
// readability rules of the research_paper decks: body text >= 20 pt, tables >= 18 pt, figure and diagram labels
// >= 16 pt, page numbers and source lines 16 pt, no grey text (all text is dark navy INK, dark blue for links,
// dark orange for the "preliminary" tag, or white on dark fills), no auto-shrinking of text.
// checkSizes() stops the build if a font is smaller.
const fs = require("fs");
const path = require("path");

const DIR = __dirname;
const FIG = path.join(DIR, "..", "..", "analysis_report", "figures");   // single source: cross_dataset_plot.py
const W = 13.333, H = 7.5, MX = 0.6;            // slide size and side margin (inches)
const CW = W - 2 * MX;                           // content width
const TOP = 1.62, BOTTOM = 6.9;                  // content area below the title, above the page number
const FONT = "Calibri";
const C = {
  ink: "1B2A41",        // all text
  link: "00508C",       // dark blue: section kicker, links
  darkOrange: "9C3D00", // the "preliminary" tag
  accent: "0072B2", accent2: "D55E00", good: "2E8B57", bad: "C0392B", warn: "B7791F", white: "FFFFFF",
  line: "C5CED9",       // lines and borders only, never text
  panel: "F2F5F9",      // light fills behind dark text
  blueFill: "E3EEF8", orangeFill: "FBE6D8", greenFill: "E1F1E7", redFill: "F8E0DE", amberFill: "FFF1D6",
};
const MIN_PT = { body: 20, table: 18, diagram: 16, page: 16 };   // checked by checkSizes() at build time

let pptx, slideNo = 0, appendixNo = 0, total = 0;
const used = [];                                 // every font size used, for the build-time check
// start: page counters already used when these slides are part of a larger deck (../combined/build.js passes
// { slideNo, appendixNo } so page numbers run on); this deck alone calls init(pptx) and starts at 1 and A1.
function init(p, start = {}) { pptx = p; slideNo = start.slideNo || 0; appendixNo = start.appendixNo || 0; }
function where() { return appendixNo ? `A${appendixNo}` : String(slideNo); }
function track(size, kind, what) { used.push({ size, kind, where: `slide ${where()}: ${what}` }); }

// A content slide: section kicker, the action title, page number. opt.appendix: hidden, numbered A1, A2, ...
function slide(section, title, opt = {}) {
  const s = pptx.addSlide();
  total += 1;
  let label;
  if (opt.appendix) {
    appendixNo += 1;
    s.hidden = true;
    label = `A${appendixNo}`;
  } else {
    slideNo += 1;
    label = String(slideNo);
  }
  s.background = { color: C.white };
  if (section) {
    s.addText(section.toUpperCase(), { x: MX, y: 0.3, w: 9, h: 0.34, fontFace: FONT, fontSize: 16, bold: true, color: C.link, charSpacing: 1.5, margin: 0 });
    track(16, "diagram", "section kicker");
  }
  if (title) {
    const size = opt.titleSize || (title.length > 118 ? 26 : 28);   // long conclusion titles stay on two lines
    s.addText(title, { x: MX, y: 0.66, w: CW, h: 0.82, fontFace: FONT, fontSize: size, bold: true, color: C.ink, valign: "middle", margin: 0, lineSpacingMultiple: 0.92 });
    track(size, "body", "title");
  }
  s.addText(label, { x: W - MX - 0.8, y: H - 0.46, w: 0.8, h: 0.34, fontFace: FONT, fontSize: MIN_PT.page, color: C.ink, align: "right", margin: 0 });
  track(MIN_PT.page, "page", "page number");
  return s;
}

// A slide without title layout or page number (title slide)
function blank() {
  const s = pptx.addSlide();
  slideNo += 1;
  total += 1;
  s.background = { color: C.white };
  return s;
}

// Bullets: items are strings, {t, sub: [..]}, {t, b: true}, or {runs: [{text, options}]}
function bullets(s, items, o = {}) {
  const fs_ = o.fontSize || 22, sub = o.subSize || Math.max(MIN_PT.body, fs_ - 2), runs = [];
  const push = (it, lvl) => {
    if (typeof it === "string") it = { t: it };
    const base = { fontFace: FONT, fontSize: lvl ? sub : fs_, color: C.ink, breakLine: true,
      bullet: o.noBullet ? false : (lvl ? { code: "2013", indent: 20 } : { code: "25AA", indent: 20 }),
      indentLevel: lvl, paraSpaceAfter: lvl ? 3 : (o.gap ?? 10) };
    if (Array.isArray(it.runs)) {
      it.runs.forEach((r, i) => { const x = { ...base, ...r.options, breakLine: i === it.runs.length - 1 }; if (i > 0) delete x.bullet; runs.push({ text: r.text, options: x }); });
    } else runs.push({ text: it.t, options: { ...base, bold: !!it.b } });
    (it.sub || []).forEach((x) => push(x, lvl + 1));
  };
  items.forEach((it) => push(it, 0));
  s.addText(runs, { x: o.x ?? MX, y: o.y ?? TOP, w: o.w ?? CW, h: o.h ?? BOTTOM - TOP, valign: o.valign || "top", margin: [2, 4, 2, 0], lineSpacingMultiple: o.ls || 1.0 });
  track(fs_, "body", "bullets");
  if (items.some((it) => it.sub)) track(sub, "body", "sub-bullets");
}

// Bold lead-in followed by normal text, as one bullet item
function lead(b, t) { return { runs: [{ text: b, options: { bold: true } }, { text: t }] }; }

// Diagram block with centred text
function box(s, x, y, w, h, t, o = {}) {
  const size = o.fontSize || 18;
  s.addText(t, {
    x, y, w, h, shape: o.shape || pptx.ShapeType.roundRect, rectRadius: o.r ?? 0.08,
    fill: { color: o.fill || C.panel }, line: { color: o.line || C.line, width: o.lw ?? 1.25, dashType: o.dash || "solid" },
    fontFace: FONT, fontSize: size, color: o.color || C.ink, bold: !!o.bold, align: o.align || "center", valign: o.valign || "middle",
    margin: o.margin ?? [3, 6, 3, 6], lineSpacingMultiple: o.ls || 0.95,
  });
  track(size, size < MIN_PT.body ? "diagram" : "body", `box "${String(Array.isArray(t) ? t.map((r) => r.text).join("") : t).slice(0, 24)}"`);
}

function text(s, t, x, y, w, h, o = {}) {
  const size = o.fontSize || 20;
  s.addText(t, { x, y, w, h, fontFace: FONT, fontSize: size, color: o.color || C.ink, bold: !!o.bold, italic: !!o.italic,
    align: o.align || "left", valign: o.valign || "top", margin: o.margin ?? 0, lineSpacingMultiple: o.ls || 1.0, ...o.extra });
  track(size, o.kind || (size < MIN_PT.body ? "diagram" : "body"), `text "${String(Array.isArray(t) ? t.map((r) => r.text).join("") : t).slice(0, 24)}"`);
}

// "Preliminary" tag (dark orange on a light orange fill) for results that still wait for Gemma 4 E4B
function tag(s, t, x, y, w, o = {}) {
  box(s, x, y, w, o.h || 0.4, t, { fill: C.orangeFill, line: C.accent2, color: C.darkOrange, bold: true, fontSize: o.fontSize || 16, r: 0.05 });
}

// Source line at the bottom: dark text, 16 pt (no grey captions)
function source(s, t) {
  s.addText(t, { x: MX, y: H - 0.5, w: CW - 1.0, h: 0.38, fontFace: FONT, fontSize: 16, color: C.ink, margin: 0, valign: "middle" });
  track(16, "diagram", "source line");
}

function notes(s, t) { s.addNotes(t.replace(/^\s+/gm, "").trim()); }

// Figure at its native size (figures are drawn at slide size, so a font size in the figure equals the slide size);
// scaled down only if it does not fit the box, and the scale is reported so the build can check effective sizes.
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
const pending = [];
function figure(s, name, x, y, w, h, o = {}) {
  const f = path.join(FIG, name);
  if (!fs.existsSync(f)) {                       // figure not generated yet: a visible placeholder, reported at the end
    pending.push(`slide ${where()}: ${name}`);
    box(s, x, y, w, h, `figure pending: ${name}`, { fill: C.white, line: C.bad, dash: "dash", fontSize: 20 });
    return { x, y, w, h, scale: 1 };
  }
  const { w: pw, h: ph } = pngSize(f);
  const nw = pw / 300, nh = ph / 300;
  const r = Math.min(1, w / nw, h / nh), iw = nw * r, ih = nh * r;
  const ix = o.align === "left" ? x : o.align === "right" ? x + w - iw : x + (w - iw) / 2;
  const iy = o.valign === "top" ? y : y + (h - ih) / 2;
  s.addImage({ path: f, x: ix, y: iy, w: iw, h: ih, altText: o.alt || name.replace(/_/g, " ").replace(".png", "") });
  track(Math.round(16 * r * 10) / 10, "diagram", `figure ${name} (scale ${r.toFixed(3)})`);   // smallest figure font: 16 pt
  return { x: ix, y: iy, w: iw, h: ih, scale: r };
}

// Table with a header row; rows are arrays of strings or {text, bold, fill, color, align}
function table(s, header, rows, o = {}) {
  const fsz = o.fontSize || 18;
  const cell = (c, hdr, i, j) => {
    const t = typeof c === "object" && c !== null && !Array.isArray(c) ? c : { text: c };
    return { text: t.text, options: { fontFace: FONT, fontSize: fsz, color: hdr ? C.white : (t.color || C.ink), bold: hdr || !!t.bold,
      fill: { color: hdr ? (o.headFill || C.ink) : (t.fill || (o.zebra !== false && i % 2 ? C.panel : C.white)) },
      align: t.align || (o.align && o.align[j] ? o.align[j] : "left"), valign: "middle", margin: o.margin || [3, 6, 3, 6] } };
  };
  const data = [header.map((h, j) => cell(h, true, 0, j)), ...rows.map((r, i) => r.map((c, j) => cell(c, false, i, j)))];
  s.addTable(data, { x: o.x ?? MX, y: o.y ?? TOP, w: o.w ?? CW, colW: o.colW, rowH: o.rowH, border: { type: "solid", color: C.line, pt: 0.75 }, autoPage: false });
  track(fsz, "table", "table");
}

// Build-time check: no font below the minimum of its kind
function checkSizes() {
  const bad = used.filter((u) => u.size < MIN_PT[u.kind]);
  const min = (k) => Math.min(...used.filter((u) => u.kind === k).map((u) => u.size));
  console.log(`font sizes: body >= ${min("body")} pt, table >= ${min("table")} pt, figures/diagrams >= ${min("diagram")} pt, page ${min("page")} pt`);
  bad.forEach((u) => console.log(`  TOO SMALL: ${u.size} pt (${u.kind}) at ${u.where}`));
  pending.forEach((p_) => console.log(`  FIGURE PENDING: ${p_}`));
  return bad.length === 0;
}

module.exports = { init, slide, blank, bullets, lead, box, text, tag, source, notes, figure, table, checkSizes,
  C, W, H, MX, CW, TOP, BOTTOM, FONT, DIR, FIG, get pptx() { return pptx; }, get slideNo() { return slideNo; },
  get total() { return total; }, get appendixNo() { return appendixNo; } };
