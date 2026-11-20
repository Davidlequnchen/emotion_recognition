// Shared style and helpers for the turn-segmentation deck (pptxgenjs, 16:9 = 13.333 x 7.5 in).
// Adapted from emotion_recognition_benchmark/presentation/lib.js (same colours, fonts, title layout and notes), with the
// readability rules of this deck: body text >= 20 pt, tables and diagram labels >= 16 pt, no grey text
// (all text is dark navy INK, dark blue for links, or white on dark fills), no auto-shrinking of text.
const fs = require("fs");
const path = require("path");

const DIR = __dirname;
const W = 13.333, H = 7.5, MX = 0.6;            // slide size and side margin (inches)
const CW = W - 2 * MX;                           // content width
const TOP = 1.62, BOTTOM = 6.85;                 // content area below the title, above the citation line
const FONT = "Calibri", MONO = "Consolas";
const C = {
  ink: "1B2A41",        // all text
  link: "00508C",       // dark blue: links, small blue labels
  darkOrange: "9C3D00", darkGreen: "1D6B3F",   // dark text variants of accent2 and good
  accent: "0072B2", accent2: "D55E00", good: "2E8B57", bad: "C0392B", warn: "B7791F", white: "FFFFFF",
  line: "C5CED9",       // lines and borders only, never text
  panel: "F2F5F9",      // light fills behind dark text
  blueFill: "E3EEF8", orangeFill: "FBE6D8", greenFill: "E1F1E7", redFill: "F8E0DE", amberFill: "FFF1D6",
};
const MIN_PT = { body: 20, table: 16, diagram: 16, cite: 14 };   // checked by checkSizes() at build time

let pptx, slideNo = 0, appendixNo = 0, total = 0;
const used = [];                                 // every font size used, for the build-time check
function init(p) { pptx = p; }
function track(size, kind, where) { used.push({ size, kind, where: `slide ${slideNo}: ${where}` }); }

// Pipeline stages for the "you are here" strip on the step slides
const STAGES = ["Prepare", "Language", "Align", "Check", "VAD", "Score", "DP", "Route", "Export"];

function strip(s, active) {
  const n = STAGES.length, gap = 0.08, w = (CW - gap * (n - 1)) / n, y = 0.24, h = 0.4;
  STAGES.forEach((name, i) => {
    const on = active.includes(name);
    s.addText(name, {
      x: MX + i * (w + gap), y, w, h, shape: pptx.ShapeType.rect,
      fill: { color: on ? C.accent : C.panel }, line: { color: on ? C.accent : C.line, width: 0.75 },
      fontFace: FONT, fontSize: 16, bold: on, color: on ? C.white : C.ink, align: "center", valign: "middle", margin: 0,
    });
  });
  track(16, "diagram", "stage strip");
}

// A content slide: section kicker (or the stage strip), the action title, page number
function slide(section, title, opt = {}) {
  const s = pptx.addSlide();
  total += 1;
  let label;
  if (opt.appendix) {                            // appendix: hidden in the slide show, numbered A1, A2, ...
    appendixNo += 1;
    s.hidden = true;
    label = `A${appendixNo}`;
  } else {
    slideNo += 1;
    label = String(slideNo);
  }
  s.background = { color: C.white };
  if (opt.stage) strip(s, [].concat(opt.stage));
  else if (section) s.addText(section.toUpperCase(), { x: MX, y: 0.3, w: 9, h: 0.32, fontFace: FONT, fontSize: 16, bold: true, color: C.link, charSpacing: 1.5, margin: 0 });
  if (title) {
    s.addText(title, { x: MX, y: opt.stage ? 0.78 : 0.68, w: CW, h: 0.72, fontFace: FONT, fontSize: opt.titleSize || 30, bold: true, color: C.ink, valign: "middle", margin: 0 });
    track(opt.titleSize || 30, "body", "title");
  }
  s.addText(label, { x: W - MX - 0.6, y: H - 0.45, w: 0.6, h: 0.32, fontFace: FONT, fontSize: 14, color: C.ink, align: "right", margin: 0 });
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
      bullet: o.noBullet ? false : (lvl ? { code: "2013", indent: 22 } : { code: "25AA", indent: 22 }),
      indentLevel: lvl, paraSpaceAfter: lvl ? 3 : (o.gap ?? 10) };
    if (Array.isArray(it.runs)) {
      it.runs.forEach((r, i) => { const x = { ...base, ...r.options, breakLine: i === it.runs.length - 1 }; if (i > 0) delete x.bullet; runs.push({ text: r.text, options: x }); });
    } else runs.push({ text: it.t, options: { ...base, bold: !!it.b } });
    (it.sub || []).forEach((x) => push(x, lvl + 1));
  };
  items.forEach((it) => push(it, 0));
  s.addText(runs, { x: o.x ?? MX, y: o.y ?? TOP, w: o.w ?? CW, h: o.h ?? BOTTOM - TOP, valign: o.valign || "top", margin: [2, 4, 2, 0], lineSpacingMultiple: o.ls || 1.05 });
  track(fs_, "body", "bullets");
  if (items.some((it) => it.sub)) track(sub, "body", "sub-bullets");
}

// Diagram block with centred text
function box(s, x, y, w, h, t, o = {}) {
  const size = o.fontSize || 16;
  s.addText(t, {
    x, y, w, h, shape: o.shape || pptx.ShapeType.roundRect, rectRadius: o.r ?? 0.08,
    fill: { color: o.fill || C.panel }, line: { color: o.line || C.line, width: o.lw ?? 1.25, dashType: o.dash || "solid" },
    fontFace: o.mono ? MONO : FONT, fontSize: size, color: o.color || C.ink, bold: !!o.bold, align: o.align || "center", valign: o.valign || "middle",
    margin: o.margin ?? [3, 6, 3, 6], lineSpacingMultiple: o.ls || 0.95,
  });
  track(size, "diagram", `box "${String(Array.isArray(t) ? t.map((r) => r.text).join("") : t).slice(0, 24)}"`);
}

// Straight arrow from (x1,y1) to (x2,y2); horizontal and vertical arrows get an exact zero height or width
function arrow(s, x1, y1, x2, y2, o = {}) {
  const horiz = Math.abs(y2 - y1) < 0.005, vert = Math.abs(x2 - x1) < 0.005;
  if (horiz) y2 = y1;
  if (vert) x2 = x1;
  s.addShape(pptx.ShapeType.line, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: vert ? 0 : Math.abs(x2 - x1), h: horiz ? 0 : Math.abs(y2 - y1),
    flipH: !vert && x2 < x1, flipV: !horiz && y2 < y1,
    line: { color: o.color || C.ink, width: o.width || 1.75, endArrowType: o.noHead ? undefined : "triangle", dashType: o.dash || "solid", beginArrowType: o.both ? "triangle" : undefined },
  });
}

// Polyline through points [[x,y], ...]; arrow head on the last segment
function path_(s, pts, o = {}) {
  for (let i = 0; i < pts.length - 1; i++) arrow(s, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], { ...o, noHead: o.noHead || i < pts.length - 2 });
}

function text(s, t, x, y, w, h, o = {}) {
  const size = o.fontSize || 20;
  s.addText(t, { x, y, w, h, fontFace: o.mono ? MONO : FONT, fontSize: size, color: o.color || C.ink, bold: !!o.bold, italic: !!o.italic,
    align: o.align || "left", valign: o.valign || "top", margin: o.margin ?? 0, lineSpacingMultiple: o.ls || 1.0, ...o.extra });
  track(size, o.kind || (size < MIN_PT.body ? "diagram" : "body"), `text "${String(Array.isArray(t) ? t.map((r) => r.text).join("") : t).slice(0, 24)}"`);
}

// Citation line at the bottom: dark text, dark-blue underlined links. links: [[label, url], ...]
function cite(s, t, links = []) {
  const runs = [];
  if (t) runs.push({ text: t, options: { color: C.ink } });
  links.forEach(([label, url], i) => {
    if (i || t) runs.push({ text: i ? "   ·   " : "   ", options: { color: C.ink } });
    if (url) runs.push({ text: label, options: { color: C.link, underline: { style: "sng", color: C.link }, hyperlink: { url } } });
    else runs.push({ text: label, options: { color: C.ink } });     // a book without a stable link
  });
  s.addText(runs, { x: MX, y: H - 0.56, w: CW - 0.8, h: 0.42, fontFace: FONT, fontSize: MIN_PT.cite, margin: 0, valign: "middle" });
  track(MIN_PT.cite, "cite", "citation");
}

function notes(s, t) { s.addNotes(t.replace(/^\s+/gm, "").trim()); }

// Image at its native size (figures are drawn at slide size, so a font size in the figure equals the slide size);
// scaled down only if it does not fit the box, and the scale is reported so the build can check effective sizes.
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
const pending = [];
function image(s, file, x, y, w, h, o = {}) {
  const f = path.isAbsolute(file) ? file : path.join(DIR, file);
  if (!fs.existsSync(f)) {                       // figure not generated yet: a visible placeholder, reported at the end
    pending.push(`slide ${slideNo}: ${path.basename(f)}`);
    box(s, x, y, w, h, `figure pending: ${path.basename(f)}`, { fill: C.white, line: C.bad, dash: "dash", fontSize: 20 });
    return { x, y, w, h, scale: 1 };
  }
  const { w: pw, h: ph } = pngSize(f);
  const dpi = o.dpi || 300, nw = pw / dpi, nh = ph / dpi;
  const r = Math.min(1, w / nw, h / nh), iw = nw * r, ih = nh * r;
  const ix = o.align === "left" ? x : x + (w - iw) / 2, iy = o.valign === "top" ? y : y + (h - ih) / 2;
  s.addImage({ path: f, x: ix, y: iy, w: iw, h: ih, altText: o.alt || path.basename(f, ".png").replace(/_/g, " ") });
  const smallest = (o.minFigPt || 16) * r;     // smallest font in our figures is 16 pt
  track(Math.round(smallest * 10) / 10, "diagram", `figure ${path.basename(f)} (scale ${r.toFixed(3)})`);
  return { x: ix, y: iy, w: iw, h: ih, scale: r };
}

// Embedded audio with a play-button cover (plays on click in the slide show after fix_audio.py)
const COVER = fs.existsSync(path.join(DIR, "img/play.png")) ? "data:image/png;base64," + fs.readFileSync(path.join(DIR, "img/play.png")).toString("base64") : null;
function audio(s, file, x, y, d = 0.5) {
  const f = path.isAbsolute(file) ? file : path.join(DIR, file);
  if (!fs.existsSync(f)) { pending.push(`slide ${slideNo}: ${path.basename(f)}`); return; }
  s.addMedia({ type: "audio", path: f, x, y, w: d, h: d, cover: COVER });
}

// Table with a header row; rows are arrays of strings or {text, options}
function table(s, header, rows, o = {}) {
  const fsz = o.fontSize || 18;
  const cell = (c, hdr, i, j) => {
    const t = typeof c === "object" && c !== null && !Array.isArray(c) ? c : { text: c };
    return { text: t.text, options: { fontFace: t.mono ? MONO : FONT, fontSize: fsz, color: hdr ? C.white : C.ink, bold: hdr || !!t.bold,
      fill: { color: hdr ? (o.headFill || C.ink) : (t.fill || (o.zebra !== false && i % 2 ? C.panel : C.white)) },
      align: t.align || (o.align && o.align[j] ? o.align[j] : "left"), valign: "middle", margin: o.margin || [4, 7, 4, 7], ...t.options } };
  };
  const data = [header.map((h, j) => cell(h, true, 0, j)), ...rows.map((r, i) => r.map((c, j) => cell(c, false, i, j)))];
  s.addTable(data, { x: o.x ?? MX, y: o.y ?? TOP, w: o.w ?? CW, colW: o.colW, rowH: o.rowH, border: { type: "solid", color: C.line, pt: 0.75 }, autoPage: false });
  track(fsz, "table", "table");
}

// Big number with a caption
function stat(s, num, caption, x, y, w, o = {}) {
  text(s, num, x, y, w, 0.8, { fontSize: o.size || 40, bold: true, color: o.color || C.link, align: o.align || "left", valign: "bottom" });
  text(s, caption, x, y + 0.85, w, o.capH || 0.8, { fontSize: o.capSize || 20, color: C.ink, align: o.align || "left" });
}

// Small key (legend) entry: coloured swatch + dark label
function key(s, x, y, color, label, o = {}) {
  const sw = o.sw || 0.36;
  if (o.line) s.addShape(pptx.ShapeType.line, { x, y: y + 0.16, w: sw, h: 0, line: { color, width: 3, dashType: o.dash || "solid" } });
  else s.addShape(pptx.ShapeType.rect, { x, y: y + 0.04, w: sw, h: 0.26, fill: { color }, line: { color: o.edge || color, width: 1 } });
  text(s, label, x + sw + 0.1, y, o.w || 3.5, 0.34, { fontSize: o.fontSize || 16, valign: "middle", kind: "diagram" });
}

// Build-time check: no font below the minimum of its kind
function checkSizes() {
  const bad = used.filter((u) => u.size < MIN_PT[u.kind]);
  const min = (k) => Math.min(...used.filter((u) => u.kind === k).map((u) => u.size));
  console.log(`font sizes: body >= ${min("body")} pt, table >= ${min("table")} pt, diagram >= ${min("diagram")} pt, citations ${min("cite")} pt`);
  bad.forEach((u) => console.log(`  TOO SMALL: ${u.size} pt (${u.kind}) at ${u.where}`));
  pending.forEach((p_) => console.log(`  FIGURE PENDING: ${p_}`));
  return bad.length === 0;
}

module.exports = { init, slide, blank, strip, bullets, box, arrow, path: path_, text, cite, notes, image, audio, table, stat, key, checkSizes,
  C, W, H, MX, CW, TOP, BOTTOM, FONT, MONO, STAGES, DIR, get pptx() { return pptx; }, get slideNo() { return slideNo; },
  get total() { return total; }, get appendixNo() { return appendixNo; } };
