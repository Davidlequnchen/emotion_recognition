// Shared style and helpers for the consensus deck (pptxgenjs, 16:9 = 13.333 x 7.5 in).
// Adapted from preparation_tools/presentation/lib.js (same colours, fonts, title layout and notes) with the
// readability rules of these decks: body text >= 20 pt, tables >= 18 pt, figure and diagram labels >= 16 pt,
// page numbers 16 pt, no grey text (all text is dark navy INK, dark blue for links, or white on dark fills),
// no auto-shrinking of text. checkSizes() stops the build if a font is smaller.
const fs = require("fs");
const path = require("path");

const DIR = __dirname;
const W = 13.333, H = 7.5, MX = 0.6;            // slide size and side margin (inches)
const CW = W - 2 * MX;                           // content width
const TOP = 1.62, BOTTOM = 6.9;                  // content area below the title, above the page number
const FONT = "Calibri", MONO = "Consolas";
const C = {
  ink: "1B2A41",        // all text
  link: "00508C",       // dark blue: section kicker, links
  accent: "0072B2", good: "2E8B57", bad: "C0392B", warn: "B7791F", white: "FFFFFF",
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
function track(size, kind, where) { used.push({ size, kind, where: `slide ${appendixNo ? "A" + appendixNo : slideNo}: ${where}` }); }

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
    s.addText(title, { x: MX, y: 0.66, w: CW, h: 0.76, fontFace: FONT, fontSize: opt.titleSize || 30, bold: true, color: C.ink, valign: "middle", margin: 0 });
    track(opt.titleSize || 30, "body", "title");
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
  const size = o.fontSize || 18;
  s.addText(t, {
    x, y, w, h, shape: o.shape || pptx.ShapeType.roundRect, rectRadius: o.r ?? 0.08,
    fill: { color: o.fill || C.panel }, line: { color: o.line || C.line, width: o.lw ?? 1.25, dashType: o.dash || "solid" },
    fontFace: o.mono ? MONO : FONT, fontSize: size, color: o.color || C.ink, bold: !!o.bold, align: o.align || "center", valign: o.valign || "middle",
    margin: o.margin ?? [3, 6, 3, 6], lineSpacingMultiple: o.ls || 0.95,
  });
  track(size, size < MIN_PT.body ? "diagram" : "body", `box "${String(Array.isArray(t) ? t.map((r) => r.text).join("") : t).slice(0, 24)}"`);
}

// Straight arrow from (x1,y1) to (x2,y2); horizontal and vertical arrows get an exact zero height or width
function arrow(s, x1, y1, x2, y2, o = {}) {
  const horiz = Math.abs(y2 - y1) < 0.005, vert = Math.abs(x2 - x1) < 0.005;
  if (horiz) y2 = y1;
  if (vert) x2 = x1;
  s.addShape(pptx.ShapeType.line, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: vert ? 0 : Math.abs(x2 - x1), h: horiz ? 0 : Math.abs(y2 - y1),
    flipH: !vert && x2 < x1, flipV: !horiz && y2 < y1,
    line: { color: o.color || C.ink, width: o.width || 1.75, endArrowType: o.noHead ? undefined : "triangle", dashType: o.dash || "solid" },
  });
}

function text(s, t, x, y, w, h, o = {}) {
  const size = o.fontSize || 20;
  s.addText(t, { x, y, w, h, fontFace: o.mono ? MONO : FONT, fontSize: size, color: o.color || C.ink, bold: !!o.bold, italic: !!o.italic,
    align: o.align || "left", valign: o.valign || "top", margin: o.margin ?? 0, lineSpacingMultiple: o.ls || 1.0, ...o.extra });
  track(size, o.kind || (size < MIN_PT.body ? "diagram" : "body"), `text "${String(Array.isArray(t) ? t.map((r) => r.text).join("") : t).slice(0, 24)}"`);
}

function notes(s, t) { s.addNotes(t.replace(/^\s+/gm, "").trim()); }

// Image at its native size (figures are drawn at slide size by consensus_plot.plot_slides, so a font size in the
// figure equals the size on the slide); scaled down only if it does not fit, and the effective size is checked.
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
  track(Math.round((o.minFigPt || 16) * r * 10) / 10, "diagram", `figure ${path.basename(f)} (scale ${r.toFixed(3)})`);
  return { x: ix, y: iy, w: iw, h: ih, scale: r };
}

// Table with a header row; rows are arrays of strings or {text, options}
function table(s, header, rows, o = {}) {
  const fsz = o.fontSize || 20;
  const cell = (c, hdr, i, j) => {
    const t = typeof c === "object" && c !== null && !Array.isArray(c) ? c : { text: c };
    return { text: t.text, options: { fontFace: t.mono ? MONO : FONT, fontSize: fsz, color: hdr ? C.white : C.ink, bold: hdr || !!t.bold,
      fill: { color: hdr ? (o.headFill || C.ink) : (t.fill || (o.zebra !== false && i % 2 ? C.panel : C.white)) },
      align: t.align || (o.align && o.align[j] ? o.align[j] : "left"), valign: "middle", margin: o.margin || [3, 7, 3, 7], ...t.options } };
  };
  const data = [header.map((h, j) => cell(h, true, 0, j)), ...rows.map((r, i) => r.map((c, j) => cell(c, false, i, j)))];
  s.addTable(data, { x: o.x ?? MX, y: o.y ?? TOP, w: o.w ?? CW, colW: o.colW, rowH: o.rowH, border: { type: "solid", color: C.line, pt: 0.75 }, autoPage: false });
  track(fsz, "table", "table");
}

// Big number with a caption
function stat(s, num, caption, x, y, w, o = {}) {
  text(s, num, x, y, w, 0.8, { fontSize: o.size || 44, bold: true, color: o.color || C.link, align: o.align || "left", valign: "bottom" });
  text(s, caption, x, y + 0.85, w, o.capH || 0.9, { fontSize: o.capSize || 20, color: C.ink, align: o.align || "left" });
}

// Group marker: the figure's marker shape in the group's colour (circle, triangle, diamond, square, ...)
function marker(s, g, x, y, d = 0.26) {
  const shapes = { o: "ellipse", "^": "triangle", D: "diamond", s: "rect", v: "flowChartMerge", p: "pentagon", h: "hexagon" };
  s.addShape(pptx.ShapeType[shapes[g.marker]] || pptx.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: g.colour }, line: { color: C.ink, width: 0.75 } });
}

// Build-time check: no font below the minimum of its kind
function checkSizes() {
  const bad = used.filter((u) => u.size < MIN_PT[u.kind]);
  const min = (k) => Math.min(...used.filter((u) => u.kind === k).map((u) => u.size));
  console.log(`font sizes: body >= ${min("body")} pt, table >= ${min("table")} pt, diagram/figure >= ${min("diagram")} pt, page numbers ${min("page")} pt`);
  bad.forEach((u) => console.log(`  TOO SMALL: ${u.size} pt (${u.kind}) at ${u.where}`));
  pending.forEach((p_) => console.log(`  FIGURE PENDING: ${p_}`));
  return bad.length === 0;
}

module.exports = { init, slide, blank, bullets, box, arrow, text, notes, image, table, stat, marker, checkSizes,
  C, W, H, MX, CW, TOP, BOTTOM, FONT, MONO, DIR, get pptx() { return pptx; }, get slideNo() { return slideNo; },
  get total() { return total; }, get appendixNo() { return appendixNo; } };
