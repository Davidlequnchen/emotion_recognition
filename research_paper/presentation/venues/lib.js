// Shared style and helpers for the venue deck (pptxgenjs, 16:9 = 13.333 x 7.5 in); also used by ../combined/build.js
// for the cover and the section dividers. Adapted from ../consensus/lib.js (same colours, fonts, title layout, page
// numbers and notes) with the readability rules of the research_paper decks: body text >= 20 pt, tables >= 18 pt,
// source lines and page numbers 16 pt, no grey text (all text is dark navy INK, dark blue for links, dark green,
// dark orange or dark red for role tags, or white on dark fills), no auto-shrinking of text. checkSizes() stops the
// build if a font is smaller.
const DIR = __dirname;
const W = 13.333, H = 7.5, MX = 0.6;            // slide size and side margin (inches)
const CW = W - 2 * MX;                           // content width
const TOP = 1.62, BOTTOM = 6.9;                  // content area below the title, above the page number
const FONT = "Calibri";
const C = {
  ink: "1B2A41",        // all text
  link: "00508C",       // dark blue: section kicker, links, "main target"
  darkGreen: "1D6B3F",  // "focus candidate"
  darkOrange: "9C3D00", // "fallback"
  darkRed: "8E1B1B",    // "not recommended", "no window" (additional candidates)
  accent: "0072B2", white: "FFFFFF",
  line: "C5CED9",       // lines and borders only, never text
  panel: "F2F5F9",      // light fills behind dark text
  blueFill: "E3EEF8",
};
const MIN_PT = { body: 20, table: 18, small: 16, page: 16 };   // checked by checkSizes() at build time

let pptx, slideNo = 0, appendixNo = 0, total = 0;
const used = [];                                 // every font size used, for the build-time check
// start: page counters already used when these slides are part of a larger deck (../combined/build.js passes
// { slideNo, appendixNo } so page numbers run on); this deck alone calls init(pptx) and starts at 1.
function init(p, start = {}) { pptx = p; slideNo = start.slideNo || 0; appendixNo = start.appendixNo || 0; }
function track(size, kind, where) { used.push({ size, kind, where: `slide ${slideNo}: ${where}` }); }

// A content slide: section kicker, the action title, page number
function slide(section, title, opt = {}) {
  const s = pptx.addSlide();
  total += 1;
  slideNo += 1;
  s.background = { color: C.white };
  if (section) {
    s.addText(section.toUpperCase(), { x: MX, y: 0.3, w: 9, h: 0.34, fontFace: FONT, fontSize: 16, bold: true, color: C.link, charSpacing: 1.5, margin: 0 });
    track(16, "small", "section kicker");
  }
  if (title) {
    s.addText(title, { x: MX, y: 0.66, w: CW, h: 0.76, fontFace: FONT, fontSize: opt.titleSize || 30, bold: true, color: C.ink, valign: "middle", margin: 0 });
    track(opt.titleSize || 30, "body", "title");
  }
  s.addText(String(slideNo), { x: W - MX - 0.8, y: H - 0.46, w: 0.8, h: 0.34, fontFace: FONT, fontSize: MIN_PT.page, color: C.ink, align: "right", margin: 0 });
  track(MIN_PT.page, "page", "page number");
  return s;
}

// A slide without title layout or page number (title slide, cover, section divider)
function blank(o = {}) {
  const s = pptx.addSlide();
  slideNo += 1;
  total += 1;
  s.background = { color: o.background || C.white };
  return s;
}

function text(s, t, x, y, w, h, o = {}) {
  const size = o.fontSize || 20;
  s.addText(t, { x, y, w, h, fontFace: FONT, fontSize: size, color: o.color || C.ink, bold: !!o.bold, italic: !!o.italic,
    align: o.align || "left", valign: o.valign || "top", margin: o.margin ?? 0, lineSpacingMultiple: o.ls || 1.0, ...o.extra });
  track(size, o.kind || (size < MIN_PT.body ? "small" : "body"), `text "${String(Array.isArray(t) ? t.map((r) => r.text).join("") : t).slice(0, 24)}"`);
}

// Source line at the bottom (one or two lines): dark text, 16 pt, no grey captions
function source(s, t, o = {}) {
  const h = o.h || 0.38;
  s.addText(t, { x: MX, y: H - 0.12 - h, w: CW - 1.0, h, fontFace: FONT, fontSize: 16, color: C.ink, margin: 0, valign: "bottom", lineSpacingMultiple: 0.95 });
  track(16, "small", "source line");
}

function notes(s, t) { s.addNotes(t.replace(/^\s+/gm, "").trim()); }

// Table with a header row. A cell is a string ("\n" breaks lines), {text | lines, bold, fill, align, colspan}, or an
// array of lines; a line is a string, a segment {t, b, color, url} (url: a dark-blue underlined hyperlink), or an
// array of segments printed on the same line (e.g. a bold lead-in and normal text).
function table(s, header, rows, o = {}) {
  const fsz = o.fontSize || 18;
  // keep a number with its sign or unit on one line: "≤ 2", "~40%", "8 p", "1 week"
  const nb = (t) => t.replace(/([≤≥<~]) (?=\d)/g, "$1\u00A0").replace(/(\d) (p|week)\b/g, "$1\u00A0$2");
  const runs = (lines, hdr) => lines.flatMap((ln, k) => {
    const segs = (Array.isArray(ln) ? ln : [ln]).map((g) => (typeof g === "string" ? { t: g } : g));
    return segs.map((l, m) => {
      const opt = { fontFace: FONT, fontSize: fsz, color: hdr ? C.white : (l.color || C.ink), bold: hdr || !!l.b,
        breakLine: m === segs.length - 1 && k < lines.length - 1 };
      if (l.url) Object.assign(opt, { color: C.link, underline: { style: "sng", color: C.link }, hyperlink: { url: l.url, tooltip: l.url } });
      return { text: nb(l.t), options: opt };
    });
  });
  const cell = (c, hdr, i, j) => {
    const t = Array.isArray(c) ? { lines: c } : typeof c === "object" && c !== null ? c : { text: c };
    const opt = { fontFace: FONT, fontSize: fsz, color: hdr ? C.white : C.ink, bold: hdr || !!t.bold,
      fill: { color: hdr ? (o.headFill || C.ink) : (t.fill || (o.zebra !== false && i % 2 ? C.panel : C.white)) },
      align: t.align || (o.align && o.align[j] ? o.align[j] : "left"), valign: o.valign || "middle", margin: o.margin || [3, 7, 3, 7] };
    if (t.colspan) opt.colspan = t.colspan;
    return { text: t.lines ? runs(t.lines, hdr) : nb(String(t.text)), options: opt };
  };
  const data = [header.map((h, j) => cell(h, true, 0, j)), ...rows.map((r, i) => r.map((c, j) => cell(c, false, i, j)))];
  s.addTable(data, { x: o.x ?? MX, y: o.y ?? TOP, w: o.w ?? CW, colW: o.colW, rowH: o.rowH, border: { type: "solid", color: C.line, pt: 0.75 }, autoPage: false });
  track(fsz, "table", "table");
}

// Build-time check: no font below the minimum of its kind
function checkSizes() {
  const bad = used.filter((u) => u.size < MIN_PT[u.kind]);
  const min = (k) => Math.min(...used.filter((u) => u.kind === k).map((u) => u.size));
  console.log(`font sizes: body >= ${min("body")} pt, table >= ${min("table")} pt, source lines/kickers >= ${min("small")} pt, page numbers ${min("page")} pt`);
  bad.forEach((u) => console.log(`  TOO SMALL: ${u.size} pt (${u.kind}) at ${u.where}`));
  return bad.length === 0;
}

module.exports = { init, slide, blank, text, source, notes, table, checkSizes,
  C, W, H, MX, CW, TOP, BOTTOM, FONT, DIR, get pptx() { return pptx; }, get slideNo() { return slideNo; },
  get total() { return total; }, get appendixNo() { return appendixNo; } };
