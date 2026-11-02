// Shared style and helpers for the SER evaluation deck (pptxgenjs, 16:9 = 13.333 x 7.5 in)
const fs = require("fs");
const path = require("path");

const DIR = __dirname;
const W = 13.333, H = 7.5, MX = 0.6;            // slide size and side margin (inches)
const FONT = "Calibri";
const C = {
  ink: "1B2A41", text: "2F3B4C", muted: "6B7785", faint: "9AA4B1", line: "D5DAE1", panel: "F4F6F9",
  accent: "0072B2", accent2: "D55E00", good: "2E8B57", bad: "C0392B", warn: "B7791F", white: "FFFFFF",
};
// Same colours as the bar charts (Okabe-Ito, scripts/viewer/plot_results.py)
const MODEL = {
  e2v: { name: "emotion2vec+ large", color: "0072B2" },
  meralion: { name: "MERaLiON-SER-v1", color: "E69F00" },
  melati: { name: "Melati", color: "009E73" },
  voxw: { name: "Vox-Profile Whisper", color: "CC79A7" },
  voxl: { name: "Vox-Profile WavLM", color: "56B4E9" },
  gpt: { name: "GPT Realtime", color: "D55E00" },
  gemini: { name: "Gemini 3.5 Flash", color: "F0E442" },
};
const MODEL_KEY = {
  emotion2vec_plus_large: "e2v", meralion_ser_v1: "meralion", melati_emotion: "melati",
  vox_profile_whisper: "voxw", vox_profile_wavlm: "voxl",
};
// Emotion colours (the labeling tool's palette)
const EMO = {
  angry: "E5484D", disgusted: "46A758", fearful: "8E4EC6", happy: "F5A524", neutral: "8B8D98", sad: "3E63DD",
  surprised: "12A594", calm: "8B8D98", excited: "F5A524",
};

let pptx, slideNo = 0;
function init(p) { pptx = p; }

// A content slide: small section kicker, an action title (the slide's message), page number
function slide(section, title, opt = {}) {
  const s = pptx.addSlide();
  slideNo += 1;
  s.background = { color: C.white };
  if (section) s.addText(section.toUpperCase(), { x: MX, y: 0.28, w: 8, h: 0.3, fontFace: FONT, fontSize: 11, bold: true, color: C.accent, charSpacing: 1.5, margin: 0 });
  if (title) s.addText(title, { x: MX, y: 0.55, w: W - 2 * MX, h: opt.titleH || 0.8, fontFace: FONT, fontSize: opt.titleSize || 26, bold: true, color: C.ink, valign: "top", margin: 0, fit: "shrink" });
  s.addText(String(slideNo), { x: W - MX - 0.6, y: H - 0.42, w: 0.6, h: 0.3, fontFace: FONT, fontSize: 10, color: C.faint, align: "right", margin: 0 });
  return s;
}

// Bullets: items are strings, or {t, sub: [..]}, or {t, b: true} for bold, or arrays of runs [{text, options}]
function bullets(s, items, o = {}) {
  const fs_ = o.fontSize || 18, runs = [];
  const push = (it, lvl) => {
    if (typeof it === "string") it = { t: it };
    const base = { fontFace: FONT, fontSize: lvl ? fs_ - 3 : fs_, color: lvl ? C.text : (o.color || C.ink), breakLine: true,
      bullet: o.noBullet ? false : (lvl ? { code: "2013", indent: 18 } : { code: "25AA", indent: 18 }),
      indentLevel: lvl, paraSpaceAfter: lvl ? 2 : (o.gap ?? 8) };
    if (Array.isArray(it.runs)) {
      it.runs.forEach((r, i) => { const o = { ...base, ...r.options, breakLine: i === it.runs.length - 1 }; if (i > 0) delete o.bullet; runs.push({ text: r.text, options: o }); });
    } else runs.push({ text: it.t, options: { ...base, bold: !!it.b, color: it.color || base.color } });
    (it.sub || []).forEach((x) => push(x, lvl + 1));
  };
  items.forEach((it) => push(it, 0));
  s.addText(runs, { x: o.x ?? MX, y: o.y ?? 1.55, w: o.w ?? W - 2 * MX, h: o.h ?? 5.2, valign: "top", margin: [2, 4, 2, 0], fit: "shrink", lineSpacingMultiple: o.ls || 1.05 });
}

// A box with centred text (diagram block)
function box(s, x, y, w, h, text, o = {}) {
  s.addText(text, {
    x, y, w, h, shape: o.shape || pptx.ShapeType.roundRect, rectRadius: o.r ?? 0.08,
    fill: { color: o.fill || C.panel }, line: { color: o.line || C.line, width: o.lw ?? 1, dashType: o.dash || "solid" },
    fontFace: FONT, fontSize: o.fontSize || 13, color: o.color || C.ink, bold: !!o.bold, align: o.align || "center", valign: o.valign || "middle",
    margin: o.margin ?? [3, 5, 3, 5], fit: o.fit || "shrink", lineSpacingMultiple: 0.95,
  });
}

// Straight arrow from (x1,y1) to (x2,y2); horizontal and vertical arrows get an exact zero height or width
function arrow(s, x1, y1, x2, y2, o = {}) {
  const horiz = Math.abs(y2 - y1) < 0.005, vert = Math.abs(x2 - x1) < 0.005;
  if (horiz) y2 = y1;
  if (vert) x2 = x1;
  s.addShape(pptx.ShapeType.line, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: vert ? 0 : Math.abs(x2 - x1), h: horiz ? 0 : Math.abs(y2 - y1),
    flipH: !vert && x2 < x1, flipV: !horiz && y2 < y1,
    line: { color: o.color || C.muted, width: o.width || 1.75, endArrowType: o.noHead ? undefined : "triangle", dashType: o.dash || "solid", beginArrowType: o.both ? "triangle" : undefined },
  });
}

// Right-angled connector: along x first, then y (or y first with o.yFirst); arrow head on the last segment
function elbow(s, x1, y1, x2, y2, o = {}) {
  if (o.yFirst) { arrow(s, x1, y1, x1, y2, { ...o, noHead: true }); arrow(s, x1, y2, x2, y2, o); }
  else { arrow(s, x1, y1, x2, y1, { ...o, noHead: true }); arrow(s, x2, y1, x2, y2, o); }
}

function text(s, t, x, y, w, h, o = {}) {
  s.addText(t, { x, y, w, h, fontFace: FONT, fontSize: o.fontSize || 14, color: o.color || C.text, bold: !!o.bold, italic: !!o.italic,
    align: o.align || "left", valign: o.valign || "top", margin: o.margin ?? 0, fit: o.fit || "shrink", lineSpacingMultiple: o.ls || 1.0, ...o.extra });
}

// Small grey source line at the bottom
function source(s, t) {
  s.addText(t, { x: MX, y: H - 0.45, w: W - 2 * MX - 0.8, h: 0.32, fontFace: FONT, fontSize: 9, color: C.faint, margin: 0, valign: "bottom", fit: "shrink" });
}

function notes(s, t) { s.addNotes(t.replace(/^\s+/gm, "").trim()); }

// Image scaled to fit inside a box (keeps aspect ratio), from the PNG header
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function image(s, file, x, y, w, h, o = {}) {
  const f = path.isAbsolute(file) ? file : path.join(DIR, file);
  const { w: pw, h: ph } = pngSize(f);
  const r = Math.min(w / pw, h / ph), iw = pw * r, ih = ph * r;
  const ix = o.align === "left" ? x : x + (w - iw) / 2, iy = o.valign === "top" ? y : y + (h - ih) / 2;
  s.addImage({ path: f, x: ix, y: iy, w: iw, h: ih, altText: o.alt || path.basename(f, ".png").replace(/_/g, " ") });
  return { x: ix, y: iy, w: iw, h: ih };
}

// Pill-shaped emotion tag
function tag(s, label, x, y, o = {}) {
  const w = o.w || Math.max(0.75, 0.13 * label.length + 0.3), h = o.h || 0.3;
  const col = o.color || EMO[label.split(" ")[0]] || C.muted;
  s.addText(label, { x, y, w, h, shape: pptx.ShapeType.roundRect, rectRadius: h / 2, fill: { color: col }, line: { color: col, width: 0.5 },
    fontFace: FONT, fontSize: o.fontSize || 12, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
  return w;
}

// Embedded audio with a play-button cover; plays on click in slide show
const COVER = "data:image/png;base64," + fs.readFileSync(path.join(DIR, "img/play.png")).toString("base64");
const COVER_W = "data:image/png;base64," + fs.readFileSync(path.join(DIR, "img/play_white.png")).toString("base64");
function audio(s, file, x, y, d = 0.55, white = false) {
  const f = path.isAbsolute(file) ? file : path.join(DIR, file);
  s.addMedia({ type: "audio", path: f, x, y, w: d, h: d, cover: white ? COVER_W : COVER });
}

// Table with a header row; rows are arrays of strings or {text, options}
function table(s, header, rows, o = {}) {
  const fsz = o.fontSize || 12;
  const cell = (c, hdr, i) => {
    const t = typeof c === "object" && c !== null ? c : { text: String(c) };
    return { text: t.text, options: { fontFace: FONT, fontSize: fsz, color: hdr ? C.white : C.ink, bold: hdr || !!t.bold,
      fill: { color: hdr ? (o.headFill || C.ink) : (t.fill || (o.zebra !== false && i % 2 ? C.panel : C.white)) },
      align: t.align || (o.align && o.align[t.col] ? o.align[t.col] : "left"), valign: "middle", margin: [3, 5, 3, 5], ...t.options } };
  };
  const data = [header.map((h) => cell(h, true, 0)), ...rows.map((r, i) => r.map((c) => cell(c, false, i)))];
  s.addTable(data, { x: o.x ?? MX, y: o.y ?? 1.55, w: o.w ?? W - 2 * MX, colW: o.colW, rowH: o.rowH, border: { type: "solid", color: C.line, pt: 0.75 }, autoPage: false });
}

// Big number with a caption
function stat(s, num, caption, x, y, w, o = {}) {
  text(s, num, x, y, w, 0.75, { fontSize: o.size || 40, bold: true, color: o.color || C.accent, align: o.align || "left" });
  text(s, caption, x, y + 0.75, w, o.capH || 0.7, { fontSize: o.capSize || 13, color: C.text, align: o.align || "left" });
}

module.exports = { init, slide, bullets, box, arrow, elbow, text, source, notes, image, tag, audio, table, stat, C, W, H, MX, FONT, MODEL, MODEL_KEY, EMO, DIR, get pptx() { return pptx; } };
