// Build the deck: node build.js  ->  turn_segmentation.pptx
const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");
const L = require("./lib");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Chen Lequn";
pptx.title = "Turn segmentation for emotion labelling";
pptx.subject = "preparation_tools: forced alignment, Silero VAD and dynamic programming";
L.init(pptx);

// Numbers computed by scripts/make_figures.py and scripts/make_data_figures.py (see facts.md)
const read = (f) => JSON.parse(fs.readFileSync(path.join(__dirname, "assets", f), "utf8"));
const A = { worked: read("worked_example.json"), data: read("data_stats.json"), sens: read("sensitivity.json"),
  audio: JSON.parse(fs.readFileSync(path.join(__dirname, "assets", "audio", "manifest.json"), "utf8")) };

require("./s_intro")(L, A);       // title, problem, inputs, building blocks, pipeline
require("./s_align")(L, A);       // prepare, language, forced alignment, text, quality checks
require("./s_vad")(L, A);         // Silero VAD: model, its own segments, how we use it
require("./s_dp")(L, A);          // boundary score, DP, worked example, routing to four folders
require("./s_results")(L, A);     // outputs, call context, validation, settings, deployment, limitations, references

if (!L.checkSizes()) { console.error("font size check failed"); process.exit(1); }
// pptxgenjs writes one <a:pPr> per text run, so a paragraph with several runs (bold + normal text, links) gets
// repeated paragraph properties, which the OOXML schema does not allow (LibreOffice then drops the bullet).
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

const out = process.argv[2] || "turn_segmentation.pptx";
(async () => {
  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(await pptx.write({ outputType: "nodebuffer" }));
  let total = 0;
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/(slides|notesSlides)\/[^/]+\.xml$/.test(n))) {
    const [xml, fixed] = cleanParagraphs(await zip.file(name).async("string"));
    total += fixed;
    zip.file(name, xml);
  }
  const buf = await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" });
  fs.writeFileSync(path.join(__dirname, out), buf);
  console.log("wrote", out, `(${L.total} slides: ${L.slideNo} shown + ${L.appendixNo} hidden appendix; removed ${total} repeated paragraph properties)`);
})();
