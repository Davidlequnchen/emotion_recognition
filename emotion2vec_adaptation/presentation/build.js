// Build the deck: node build.js  ->  emotion2vec_adaptation.pptx
const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");
const L = require("./lib");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Chen Lequn";
pptx.title = "Adapting emotion2vec+ large to customer calls";
pptx.subject = "emotion2vec_adaptation: frozen-encoder head and continued self-supervised pretraining";
L.init(pptx);

require("./s_intro")(L);        // title, problem, two paths, data sources, overview
require("./s_head")(L);         // Path A: architecture and training recipe
require("./s_ssl")(L);          // Path E: encoder, teacher-student, masking, equations, one step, original vs ours
require("./s_ops")(L);          // encoder + head, leakage, data scale, B200, monitoring, validation, plan
require("./s_appendix")(L);     // hidden appendix: settings, data-scale numbers, code map, references

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

const out = process.argv[2] || "emotion2vec_adaptation.pptx";
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
