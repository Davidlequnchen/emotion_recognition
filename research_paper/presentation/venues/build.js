// Build the deck: node build.js [out.pptx]  ->  target_venues.pptx
// All facts are typed in slides.js from research_paper/docs/venues.md; update slides.js when that document changes.
const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");
const L = require("./lib");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Chen Lequn";
pptx.title = "Target venues for the benchmark paper";
pptx.subject = "ACL 2027, EMNLP 2027, Interspeech 2027 and IEEE ASLI 2027 (from research_paper/docs/venues.md)";
L.init(pptx);

require("./slides")(L);           // title, venue table (2 slides), special requirements

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

const out = process.argv[2] || "target_venues.pptx";
(async () => {
  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(await pptx.write({ outputType: "nodebuffer" }));
  let total = 0;
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/(slides|notesSlides)\/[^/]+\.xml$/.test(n))) {
    const [xml, fixed] = cleanParagraphs(await zip.file(name).async("string"));
    total += fixed;
    zip.file(name, xml);
  }
  // pptxgenjs lists a slide master per slide in [Content_Types].xml but writes only one: drop the entries without a part
  const types = await zip.file("[Content_Types].xml").async("string");
  zip.file("[Content_Types].xml", types.replace(/<Override PartName="\/([^"]+)"[^>]*\/>/g, (m, part) => (zip.file(part) ? m : "")));
  const buf = await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" });
  fs.writeFileSync(path.resolve(__dirname, out), buf);
  console.log("wrote", out, `(${L.total} slides; removed ${total} repeated paragraph properties)`);
})();
