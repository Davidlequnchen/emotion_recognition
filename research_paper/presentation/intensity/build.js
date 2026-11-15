// Build the deck: node build.js  ->  intensity_analysis.pptx
// Numbers come from ../../output/intensity_analysis.json (data.js); figures from ../../analysis_report/figures/.
const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");
const L = require("./lib");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Chen Lequn";
pptx.title = "Is a stronger emotional expression easier to classify?";
pptx.subject = "research_paper: emotion intensity and speech emotion recognition accuracy";
L.init(pptx);

const data = require("./data");
require("./write_facts")(data);
require("./slides")(L, data);

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

const out = process.argv[2] || "intensity_analysis.pptx";
(async () => {
  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(await pptx.write({ outputType: "nodebuffer" }));
  let total = 0;
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/(slides|notesSlides)\/[^/]+\.xml$/.test(n))) {
    const [xml, fixed] = cleanParagraphs(await zip.file(name).async("string"));
    total += fixed;
    zip.file(name, xml);
  }
  // Remove content-type declarations for slide masters that pptxgenjs does not write.
  const types = await zip.file("[Content_Types].xml").async("string");
  zip.file("[Content_Types].xml", types.replace(/<Override PartName="\/([^"]+)"[^>]*\/>/g,
    (match, part) => zip.file(part) ? match : ""));
  const buf = await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" });
  fs.writeFileSync(path.resolve(__dirname, out), buf);
  console.log("wrote", out, `(${L.total} slides: ${L.slideNo} shown + ${L.appendixNo} hidden appendix; removed ${total} repeated paragraph properties)`);
})();
