// Build the deck: node build.js  ->  consensus_analysis.pptx
// Numbers come from assets/numbers.json and figures from fig/, both written by scripts/make_assets.py from the
// saved consensus outputs (research_paper/output/). Rebuild everything with ./rebuild.sh.
const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");
const L = require("./lib");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Chen Lequn";
pptx.title = "Model consensus as a pseudo-label filter";
pptx.subject = "Accuracy and coverage of agreed labels in the SER benchmark (public datasets only)";
L.init(pptx);

const numbers = path.join(__dirname, "assets", "numbers.json");
if (!fs.existsSync(numbers)) { console.error("assets/numbers.json is missing: run python scripts/make_assets.py"); process.exit(1); }
const N = JSON.parse(fs.readFileSync(numbers, "utf8"));

require("./s_intro")(L, N);       // title, problem, trade-off, definitions, vote rules, worked example, groups
require("./s_results")(L, N);     // results, summary, caveats, next steps, hidden appendix

if (!L.checkSizes()) { console.error("font size check failed"); process.exit(1); }
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

const out = "consensus_analysis.pptx";
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
  fs.writeFileSync(path.join(__dirname, out), buf);
  console.log("wrote", out, `(${L.total} slides: ${L.slideNo} shown + ${L.appendixNo} hidden appendix; removed ${total} repeated paragraph properties)`);
})();
