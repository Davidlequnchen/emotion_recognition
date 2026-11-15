// Build the deck: node build.js  ->  cross_dataset_analysis.pptx
// Numbers come from assets/facts.json (scripts/make_facts.py, from research_paper/output/cross_dataset_*) and the
// figures from research_paper/analysis_report/figures/ (research_paper/scripts/cross_dataset_plot.py), so the
// deck never holds a number or a figure of its own. Rebuild everything with ./rebuild.sh.
const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");
const L = require("./lib");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Chen Lequn";
pptx.title = "Why SER models win on some datasets and lose on others";
pptx.subject = "Cross-dataset analysis of the SER benchmark: language, conversational vs acted speech, clip duration";
L.init(pptx);

const factsFile = path.join(__dirname, "assets", "facts.json");
if (!fs.existsSync(factsFile)) { console.error("assets/facts.json is missing: run python scripts/make_facts.py"); process.exit(1); }
const D = JSON.parse(fs.readFileSync(factsFile, "utf8"));

// Helpers shared by the slide files: F = formatted fact, V = raw value, list = "A, B and C", claim = wording guard.
const F = (k) => { if (!(k in D.facts)) throw new Error(`unknown fact ${k}`); return D.facts[k].text; };
const V = (k) => { if (!(k in D.facts)) throw new Error(`unknown fact ${k}`); return D.facts[k].value; };
const list = (a) => (a.length <= 1 ? a.join("") : `${a.slice(0, -1).join(", ")} and ${a[a.length - 1]}`);
const signed = (v, d = 1) => { if (v === null || v === undefined) return "—"; let t = (v >= 0 ? "+" : "") + v.toFixed(d); if (Number(t) === 0) t = (0).toFixed(d); return t.replace("-", "−"); };
const warnings = [];
// The fixed wording around the numbers was written for the current results. A claim that no longer holds after a
// rebuild (e.g. when Gemma 4 E4B arrives) is printed, so the slide text can be reviewed before the deck is shared.
const claim = (ok, message) => { if (!ok) warnings.push(message); };
const H = { D, F, V, list, signed, claim, gemmaPending: !V("gemma_complete") };

require("./s_intro")(L, H);       // title, problem, how we test, datasets, clip durations
require("./s_results")(L, H);     // setting, training exposure, language, duration (3 slides)
require("./s_end")(L, H);         // summary, limitations, next steps, hidden appendix

if (!L.checkSizes()) { console.error("font size check failed"); process.exit(1); }
warnings.forEach((w) => console.log(`  CHECK WORDING: ${w}`));
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

const out = process.argv[2] || "cross_dataset_analysis.pptx";
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
  console.log("wrote", out, `(${L.total} slides: ${L.slideNo} shown + ${L.appendixNo} hidden appendix; removed ${total} repeated paragraph properties; ${warnings.length} wording checks failed)`);
})();
