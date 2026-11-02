// Build the deck: node build.js  ->  SER_evaluation.pptx
const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");
const L = require("./lib");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Chen Lequn";
pptx.title = "Speech Emotion Recognition: evaluation and next steps";
L.init(pptx);

const read = (f) => (fs.existsSync(path.join(__dirname, f)) ? JSON.parse(fs.readFileSync(path.join(__dirname, f), "utf8")) : null);
const A = require("./content")(L, { manifest: read("assets/manifest.json"), stats: read("assets/stats.json") });

require("./s_intro")(L, A);
if (fs.existsSync(path.join(__dirname, "s_data.js"))) require("./s_data")(L, A);
require("./s_models")(L, A);
require("./s_results")(L, A);
if (fs.existsSync(path.join(__dirname, "s_cases.js"))) require("./s_cases")(L, A);
require("./s_future")(L, A);
if (fs.existsSync(path.join(__dirname, "s_appendix.js"))) require("./s_appendix")(L, A);

const out = process.argv[2] || "SER_evaluation.pptx";
pptx.writeFile({ fileName: path.join(__dirname, out) }).then((f) => console.log("wrote", f));
