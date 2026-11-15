// Numbers for the slides, read from the saved analysis (research_paper/output/intensity_analysis.json, written by
// scripts/intensity_analysis.py). Nothing is typed in by hand except the facts listed in facts.md as "fixed".
const fs = require("fs");
const path = require("path");

const PAPER = path.resolve(__dirname, "../..");
// INTENSITY_JSON / INTENSITY_FIGS (lib.js) only for dry runs, e.g. a simulated Gemma result; normally unset
const D = JSON.parse(fs.readFileSync(process.env.INTENSITY_JSON || path.join(PAPER, "output/intensity_analysis.json"), "utf8"));
const C = D.contrast_rows;

// Benchmark display names (config/models.yaml), in benchmark order
const NAMES = {
  emotion2vec_plus_large: "emotion2vec+ large", meralion_ser_v1: "MERaLiON-SER-v1", melati_emotion: "Melati",
  vox_profile_whisper: "Vox-Profile Whisper", vox_profile_wavlm: "Vox-Profile WavLM", qwen2_5_omni_7b: "Qwen2.5-Omni 7B",
  qwen2_audio_7b: "Qwen2-Audio 7B Instruct", gemma4_e4b: "Gemma 4 E4B", gpt_realtime: "GPT Realtime",
  "gemini-3.8-flash": "Gemini 3.8 Flash",
};
const SHORT = { ...NAMES, emotion2vec_plus_large: "emotion2vec+", meralion_ser_v1: "MERaLiON", qwen2_5_omni_7b: "Qwen2.5-Omni",
  qwen2_audio_7b: "Qwen2-Audio", gemma4_e4b: "Gemma 4", "gemini-3.8-flash": "Gemini" };
const DS = { crema_d: "CREMA-D", ravdess: "RAVDESS", mead: "MEAD", thai_ser: "THAI-SER", mcae_spps: "MCAE-SPPS", dailytalk: "DailyTalk" };
const ACTED = ["crema_d", "ravdess", "mead", "thai_ser"];

const rows = (ds) => C.filter((r) => r.dataset === ds);
const get = (ds, m) => C.find((r) => r.dataset === ds && r.model === m);
const cls = (lo, hi) => (lo == null || hi == null ? "withheld" : lo > 0 ? "up" : hi < 0 ? "down" : "zero");
const adj = (r) => cls(r.adjusted_ci_lower_pp, r.adjusted_ci_upper_pp);
const mat = (r) => (r.matched_pairs ? cls(r.matched_ci_lower_pp, r.matched_ci_upper_pp) : null);
const count = (list, f, v) => list.filter((r) => f(r) === v).length;

// Formatting: signed whole percentage points with a real minus sign; whole percent; thousands separators
const pp = (x) => { const v = Math.round(x); return v === 0 ? "0" : (v > 0 ? "+" : "−") + Math.abs(v); };
const pct = (x) => `${Math.round(100 * x)}%`;
const int = (x) => Number(x).toLocaleString("en-US");
const range = (list, key) => {
  const v = list.map((r) => r[key]).filter((x) => x != null);
  return [Math.min(...v), Math.max(...v)];
};
const rangeText = (list, key) => { const [a, b] = range(list, key); return `${pp(a)} to ${pp(b)} pp`; };

const models = [...new Set(C.map((r) => r.model))];
const gemma = models.includes("gemma4_e4b");
const scope = Object.fromEntries(D.scope_rows.map((r) => [r.dataset, r]));
const clusters = {};
D.effect_rows.forEach((r) => { clusters[r.dataset] = Math.max(clusters[r.dataset] || 0, r.nclusters); });
const levels = (ds, m) => D.level_rows.filter((r) => r.dataset === ds && r.model === (m || models[0]));
const smallStrata = D.level_rows.filter((r) => r.model === models[0] && r.n < D.manifest.parameters.display_min_level_n);

const acted = C.filter((r) => ACTED.includes(r.dataset));
const S = {
  pairs: C.length, models: models.length, gemma,
  observedUp: C.filter((r) => r.observed_change_pp > 0).length,
  actedFits: acted.length,
  actedUp: count(acted, adj, "up"), actedZero: count(acted, adj, "zero"), actedDown: count(acted, adj, "down"),
  byDataset: Object.fromEntries(Object.keys(DS).map((ds) => {
    const list = rows(ds);
    return [ds, { n: list.length, observedUp: list.filter((r) => r.observed_change_pp > 0).length,
      adjUp: count(list, adj, "up"), adjZero: count(list, adj, "zero"), adjDown: count(list, adj, "down"),
      adjWithheld: count(list, adj, "withheld"),
      matUp: count(list, mat, "up"), matZero: count(list, mat, "zero"), matDown: count(list, mat, "down"),
      matched: list.some((r) => r.matched_pairs), pairs: Math.max(0, ...list.map((r) => r.matched_pairs || 0)),
      neutralMajority: list.filter((r) => r.low_neutral_rate > 0.5).length,
      low: list.length ? list[0].low_label.replace("ECSS level ", "level ") : "",
      high: list.length ? list[0].high_label.replace("ECSS level ", "level ") : "" }];
  })),
};

// Qwen2-Audio on MEAD: recall per reference class and level, and the share of "angry" answers
function meadClasses(model = "qwen2_audio_7b") {
  const list = D.class_rows.filter((r) => r.dataset === "mead" && r.model === model);
  const ids = [...new Set(list.map((r) => r.level_id))].sort();
  const labels = ids.map((i) => list.find((r) => r.level_id === i).level_label);
  const classes = [...new Set(list.map((r) => r.gold))].sort();
  const cell = (g, i) => list.find((r) => r.gold === g && r.level_id === i);
  const lv = levels("mead", model);
  const angryShare = ids.map((i) => lv.find((r) => r.level_id === i).predicted_angry_rate);
  const total = ids.map((i) => lv.find((r) => r.level_id === i).n);
  const angryClips = ids.map((i) => cell("angry", i).n);
  return { ids, labels, classes, cell, angryShare, total, angryClips, accuracy: ids.map((i) => lv.find((r) => r.level_id === i).accuracy) };
}

module.exports = { D, C, NAMES, SHORT, DS, ACTED, rows, get, adj, mat, cls, pp, pct, int, range, rangeText, models, gemma,
  scope, clusters, levels, smallStrata, S, meadClasses };
