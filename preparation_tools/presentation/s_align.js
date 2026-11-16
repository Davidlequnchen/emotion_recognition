// Prepare, language check, forced alignment (concept, tool, CTC), text normalisation, quality checks
const R = require("./refs");

module.exports = function (L) {
  const { C, W, H, MX, CW, TOP, FONT } = L;
  const p = L.pptx;

  // ---- 6 Prepare
  {
    const s = L.slide("", "Step 1 · Prepare the audio", { stage: "Prepare" });
    const y = TOP + 0.25, bw = 2.6, bh = 1.15, gap = 0.55;
    const steps = [
      ["Source audio\nWAV · MP3 · FLAC\nkept for export", C.panel, C.ink],
      ["Mono 16 kHz\nanalysis copy", C.blueFill, C.accent],
      ["Silero VAD\np(speech) every 32 ms", C.blueFill, C.accent],
    ];
    steps.forEach(([t, f, l], i) => {
      const x = MX + i * (bw + gap);
      L.box(s, x, y, bw, bh, t, { fill: f, line: l, fontSize: 18 });
      L.arrow(s, x + bw, y + bh / 2, x + bw + gap, y + bh / 2, { width: 1.75 });
    });
    const ox = MX + 3 * (bw + gap);
    L.box(s, ox, y, W - MX - ox, bh, "→ Step 2:\nlanguage check", { fill: C.white, line: C.ink, dash: "dash", fontSize: 18 });
    L.bullets(s, [
      "Clips are cut from the untouched source, at its own sample rate",
      "The 16 kHz mono copy is only for VAD and alignment",
      "Edge-silence trim: off by default (VAD can miss quiet speech, crying, sighs)",
    ], { y: 3.55, h: 3.2, fontSize: 22, gap: 12 });
    L.notes(s, `The first step prepares the turn. The source file is decoded once, with soundfile, which reads WAV, FLAC and MP3. The source samples are kept as they are: at the end, the clips are cut from this source at its own sample rate and channel count, so nothing is resampled in the exported audio.
    For analysis we make a separate mono copy at 16 kHz, because both Silero VAD and the wav2vec 2.0 aligner expect 16 kHz. Telephone audio is often 8 kHz; upsampling does not bring back the missing high frequencies, but alignment only needs approximate word positions, and we cut inside pauses anyway.
    Silero VAD then runs once over the whole turn and gives a speech probability every 32 milliseconds. These probabilities are used later to judge pauses and to measure how much of each segment is speech.
    There is an option to trim long silence at the very start and end of a turn. It is switched off by default, because VAD can miss quiet speech, crying or sighs at the edges, and deleting those would remove emotional content.`);
  }

  // ---- 7 Language check, then the 10 s rule
  {
    const s = L.slide("", "Step 2 · Language first, then the 10 s rule", { stage: "Language" });
    const y = TOP + 0.1;
    L.box(s, MX, y, 2.4, 1.3, "Transcript text", { fill: C.panel, line: C.ink, fontSize: 20 });
    L.arrow(s, MX + 2.4, y + 0.65, MX + 2.85, y + 0.65);
    L.box(s, MX + 2.85, y, 4.0, 1.3, "Count cue words\nSpanish (+ á é í ó ú ñ ü ¿ ¡)\nvs English function words", { fill: C.blueFill, line: C.accent, fontSize: 18 });
    L.arrow(s, MX + 6.85, y + 0.65, MX + 7.3, y + 0.65);
    const rx = MX + 7.3, rw = W - MX - rx;
    const out = [["Spanish share ≥ 0.8", "es"], ["≤ 0.2, or fewer than 2 cue words", "en"], ["in between", "mixed"]];
    out.forEach(([cond, lang], i) => {
      const yy = y - 0.1 + i * 0.52;
      L.box(s, rx, yy, rw - 1.1, 0.44, cond, { fill: C.white, line: C.line, fontSize: 16, align: "left" });
      L.box(s, rx + rw - 1.0, yy, 1.0, 0.44, lang, { fill: C.amberFill, line: C.warn, fontSize: 18, bold: true });
    });
    const nonEn = { text: "non_english/ : the whole turn, any length", bold: true, fill: C.amberFill, options: { colspan: 2, align: "center" } };
    L.table(s, ["Language", "Turn ≤ 10 s", "Turn > 10 s"], [
      ["en", { text: "final/ : the whole turn", bold: true, fill: C.greenFill }, "forced alignment and cutting"],
      ["es", nonEn],
      ["mixed", { ...nonEn }],
    ], { y: 3.3, colW: [1.8, 5.0, 5.33], fontSize: 20, rowH: 0.55 });
    L.text(s, "The English model must not align Spanish text; a Spanish model can be added later.", MX, 5.65, CW, 0.5, { fontSize: 20 });
    L.notes(s, `The language check now comes first, before any length rule. Most calls are English, but a minority are Spanish, and the English alignment model must never be given Spanish text, because it would produce confident-looking but wrong word times.
    A small built-in check looks at the transcript only. It counts common Spanish function words, Spanish accented letters and the inverted marks, against common English function words. Words shared by both languages are left out of both lists, and a capitalised accented word such as José counts as a name, not as Spanish. If at least 80 percent of the cue words are Spanish, the turn is Spanish. If at most 20 percent are Spanish, or there are fewer than two cue words, it is English, which is the default. In between it is mixed.
    The table shows the routing. Every Spanish or mixed turn, short or long, goes whole into its own folder, non_english, with no segmentation; it keeps its call and turn identity, so it still sits in the right place on the call timeline. English turns of 10 seconds or less are already one or two sentences, so they go whole into final. Only English turns longer than 10 seconds go on to forced alignment and cutting.
    Adding Spanish later only needs a Spanish alignment model, a line in the settings and a Spanish text normaliser.`);
  }

  // ---- 8 Forced alignment: concept
  {
    const s = L.slide("", "Step 3 · Forced alignment: the text is known, only the timing is missing", { stage: "Align", titleSize: 28 });
    const y1 = TOP + 0.15, y2 = TOP + 1.65, bh = 1.0;
    L.text(s, "Speech recognition", MX, y1 + 0.1, 2.2, 0.8, { fontSize: 20, bold: true, valign: "middle" });
    L.box(s, MX + 2.3, y1, 1.9, bh, "audio", { fill: C.panel, line: C.ink, fontSize: 18 });
    L.arrow(s, MX + 4.2, y1 + bh / 2, MX + 4.7, y1 + bh / 2);
    L.box(s, MX + 4.7, y1, 3.1, bh, "ASR model\n“What was said?”", { fill: C.orangeFill, line: C.accent2, fontSize: 18 });
    L.arrow(s, MX + 7.8, y1 + bh / 2, MX + 8.3, y1 + bh / 2);
    L.box(s, MX + 8.3, y1, W - MX - (MX + 8.3), bh, "a new text, which can differ\nfrom our transcript", { fill: C.white, line: C.accent2, fontSize: 18 });
    L.text(s, "Forced alignment", MX, y2 + 0.1, 2.2, 0.8, { fontSize: 20, bold: true, valign: "middle", color: C.ink });
    L.box(s, MX + 2.3, y2, 1.9, bh, "audio +\ntranscript", { fill: C.panel, line: C.ink, fontSize: 18 });
    L.arrow(s, MX + 4.2, y2 + bh / 2, MX + 4.7, y2 + bh / 2);
    L.box(s, MX + 4.7, y2, 3.1, bh, "aligner\n“When was each word said?”", { fill: C.blueFill, line: C.accent, fontSize: 18 });
    L.arrow(s, MX + 7.8, y2 + bh / 2, MX + 8.3, y2 + bh / 2);
    L.box(s, MX + 8.3, y2, W - MX - (MX + 8.3), bh, "Yeah. 0.16–0.50 s\nUmm 1.24–1.34 s · …", { fill: C.white, line: C.accent, fontSize: 18 });
    L.bullets(s, [
      "No re-recognition: our transcript stays the text",
      "Never split text by word count or share of the duration",
      "torchaudio forced_align + WAV2VEC2_ASR_BASE_960H (English, 94.4M, MIT)",
    ], { y: 4.55, h: 2.3, fontSize: 22, gap: 10 });
    L.notes(s, `To cut audio and text at the same place we need to know when each word was spoken, and our transcripts have no word times. Forced alignment adds them.
    The top row is speech recognition: the model hears the audio and has to work out what was said. Its output is a new text, which can differ from the transcript we have. The bottom row is forced alignment: the model is given the audio and the words that were said, and only has to find where each word sits in time. Its output is a start time, an end time and a confidence for every word; the example shows the first two words of a synthetic DailyTalk turn, d418_long_en. Because the text is fixed, this is much simpler and more robust than recognition, and our transcript stays the text of record.
    One tempting shortcut is to split the text in proportion to the audio, for example the first 20 percent of words go to the first 20 percent of the audio. That is wrong whenever the speaking rate changes, which is exactly the case for slow, pausy speakers, so we never do it.
    The tool is torchaudio's forced_align function with the WAV2VEC2_ASR_BASE_960H model: an English wav2vec 2.0 base model with 94.4 million parameters, fine-tuned for character recognition on the 960 hours of LibriSpeech, and published under the MIT licence.`);
    L.cite(s, "", [[R.torchaudio.short, R.torchaudio.url], [R.wav2vec2.short, R.wav2vec2.url], [R.librispeech.short, R.librispeech.url]]);
  }

  // ---- 9 Tool choice
  {
    const s = L.slide("", "Why torchaudio’s forced_align with the English wav2vec 2.0 model", { stage: "Align", titleSize: 28 });
    L.table(s, ["Option", "Licence", "Verdict"], [
      [{ text: "forced_align + WAV2VEC2_ASR_BASE_960H", bold: true }, "MIT", { text: "default: nothing extra to install", bold: true, fill: C.greenFill }],
      ["forced_align + WAV2VEC2_ASR_LARGE_LV60K_960H", "MIT", "upgrade if base aligns poorly"],
      ["forced_align + MMS_FA (multilingual)", { text: "CC BY-NC 4.0", bold: true }, { text: "not for commercial use", fill: C.redFill }],
      ["WhisperX align()", "same model", "no gain; heavy dependencies"],
      ["Montreal Forced Aligner", "check model", "fallback only (needs a dictionary)"],
    ], { y: TOP + 0.1, colW: [6.3, 1.9, 3.93], fontSize: 20, rowH: 0.66 });
    L.notes(s, `This table is the tool decision from the method document.
    The default is torchaudio's built-in forced_align function with the English base wav2vec 2.0 model. Its weights are MIT-licensed, and torchaudio 2.7.1 is already in our environment, so nothing extra needs to be installed. If the base model places words poorly on real calls, the larger LV60K model is the upgrade, also MIT.
    MMS_FA is just as convenient, uses the same function and is multilingual, but its weights are published under CC BY-NC 4.0, which forbids commercial use, and internal business use is commercial. Our calls are English, so the English MIT model loses nothing.
    WhisperX uses exactly the same English wav2vec 2.0 model for alignment, so it would give the same quality, but it pins a different torch version and pulls in pyannote, faster-whisper and more. The Montreal Forced Aligner is a separate Kaldi-based tool that needs a pronunciation dictionary; it stays a fallback in case both wav2vec 2.0 models fail the pilot.
    For completeness, CTC-Segmentation is a related method for aligning very long recordings with their text, for example whole audiobooks. Our turns are already short and diarized, so plain forced alignment is enough.`);
    L.cite(s, "", [[R.whisperx.short, R.whisperx.url], [R.mfa.short, R.mfa.url], [R.ctcseg.short + ", related", R.ctcseg.url]]);
  }

  // ---- 10 CTC forced alignment
  {
    const s = L.slide("", "How CTC forced alignment finds the word times", { stage: "Align" });
    const fig = L.image(s, "fig/ctc_trellis.png", W - MX - 6.6, TOP - 0.05, 6.6, 4.75, { alt: "CTC trellis for the word YES with the best path" });
    L.bullets(s, [
      "Every 20 ms: a probability for each of 29 labels (blank, |, A–Z, ')",
      "The transcript fixes the letter order",
      "Viterbi finds the best path through the trellis",
      "Letter spans → word start and end; confidence = mean letter probability",
    ], { x: MX, y: TOP + 0.1, w: fig.x - MX - 0.25, h: 4.8, fontSize: 22, gap: 14 });
    L.text(s, "Scores in the figure are illustrative; the path is the best path for them.", fig.x, fig.y + fig.h + 0.02, fig.w, 0.34, { fontSize: 16, align: "center" });
    L.notes(s, `How does forced alignment actually work? The acoustic model, wav2vec 2.0, looks at the audio and every 20 milliseconds outputs a probability for each of its 29 labels: a special blank label, the word separator, the 26 letters and the apostrophe. This is a CTC model, trained with Connectionist Temporal Classification, which is why it has the blank: blank means "no new letter in this frame".
    In recognition, the model would be free to choose any letters. In forced alignment, the transcript fixes which letters must appear and in which order. The figure is the classic CTC trellis for the word YES: time runs to the right, and the rows are the letters with optional blanks between them. Darker cells mean higher probability. A valid path starts at the top left, moves one frame to the right at every step, and either stays on its row or moves down to the next letter, skipping a blank if needed.
    The Viterbi algorithm, which is itself dynamic programming, finds the single most probable valid path. In torchaudio this is the forced_align function. Reading the path off gives letter spans: here Y covers frames 2 and 3, E frames 5 to 7 and S frames 9 and 10. Letters are grouped into words, which gives each word a start time, an end time and a confidence, the mean probability of its letters.
    The scores in this small figure are made up for illustration, but the drawn path is the true best path for those scores. In the real model the trellis has one column per 20 milliseconds of the turn.`);
    L.cite(s, "", [[R.ctc.short, R.ctc.url], [R.viterbi.short, R.viterbi.url], [R.tutorial.short, R.tutorial.url]]);
  }

  // ---- 11 Text normalisation
  {
    const s = L.slide("", "The aligner needs letters; the output keeps the original text", { stage: "Align" });
    L.table(s, ["Transcript token", "What the aligner gets"], [
      ["$50", "FIFTY DOLLARS"],
      ["3:05  ·  24/7", "THREE OH FIVE  ·  TWENTY FOUR SEVEN"],
      ["Café  ·  don't", "CAFE  ·  DON'T"],
      ["[NAME]  ·  ***  ·  (inaudible speech)", { text: "*  wildcard: matches any speech", bold: true }],
      ["A12B  ·  4420013  ·  555-1234  ·  1/2  ·  -50", { text: "*  wildcard: a code, or several readings", bold: true }],
    ], { x: MX, y: TOP + 0.05, colW: [5.6, 6.53], fontSize: 20, rowH: 0.6 });
    L.bullets(s, [
      "A wildcard is never next to a cut",
      "Each word keeps its character span: the output text is an exact slice of the original",
    ], { y: 5.4, h: 1.45, fontSize: 22, gap: 8 });
    L.notes(s, `The English model only knows the letters A to Z and the apostrophe, so it cannot align "$50" or "3:05" as written. The segmenter therefore works on a copy of the text. Each token is upper-cased, accents are removed, and numbers, money, percentages, ordinals, clock times and years are written out as words, as in the first rows of the table.
    Some tokens cannot be spelled reliably. Redaction placeholders such as [NAME] or three stars, and annotations in brackets or parentheses such as (inaudible speech), become one wildcard token, even if the annotation has several words. Codes like A12B, long digit strings, numbers with a leading zero, and numbers that have several possible readings, such as 555-1234, one half or minus 50, also become a wildcard. A wildcard matches any speech: in each frame it takes the best score of any non-blank label, which is how WhisperX handles unknown characters. Because a wildcard's edges are uncertain, a cut is never placed next to one.
    The original transcript is never changed. Every word remembers its exact character span in the original text, so when we cut, each output transcript is an exact slice of the original, with its spelling and punctuation kept. Only the aligner ever sees the upper-case copy.`);
    L.cite(s, "", [[R.whisperx.short, R.whisperx.url]]);
  }

  // ---- 12 Quality checks
  {
    const s = L.slide("", "Step 4 · Word confidence decides whether a turn may be cut", { stage: "Check" });
    L.bullets(s, [
      { runs: [{ text: "Turn: ", options: { bold: true } }, { text: "mean ≥ 0.5 and ≤ 40% weak words, else the whole turn → alignment_issues/" }] },
      { runs: [{ text: "Gap: ", options: { bold: true } }, { text: "no cut next to a word below 0.3 or a wildcard" }] },
    ], { y: TOP, h: 1.3, fontSize: 22, gap: 6 });
    L.image(s, "fig/confidence.png", MX, 3.0, CW, 3.9, { alt: "Alignment confidence ranges for four transcript conditions" });
    L.notes(s, `After alignment we check whether the result can be trusted, at two levels.
    The turn check looks at the whole turn. If the mean word confidence is below 0.5, or more than 40 percent of the words are weak, meaning below 0.3, or could not be spelled, then the transcript and the audio probably disagree: an ASR error, agent speech, noise. The whole turn then goes into the alignment_issues folder, uncut. We never invent timings.
    The local check looks at each gap. A gap next to a word below 0.3, or next to a wildcard, is not allowed as a cut point, because that word's edges are uncertain. This also keeps cuts away from places where agent speech, which is not in the customer transcript, confused the aligner.
    Where do these thresholds come from? The figure shows a scratch experiment on the five synthetic DailyTalk turns. With the correct transcript the mean confidence per turn was 0.90 to 0.95, and almost no words were weak. With the same audio converted to 8 kHz mu-law telephone quality it was hardly lower, 0.88 to 0.94. With 30 percent of the words replaced it dropped to 0.63 to 0.74, with 18 to 28 percent weak words; such a turn still passes, and the local check keeps the cuts away from the wrong words. With a completely wrong transcript it was 0.27 to 0.36, with more than half the words weak, so the turn goes whole into alignment_issues. The thresholds sit between the last two conditions. They are provisional and must be set again from real calls in the pilot.
    A practical detail: audio longer than 64 seconds is run through wav2vec 2.0 in 60-second chunks with 2 seconds of context on each side, to limit memory; the frame count is the same as in one pass.`);
    L.cite(s, "Synthetic DailyTalk turns; README and config/segment_turns.yaml", [[R.dailytalk.short, R.dailytalk.url]]);
  }
};
