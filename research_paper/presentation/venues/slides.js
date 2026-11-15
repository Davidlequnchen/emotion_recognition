// Title, the merged venue table for the shortlist (two slides: when and where; how hard and how long), the special
// requirements, the additional candidates (four slides, same columns as the shortlist tables) and the review windows.
// Every fact comes from research_paper/docs/venues.md: shortlist and markers (top, 2026-11-09), ASRU/SLT predecessor
// data (section 1, 2026-11-14), venues at a glance (section 2a), acceptance, difficulty, paper types and special
// requirements (section 2b), the service contributor rule (section 2.3), length and readiness (section 2c), the
// additional candidates and review windows (section 2d, 2026-11-14), plan (section 4). Sections 2a/2b were rechecked
// 2026-11-06; rows marked "checked 2026-11-14" in the document were added or updated on that date.
// The "fit" column of the document is left out of the shortlist tables on purpose; for the additional candidates the
// document's one-line classification is shown as the role tag, and fit goes into the difficulty line.
// "estimate", "unverified" and "not found" are kept as in the document.
module.exports = function (L) {
  const { C, W, MX, CW } = L;
  const p = L.pptx;
  const ROLE = {                                  // the document's own markers and classifications; not a ranking
    main: { t: "main target", b: true, color: C.link },
    focus: { t: "focus candidate", b: true, color: C.darkGreen },
    fallback: { t: "fallback", b: true, color: C.darkOrange },
    late: { t: "late fallback", b: true, color: C.darkOrange },
    cond: { t: "conditional fallback", b: true, color: C.darkOrange },
    notrec: { t: "not recommended", b: true, color: C.darkRed },
    nowin: { t: "no window", b: true, color: C.darkRed },
  };
  const venue = (name, role) => [{ t: name, b: true }, ROLE[role]];
  const diff = (level, why) => [{ t: `Difficulty: ${level}`, b: true }, why];
  const CHECKED = "Source: research_paper/docs/venues.md; first checked 2026-11-03, tables rechecked 2026-11-06, shortlist 2026-11-09";
  const CHECKED_ASLI = "Source: research_paper/docs/venues.md; tables rechecked 2026-11-06, shortlist 2026-11-09, ASRU/SLT data 2026-11-14";
  const CHECKED_EXTRA = "Source: research_paper/docs/venues.md, section 2d and rows marked “checked 2026-11-14”; checked 2026-11-14";
  const RECOMMEND = `The document's recommendation: ACL 2027 through ARR (full paper 4 January 2027) is the main target, but only if a pilot establishes a coherent contribution with adequate validation, a qualified service contributor is confirmed and the company approves publication. Choose the venue and the paper length after the pilot. EMNLP 2027 (ARR May 2027, estimate) is the fallback for a revised submission with the earlier reviews; NeurIPS 2027 Evaluations and Datasets (early May, estimate) is the other route once the paper is out of ARR. Interspeech 2027 (9 February 2027) and IEEE ASLI 2027 (21 April 2027) are the focus candidates for a focused speech-specific contribution; they are alternatives, not a guaranteed sequence, and neither can run in parallel with ARR for the same paper.`;
  const EXTRA = `Classification of the additional candidates (venues.md section 2d, checked 14 November 2026; planning labels, not acceptance forecasts): NeurIPS 2027 Evaluations and Datasets is a fallback, because the track studies evaluation itself and a benchmark and evaluation paper like ours fits it better than the main track. ICASSP 2028 is a late fallback for a focused 4-page version; its deadline, mid or late September 2027 (estimate), comes after the ASLI and EMNLP decisions. ACM MM 2027 is a conditional fallback, only with a multimodal framing, and it competes with ASLI for the April window. The NeurIPS main track and AAAI-28 are not recommended, and EACL has no submission window before late 2027.`;

  // ---- Title
  {
    const s = L.blank();
    s.addShape(p.ShapeType.rect, { x: MX, y: 1.6, w: 0.09, h: 2.95, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
    L.text(s, "Target Venues for the Benchmark Paper", MX + 0.4, 1.5, 11.8, 0.85, { fontSize: 40, bold: true });
    L.text(s, "Shortlist of 2026-11-09: ACL 2027 is the main target, Interspeech 2027 and\nIEEE ASLI 2027 are focus candidates, EMNLP 2027 is the fallback", MX + 0.4, 2.45, 11.8, 1.0, { fontSize: 24, ls: 1.1 });
    L.text(s, "Chen Lequn  ·  research_paper  ·  November 2026", MX + 0.4, 3.8, 11.8, 0.42, { fontSize: 20 });
    L.text(s, "Additional candidates (2026-11-14): NeurIPS 2027, ICASSP 2028, ACM MM 2027, AAAI-28, EACL", MX + 0.4, 4.3, 11.8, 0.42, { fontSize: 20 });
    L.text(s, "ACL 2027 is conditional on the pilot result, a qualified service contributor and company approval.", MX, 5.25, CW, 0.4, { fontSize: 20, bold: true });
    L.text(s, "The markers show present interest, not a commitment, a quality ranking or an acceptance forecast.", MX, 5.7, CW, 0.4, { fontSize: 20 });
    L.text(s, "Source: research_paper/docs/venues.md (checked 2026-11-03, tables 2026-11-06, additions 2026-11-14)", MX, 6.15, CW, 0.4, { fontSize: 20 });
    L.notes(s, `This part summarises where we could submit the benchmark and analysis paper. All facts come from research_paper/docs/venues.md: first checked on 3 November 2026, with the venue tables rechecked on 6 November 2026 on the official pages, and the shortlist set on 9 November 2026. On 14 November 2026 the predecessor data of IEEE ASLI (ASRU and SLT) and five additional candidates were checked: NeurIPS 2027 (main track and the Evaluations and Datasets track), ICASSP 2028, ACM Multimedia 2027, AAAI-28 and EACL. Anything marked "estimate" is projected from the previous cycle; "unverified" means no official page confirmed it; "not found" means we searched and found no official figure.
    The shortlist has four venues: ACL 2027 as the main target, Interspeech 2027 and IEEE ASLI 2027 as focus candidates, and EMNLP 2027 as the fallback. These markers show our present interest. They are not a commitment to submit, a quality ranking or an acceptance forecast. The first three content slides cover the shortlist; then come four slides on the additional candidates, with the same columns, and one slide on the review windows.
    ${RECOMMEND}
    ${EXTRA}`);
  }

  // ---- Merged table, part 1: when, where, community, standing, deadline (sections 2a, 1)
  {
    const s = L.slide("Target venues", "ACL 2027 is the main target and has the earliest deadline");
    L.table(s, ["Venue", "Dates & location", "Field / community", "Standing & tier", "Submission deadline"], [
      [venue("ACL 2027", "main"),
        ["17–22 Aug 2027", "Kyoto, Japan", "main conf. 20–22 Aug"],
        ["NLP", "ACL family (ARR)"],
        [{ t: "CCF-A · h5 236", b: true }, "top-tier flagship of NLP"],
        [{ t: "4 Jan 2027 AoE", b: true }, "ARR January 2027 cycle", "commitment date TBA"]],
      [venue("EMNLP 2027", "fallback"),
        ["dates TBA", "“Mexico or Central America”"],
        ["NLP", "ACL family (ARR)"],
        [{ t: "CCF-B · h5 218", b: true }, "top-tier NLP, on par with ACL internationally"],
        [{ t: "late May 2027 (estimate)", b: true }, "ARR May 2027 cycle", "2026 cycle: 25 May 2026"]],
      [venue("Interspeech 2027", "focus"),
        ["29 Aug – 2 Sep 2027", "São Paulo, Brazil"],
        ["speech"],
        [{ t: "CCF-B · h5 112", b: true }, "flagship of speech science and technology"],
        [{ t: "9 Feb 2027", b: true }, "update until 16 Feb"]],
      [venue("IEEE ASLI 2027", "focus"),
        ["29 Nov – 4 Dec 2027", "Singapore"],
        ["speech, audio, language", "IEEE SPS"],
        [{ t: "CCF: not listed · no h5 yet", b: true }, "first edition: unknown", "predecessors ASRU, SLT:", "CCF-C; h5 34 and 44"],
        [{ t: "21 Apr 2027", b: true }, "update until 28 Apr", "decision 28 Jul 2027"]],
    ], { y: 1.6, colW: [2.05, 2.4, 1.95, 2.95, 2.78], fontSize: 18 });
    L.source(s, ["Deadlines: full paper  ·  CCF: China Computer Federation list, 7th edition (2026)  ·  h5: Google Scholar h5-index (2025)",
      CHECKED_ASLI].join("\n"), { h: 0.52 });
    L.notes(s, `One row per shortlisted venue; the role tags are the document's own markers. ACL 2027 is in Kyoto from 17 to 22 August 2027, CCF-A with an h5-index of 236, and its paper deadline comes first: the full paper is due on 4 January 2027, anywhere on earth, in the ARR January cycle, with no separate abstract; the commitment date is not announced. ARR reviews are reusable, so a paper that is not accepted can be committed to a later venue or revised for a later cycle, for example the May 2027 cycle for EMNLP 2027. EMNLP's dates and city are not announced; the workshop call says Mexico or Central America, and its deadline is an estimate from the May 2026 cycle.
    Interspeech 2027 in São Paulo has an official deadline of 9 February 2027. That falls inside the ARR review period, so the same paper cannot go to both. IEEE ASLI 2027 in Singapore is new: from 2027 it brings together ASRU and SLT, the specialist speech workshops of the IEEE Signal Processing Society. Its new name does not automatically inherit a CCF class or a settled acceptance rate, so ASLI's own standing stays "first edition, unknown". Its notification date is 28 July 2027 (official important-dates page).
    ASLI's predecessors, checked on 14 November 2026: ASRU and SLT are both CCF class C in the 7th edition (2026), numbers 16 and 17 in the graphics and multimedia conference list. Google Scholar Metrics (2025 edition, papers 2020 to 2024) gives ASRU an h5-index of 34 (h5-median 56) and SLT 44 (h5-median 71; SLT is number 9 in Scholar's computational linguistics list). For comparison, Interspeech has 112 and ICASSP 137. IEEE SPS calls SLT a flagship workshop of its speech and language technical committee, and IEEE calls ASRU a major flagship workshop: both are respected specialist venues, smaller than Interspeech or ICASSP. CCF classes are classification labels, not a complete measure of standing within the speech community. The links to the calls are on the next slide.
    ${RECOMMEND}
    Checked: venue tables rechecked on the official pages on 6 November 2026; shortlist of 9 November 2026; ASRU and SLT data on 14 November 2026 (IEEE SPS news pages, Google Scholar Metrics, the official CCF 2026 PDF).`);
  }

  // ---- Merged table, part 2: acceptance and difficulty, paper format, link (sections 2b, 2a, 1)
  {
    const s = L.slide("Target venues", "Main-conference rate ~19–22% at ACL and EMNLP, ~50% at Interspeech", { titleSize: 28 });
    L.table(s, ["Venue", "Acceptance rate & difficulty", "Paper format & types", "Link"], [
      [venue("ACL 2027", "main"),
        ["18.9% main, 17.8% Findings (2026)", diff("high", " — insight beyond numbers expected")],
        ["long 8 p or short 4 p", "+ unlimited references, Limitations", "+1 p if accepted; Findings automatic"],
        [{ t: "2027.aclweb.org", url: "https://2027.aclweb.org/calls/main/" }, "call for papers"]],
      [venue("EMNLP 2027", "fallback"),
        ["22.2% main, 17.3% Findings (2025)", diff("medium-high", " — same bar, 5 more months, ACL reviews reusable")],
        ["as ACL (ARR): long 8 p or short 4 p"],
        [{ t: "aclrollingreview.org", url: "https://aclrollingreview.org/dates" }, "ARR dates (EMNLP site not live)"]],
      [venue("Interspeech 2027", "focus"),
        ["49.9% (2025); ~40% (2026, estimate)", diff("low-medium", "; Long track: medium (< 30%)")],
        ["Regular 4 p + 2 p for references and acknowledgements only", "Long Paper track 8 p + 2 p"],
        [{ t: "interspeech2027.org", url: "https://interspeech2027.org/calls/call_for_papers/" }, "call for papers"]],
      [venue("IEEE ASLI 2027", "focus"),
        ["none yet (first edition)", "predecessors: ASRU 2025 47.4%,", "SLT 2026: rate not published", diff("unknown", "")],
        ["6 p + unlimited references, appendix", "Findings track for accepted papers not in the main programme"],
        [{ t: "2027.ieeeasli.org", url: "https://2027.ieeeasli.org/call-for-papers/" }, "call for papers"]],
    ], { y: 1.55, colW: [2.05, 3.8, 3.93, 2.35], fontSize: 18 });
    L.source(s, ["“estimate”: projected from the previous cycle  ·  p: pages  ·  difficulty: the document's judgement for our paper",
      CHECKED_ASLI].join("\n"), { h: 0.52 });
    L.notes(s, `Acceptance rates are the latest the document found. ACL 2026 accepted 18.9 percent of 12,148 submissions to the main conference and 17.8 percent to Findings; for ACL 2025 it was 20.3 and 16.7 percent of 8,360. EMNLP 2025 accepted 22.2 percent to the main conference and 17.3 percent to Findings of 8,174; a third-party figure of about 15 percent for the EMNLP 2026 main conference is unverified. Interspeech 2025 accepted 49.9 percent of the 2,332 reviewed papers; Interspeech 2026 had more than 3,400 submissions, and about 40 percent is an estimate from the archive count. ASLI has no rate because 2027 is its first edition.
    ASLI's predecessors, checked on 14 November 2026, are labelled as such and are not ASLI's own figures. ASRU 2025 in Honolulu (6 to 10 December 2025) received a record 447 submissions and accepted 212 papers, counting regular, special session, challenge and demo papers, an overall rate of 47.43 percent, with 385 registered participants (IEEE SPS news, December 2025). SLT 2026 in Palermo (13 to 16 December 2026), the last standalone SLT, received about 750 submissions, the largest number in its history and nearly double the previous edition (IEEE SPS news, August 2026); its accepted count is not published. For SLT 2024 in Macao, official submission and acceptance numbers were not found; "nearly double" implies roughly 375 to 400 submissions, which is our estimate. The predecessors' rate of about 47 percent suggests low-medium difficulty, but a new name, a wider scope and the Findings track may change it, so ASLI's difficulty stays unknown.
    Difficulty is the document's judgement for our paper: high for ACL, because reviewers want insight beyond numbers and the service contributor rule applies; medium-high for EMNLP, with the same bar but five more months for controls and the ACL reviews to reuse; low-medium for a regular Interspeech paper and medium for its Long Paper track, which targets under 30 percent; unknown for ASLI.
    Length: choose it after the pilots. A substantial study of context, matched human judgments and independent validation may need the ACL long format of 8 pages; one precise, well-supported finding may suit a 4-page short paper. ACL papers that are good enough but not chosen for the main conference are offered Findings automatically; ASLI copies the idea with its own Findings track. At Interspeech one work cannot go to both the Regular and the Long Paper track.
    ${RECOMMEND}
    Checked: section 2 of venues.md, rechecked on 6 November 2026; ASRU and SLT figures on 14 November 2026.`);
  }

  // ---- Special requirements (sections 2b, 2.3)
  {
    const s = L.slide("Target venues", "ARR needs a qualified service contributor for every paper");
    L.table(s, ["Rule", "ACL / EMNLP 2027 (ARR)", "Interspeech 2027", "IEEE ASLI 2027"], [
      [{ text: "Reviewing duty", bold: true },
        "each paper names a service contributor (5 reviews or 8 meta-reviews); without one: lottery, papers not drawn are desk rejected",
        "none found (author reviewing not mandatory in 2026)", "not stated yet"],
      [{ text: "Who qualifies", bold: true },
        { colspan: 3, lines: [[{ t: "ARR contributor: ", b: true }, "PhD holders, postdocs and faculty with ≥ 2 papers in major CL venues or Findings; industry researchers (≥ Master's) and PhD students with ≥ 3 papers in major CL venues, Findings or related venues (speech area: Interspeech, ICASSP; no workshops)"]] }],
      [{ text: "Mandatory parts", bold: true },
        "Limitations section (missing = desk reject); Responsible NLP checklist",
        "scientific reporting checklist", "none listed"],
      [{ text: "Profiles, arXiv", bold: true },
        "complete OpenReview profile per author ≤ 1 week after the deadline; preprints allowed",
        "double-blind; arXiv allowed (2026 rules)", "anonymity, arXiv: not stated yet"],
      [{ text: "Caps, presenting", bold: true },
        "≤ 20 papers per author, ≤ 5 as first author; one author presents (in person or online)",
        "≤ 2 as first author; presenting in person", "one author registers and presents"],
    ], { y: 1.55, colW: [1.95, 5.18, 2.7, 2.3], fontSize: 18 });
    L.text(s, [{ text: "No parallel submission: ", options: { bold: true } }, { text: "one venue at a time, also for the additional candidates" }], MX, 6.14, CW, 0.4, { fontSize: 20 });
    L.source(s, ["OpenReview: profiles registered with a public email are moderated for up to 2 weeks", CHECKED].join("\n"), { h: 0.52 });
    L.notes(s, `The designated service contributor is ARR's rule from the ACL Sustainable Reviewing Policy, in force for every ARR submission from the October 2026 cycle, so it applies to the January 2027 cycle for ACL and the May 2027 cycle for EMNLP. It is not an OpenReview or arXiv endorsement. Each submission must be endorsed by a qualified service contributor who does the reviewing for that paper in the same cycle: five reviews or eight meta-reviews. Usually that is an author; if no author qualifies, a non-author can be named but must vouch for the work. One person can cover at most two submissions and must register within 48 hours after the deadline. Papers without a contributor go to a lottery, and those not drawn are desk rejected; if the contributor drops out without a replacement, every paper listing them is desk rejected.
    Who qualifies: PhD holders, postdocs and faculty with at least two papers in major CL venues or Findings; industry researchers with at least a Master's degree and PhD students with at least three papers in major CL events, Findings or the related venues of their area, which for speech are Interspeech and ICASSP; workshops do not count. Whether an Interspeech or ICASSP paper counts for a submission in another area is unverified. What it means for us: count the authors' papers in these venues, settle the contributor before December, otherwise ask a qualified colleague to vouch and review or accept the lottery risk.
    Other ARR rules: the Limitations section is mandatory and a missing one is a desk reject; a wrong or incomplete Responsible NLP checklist can lead to a desk reject; every author needs a complete OpenReview profile with ORCID, affiliation history, emails and DBLP or Anthology links within one week after the deadline. A profile registered with an institutional or company email is active at once; one with a public email is moderated for up to two weeks, so create the profiles now. At most 20 submissions per author and 5 as first author; preprints are allowed; one author registers by early registration and presents in person or online; clearly sub-standard papers can be desk rejected early.
    Interspeech: at most two papers as first author, double-blind review, arXiv allowed under the 2026 rules, a scientific reporting checklist, in-person presentation with no-shows removed, ISCA copyright, no parallel submission, and fully AI-written papers or fake references are desk rejected. ASLI: one author registers and presents per paper; anonymity, arXiv policy, reviewer duty and caps are not stated yet; its call asks for papers not currently under review elsewhere.
    No parallel submission: the same paper cannot be under review in ARR, Interspeech and ASLI at once, and the additional candidates have the same rule: ICASSP does not permit multiple simultaneous submissions, ACM MM rules out papers under review elsewhere including an active ARR submission, NeurIPS forbids another archival venue while under review, and AAAI forbids concurrent submissions. The review-window slide at the end shows which deadlines remain reachable after each decision.
    The gating conditions stay the same: the pilot result, a qualified service contributor and company approval for publication. Checked: section 2.3 and the rules table of venues.md, rechecked on 6 November 2026; parallel-submission rules of the additional candidates on 14 November 2026.`);
  }

  // ---- Additional candidates, part 1a: fallbacks (sections 2a, 2d)
  const COLS1 = ["Venue", "Dates & location", "Field / community", "Standing & tier", "Submission deadline"];
  const COLW1 = [2.3, 2.3, 2.35, 2.45, 2.73];
  {
    const s = L.slide("Additional candidates", "Fallbacks: NeurIPS 2027 E&D, ICASSP 2028 and ACM MM 2027", { titleSize: 28 });
    L.table(s, COLS1, [
      [venue("NeurIPS 2027 E&D", "fallback"),
        ["dates TBA, “Europe”", "2026: Sydney"],
        ["machine learning", "Evaluations & Datasets track"],
        [{ t: "CCF-A · h5 371", b: true }, "top-tier ML flagship"],
        [{ t: "early May 2027 (estimate)", b: true }, "2026: 4 May / 6 May"]],
      [venue("ICASSP 2028", "late"),
        ["28 May – 2 Jun 2028", "Tokyo, Japan"],
        ["signal processing,", "speech and audio"],
        [{ t: "CCF-B · h5 137", b: true }, "flagship of signal processing"],
        [{ t: "mid/late Sep 2027 (estimate)", b: true }, "2027 cycle: 23 Sep 2026"]],
      [venue("ACM MM 2027", "cond"),
        ["8–12 Nov 2027", "Hong Kong"],
        ["multimedia", "topic “Emotional and Social Signals”"],
        [{ t: "CCF-A · h5 119", b: true }, "top multimedia conference"],
        [{ t: "~1 Apr 2027 (estimate)", b: true }, "2026: 25 Mar / 1 Apr"]],
    ], { y: 1.6, colW: COLW1, fontSize: 18 });
    L.text(s, [{ text: "NeurIPS E&D: ", options: { bold: true } }, { text: "studies evaluation itself, so it fits our paper better than the main track" }], MX, 5.2, CW, 0.4, { fontSize: 20 });
    L.text(s, [{ text: "ICASSP 2028: ", options: { bold: true } }, { text: "after the ASLI and EMNLP decisions  ·  " }, { text: "ACM MM: ", options: { bold: true } }, { text: "only if multimodal; competes with ASLI" }], MX, 5.65, CW, 0.4, { fontSize: 20 });
    L.source(s, ["Dates “x / y”: abstract / full paper  ·  CCF: 7th edition (2026)  ·  h5: Google Scholar (2025)  ·  tags: our classification",
      CHECKED_EXTRA].join("\n"), { h: 0.52 });
    L.notes(s, `These are venues checked on 14 November 2026 beyond the shortlist. The role tags are the document's classification in section 2d: fallback means the venue could take the paper if the shortlisted route fails or the contribution changes. They are planning labels, not acceptance forecasts. The next slide has the venues that are not recommended; the two slides after that give acceptance rates, formats and links.
    NeurIPS 2027: the location is given only as Europe on the NeurIPS future-meetings page; NeurIPS 2026 is in Sydney from 6 to 12 December with satellite meetings. Since 2026 the official track name is Evaluations and Datasets, which replaces the Datasets and Benchmarks track; the call says the track studies the processes, datasets and benchmarks used to test and compare AI systems, and it welcomes analyses of existing benchmarks and negative results. Accepted papers appear in the NeurIPS proceedings alongside the main track (E&D FAQ). NeurIPS is CCF-A; the h5-index of 371 is for the whole conference. The deadline is an estimate from 2026, when the abstract was due on 4 May and the full paper on 6 May, with notification on 24 September. A benchmark and evaluation paper like ours fits this track better than the main track.
    ICASSP 2028: the official site gives 28 May to 2 June 2028 in Tokyo; the IEEE SPS event page says 28 May to 3 June. The deadline is not announced ("coming soon"); mid or late September 2027 is an estimate from the last two cycles, 17 September 2025 for ICASSP 2026 and 23 September 2026 for ICASSP 2027. CCF-B, h5-index 137. It is the late fallback for a focused 4-page version, because its deadline comes after the ASLI decision (28 July 2027) and the EMNLP decision (late August, estimate).
    ACM Multimedia 2027: 8 to 12 November 2027 at AsiaWorld-Expo, Hong Kong (official site); no call yet. The deadline is an estimate from 2026: registration 25 March, paper 1 April, notification 9 July. CCF-A, h5-index 119. The topic list includes Emotional and Social Signals, but the 2026 call says papers that make multimedia or multimodal research contributions will be preferred, so it is a fallback only with a multimodal framing, for example dialogue audio plus transcript context. Its review, about April to early July, blocks ASLI (21 April), so choose MM or ASLI for the April window.
    Not to be confused: MMM, the International Conference on MultiMedia Modeling, is a different, smaller venue (CCF-C); see the next slide.
    Checked 14 November 2026: NeurIPS future meetings, 2026 calls, handbook and E&D FAQ; ICASSP 2028 official site and SPS event page; 2027.acmmm.org; Google Scholar Metrics; the official CCF 2026 PDF.`);
  }

  // ---- Additional candidates, part 1b: not recommended (sections 2, 2a, 2d)
  {
    const s = L.slide("Additional candidates", "Not recommended: NeurIPS main track, AAAI-28; EACL has no window", { titleSize: 28 });
    L.table(s, COLS1, [
      [venue("NeurIPS 2027 main", "notrec"),
        ["dates TBA, “Europe”"],
        ["machine learning", "main track"],
        [{ t: "CCF-A · h5 371", b: true }, "top-tier ML flagship"],
        [{ t: "early May 2027 (estimate)", b: true }, "not in both tracks"]],
      [venue("AAAI-28", "notrec"),
        ["not announced", "AAAI-27: Montréal"],
        ["general AI"],
        [{ t: "CCF-A · h5 232", b: true }, "top-tier general AI"],
        [{ t: "late Jul 2027 (estimate)", b: true }, "AAAI-27: 21 Jul / 28 Jul"]],
      [venue("EACL", "nowin"),
        ["EACL 2027: Athens,", "9–14 Mar 2027", "next: not announced"],
        ["NLP", "ACL family (ARR)"],
        [{ t: "CCF: not listed", b: true }, "h5 77"],
        [{ t: "no window before late 2027", b: true }, "ARR Aug 2026 (closed)"]],
    ], { y: 1.6, colW: COLW1, fontSize: 18 });
    L.text(s, [{ text: "Why: ", options: { bold: true } }, { text: "NeurIPS main expects methods  ·  AAAI-28: general AI, deadline clashes with ASLI, EMNLP" }], MX, 5.2, CW, 0.4, { fontSize: 20 });
    L.text(s, [{ text: "Not ACM MM: ", options: { bold: true } }, { text: "MMM (MultiMedia Modeling) is a smaller venue, CCF-C; MMM 2027 has closed" }], MX, 5.65, CW, 0.4, { fontSize: 20 });
    L.source(s, ["Dates “x / y”: abstract / full paper  ·  CCF: 7th edition (2026)  ·  h5: Google Scholar (2025)  ·  tags: our classification",
      CHECKED_EXTRA].join("\n"), { h: 0.52 });
    L.notes(s, `Not recommended means a poor fit or no workable window (venues.md section 2d, checked 14 November 2026).
    NeurIPS 2027 main track: same location (Europe) and the same estimated early-May deadline as the Evaluations and Datasets track. It expects a methods contribution, while ours is an evaluation, so the E&D track is the better route; one paper cannot be submitted to both tracks (NeurIPS 2026 handbook and E&D FAQ).
    AAAI-28: location and dates are not announced (aaai.org, checked 14 November 2026); AAAI-27 is in Montréal from 16 to 23 February 2027. The deadline is an estimate from AAAI-27: abstract 21 July, full paper 28 July 2026, phase-1 rejections 24 September, final decisions 30 November. AAAI is CCF-A with an h5-index of 232, but it is a general-AI audience, so the fit is medium at best (table 2b), the rate is about 17.5 percent with phase-1 rejection, and there is no workable window for our plan. Its late-July deadline is the week of the ASLI decision (28 July 2027) and falls inside EMNLP's ARR May review, so it is reachable only after an Interspeech or ACM MM rejection.
    EACL: EACL 2027 is in Athens from 9 to 14 March 2027 and used the ARR August 2026 cycle, which is closed. No EACL after 2027 is announced on eacl.org. ACL 2028 will be held in Europe, the Middle East or Africa and is organised together with the EACL board (ACL admin report); in the earlier years with ACL in Europe, 2019 Florence and 2022 Dublin, eacl.org lists ACL in place of an EACL, so a separate EACL 2028 is uncertain; that is our inference, unverified. EACL is not in the CCF list; its h5-index is 77, behind ACL, EMNLP and NAACL. So EACL offers no usable submission window between now and late 2027.
    MMM, the International Conference on MultiMedia Modeling, published by Springer, is a different and smaller venue than ACM Multimedia, focused on multimedia and vision; it is CCF-C in the 2026 list. MMM 2027 is in Siem Reap, Cambodia, from 5 to 8 January 2027; its paper deadline, 30 August 2026, has passed (mmm2027.net). Not recommended.
    Checked 14 November 2026: aaai.org AAAI-27 pages, 2027.eacl.org, eacl.org conference list, the ACL admin report on ACL 2028, mmm2027.net, Google Scholar Metrics, the official CCF 2026 PDF.`);
  }

  // ---- Additional candidates, part 2a: fallbacks, acceptance and format (sections 2b, 2d)
  const COLS2 = ["Venue", "Acceptance rate & difficulty", "Paper format & types", "Link"];
  const COLW2 = [2.3, 3.98, 3.6, 2.25];
  {
    const s = L.slide("Additional candidates", "Fallbacks: ~23–25% at NeurIPS E&D and ACM MM, ~43% at ICASSP", { titleSize: 28 });
    L.table(s, COLS2, [
      [venue("NeurIPS 2027 E&D", "fallback"),
        ["D&B 2024: 25.3%", "E&D 2026: not found", diff("high", ""), "Fit: high, better than main track"],
        ["9 p + unlimited references,", "appendix, mandatory checklist", "+1 p if accepted", "data, code final at deadline"],
        [{ t: "neurips.cc", url: "https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets" }, "2026 E&D call"]],
      [venue("ICASSP 2028", "late"),
        ["~43% (2026: ~4,800 of ~11,200)", diff("low-medium", ""), "Fit: one focused speech finding"],
        ["4 p + optional 5th page for", "references, acknowledgements,", "ethics; or OJSP 8 + 1 p, APC", "(2027 rules)"],
        [{ t: "2028.ieeeicassp.org", url: "https://2028.ieeeicassp.org/" }, "site (no call yet)"]],
      [venue("ACM MM 2027", "cond"),
        ["23.5% (2025: 1,250 of 5,330)", diff("medium-high", ""), "Fit: medium; multimodal preferred"],
        ["6–8 p + up to 2 p references", "no appendix in the main file", "(2026 rules)"],
        [{ t: "2027.acmmm.org", url: "https://2027.acmmm.org/" }, "site (no call yet)"]],
    ], { y: 1.6, colW: COLW2, fontSize: 18 });
    L.source(s, ["“estimate”: projected from the previous cycle  ·  p: pages  ·  D&B: Datasets & Benchmarks, the track's name until 2025",
      CHECKED_EXTRA].join("\n"), { h: 0.52 });
    L.notes(s, `NeurIPS Evaluations and Datasets: the Datasets and Benchmarks track accepted 25.3 percent in 2024, close to the main track's 25.8 percent; in 2025 it received 1,995 submissions, and its rate was not published (NeurIPS blog of the track chairs). The 2026 E&D rate was not found. The call says the track aims for an equally stringent review as the main conference track; ML reviewers want general lessons about evaluation and strong statistics, so difficulty is high, but the fit is high: the track welcomes analyses of the strengths, limitations or failure modes of existing benchmarks and evaluation practices, systematic analyses of systems and benchmarks on new or existing datasets, and a paper need not introduce a new model. Format: 9 pages plus unlimited references and appendix and a mandatory checklist, one more page if accepted. Data and code are not supplementary material: they must be in final form at the full-paper deadline; code is required when the main contribution is an executable artifact and not mandatory for analytical work; a new dataset must be hosted with Croissant metadata and public by the camera-ready deadline. Every author needs an active OpenReview profile before the abstract deadline; review is double-blind by default; arXiv is allowed; one author registers in person.
    ICASSP: ICASSP 2026 received nearly 11,200 regular paper submissions and accepted close to 4,800, about 43 percent (IEEE SPS, June 2026). The ICASSP 2027 rules: 4 pages plus an optional fifth page for references, acknowledgements and ethics, or an IEEE Open Journal of Signal Processing paper of 8 plus 1 pages with an article charge; single-anonymous review; at most 9 papers per author; multiple simultaneous submissions are not permitted; in-person presentation with non-student registration. Four pages suit one focused speech finding, not the whole study.
    ACM MM: MM 2025 accepted 23.45 percent, 1,250 of 5,330; MM 2024 26.2 percent (acmmm.org). The 2026 rules: 6 to 8 pages plus up to 2 pages of references, no appendix in the main file, supplementary material up to 50 MB; every author volunteers as a reviewer; at most 10 papers per author; double-blind; no parallel submission, which also rules out an active ARR submission. Difficulty medium-high, because an audio-only paper is not what MM prefers.
    ${EXTRA}
    Checked 14 November 2026: NeurIPS blog posts and 2026 calls, IEEE SPS news on ICASSP 2026, ICASSP 2027 editorial policies, acmmm.org and the MM 2026 call.`);
  }

  // ---- Additional candidates, part 2b: not recommended, acceptance and format (sections 2b, 2d)
  {
    const s = L.slide("Additional candidates", "Not recommended: ~25% at NeurIPS main, ~17.5% at AAAI", { titleSize: 28 });
    L.table(s, COLS2, [
      [venue("NeurIPS 2027 main", "notrec"),
        ["2025: 24.5% (5,290 of 21,575)", "2026: 25.7% (third-party, unverified)", diff("high", ""), "Fit: low, methods expected"],
        ["9 p + references, appendix,", "checklist; +1 p if accepted", "code encouraged"],
        [{ t: "neurips.cc", url: "https://neurips.cc/Conferences/2026/CallForPapers" }, "2026 main call"]],
      [venue("AAAI-28", "notrec"),
        ["AAAI-26: ~17.5% (~23,000 reviewed,", "30,948 submissions in total)", diff("high", " — phase-1 reject"), "Fit: medium, general AI"],
        ["7 p + 2 p for references only", "appendix, code as supplementary", "(AAAI-27 rules)"],
        [{ t: "aaai.org", url: "https://aaai.org/conference/aaai/aaai-27/" }, "AAAI-27 page"]],
      [venue("EACL", "nowin"),
        ["no window, so not applicable"],
        ["as ACL (ARR): long 8 p", "or short 4 p"],
        [{ t: "2027.eacl.org", url: "https://2027.eacl.org/" }, "EACL 2027 (closed)"]],
    ], { y: 1.6, colW: COLW2, fontSize: 18 });
    L.source(s, ["“unverified”: no official page confirmed it  ·  p: pages",
      CHECKED_EXTRA].join("\n"), { h: 0.52 });
    L.notes(s, `NeurIPS main track: NeurIPS 2025 accepted 24.52 percent, 5,290 of 21,575 valid submissions (blog of the program chairs). For 2026, third-party trackers report 7,900 of 30,709, 25.7 percent; no official page confirmed it, so it is unverified. The 2026 handbook: 9 pages plus references, appendix and a mandatory checklist, one more page if accepted; code is encouraged but not mandatory; no other archival venue while the paper is under review, and not both tracks. Not recommended, because the main track expects a methods contribution.
    AAAI: AAAI-26 had 30,948 submissions in total, about 23,000 reviewed after filtering; about 41 percent passed phase 1 and about 17.5 percent were accepted (AAAI slides on the AAAI-26 review pilot). AAAI-27 numbers are not published yet. The AAAI-27 rules: 7 pages of content plus 2 pages for references only, appendix and code as supplementary material; at most 10 submissions per author; all qualified authors must be available to review; phase 1 rejects without rebuttal; concurrent submissions to other archival venues are not allowed; in-person presentation. Not recommended: low rate with phase-1 rejection, a general-AI audience with medium fit at best, and a deadline that clashes with the ASLI decision and the EMNLP review.
    EACL: no submission window before late 2027, so acceptance rates were not checked; it would follow ACL's ARR format and rules.
    ${EXTRA}
    Checked 14 November 2026: NeurIPS blog and 2026 handbook, csconfstats and OpenAccept for the 2026 NeurIPS figure (third-party), AAAI-26 slides and AAAI-27 calls, 2027.eacl.org.`);
  }

  // ---- Review windows (section 2d)
  {
    const s = L.slide("Target venues and additional candidates", "One venue at a time: after July 2027, ICASSP 2028 is the next step", { titleSize: 28 });
    L.table(s, ["Venue", "Submission", "Decision", "Still reachable after a rejection"], [
      [{ text: "ACL 2027 (ARR Jan)", bold: true }, "4 Jan 2027", "TBA (ACL 2026: 4 Apr)", "ASLI, NeurIPS, EMNLP, AAAI-28, ICASSP 2028 (if early April)"],
      [{ text: "Interspeech 2027", bold: true }, "9 Feb 2027", "2 Jun 2027", "AAAI-28, ICASSP 2028"],
      [{ text: "ACM MM 2027", bold: true }, "~1 Apr (estimate)", "~early Jul (2026: 9 Jul)", "AAAI-28, ICASSP 2028"],
      [{ text: "IEEE ASLI 2027", bold: true }, "21 Apr 2027", "28 Jul 2027", "ICASSP 2028 (AAAI-28 due the same week)"],
      [{ text: "NeurIPS 2027", bold: true }, "early May (estimate)", "late Sep (2026: 24 Sep)", "ICASSP 2028 only if its deadline is later"],
      [{ text: "EMNLP 2027 (ARR May)", bold: true }, "late May (estimate)", "~late Aug (2026: 20 Aug)", "ICASSP 2028"],
      [{ text: "AAAI-28", bold: true }, "late Jul (estimate)", "phase 1 late Sep, final ~end Nov", "ICASSP 2028 only after a phase-1 reject"],
      [{ text: "ICASSP 2028", bold: true }, "mid/late Sep (estimate)", "~mid-Jan 2028 (2027: 13 Jan)", "later cycles, outside this plan"],
    ], { y: 1.55, colW: [2.75, 2.35, 3.15, 3.88], fontSize: 18 });
    L.source(s, ["Every venue here forbids parallel review of the same paper  ·  “estimate”: projected from the previous cycle",
      CHECKED_EXTRA].join("\n"), { h: 0.52 });
    L.notes(s, `This slide is the review-window table of venues.md section 2d, checked on 14 November 2026. ARR, Interspeech, ICASSP, ACM MM, NeurIPS and AAAI all forbid parallel review of the same paper, and the ASLI call asks for papers not currently under review elsewhere. So a rejected paper can only go to a venue whose deadline comes after the decision.
    ACL 2027: the ARR deadline is 4 January 2027; the commitment and notification dates are not announced. In the ACL 2026 cycle the commitment deadline was 14 March and the notification 4 April. If ACL 2027 decides in early April as in 2026, which is an estimate, ASLI (21 April), NeurIPS (early May), EMNLP through the ARR May cycle, AAAI-28 and ICASSP 2028 remain open after a rejection; ACM MM, due about 1 April, does not, if the paper was committed to ACL.
    Interspeech 2027: deadline 9 February, notification 2 June 2027 (official call). After a rejection, the ASLI and EMNLP deadlines have passed; AAAI-28 and ICASSP 2028 remain.
    ACM MM 2027: deadline about 1 April and notification about early July, estimates from 2026 (1 April and 9 July). After a rejection, AAAI-28 and ICASSP 2028 remain.
    IEEE ASLI 2027: deadline 21 April, notification 28 July 2027 (official important-dates page). AAAI-28's full paper is due about the same week (estimate), so in practice only ICASSP 2028 remains.
    NeurIPS 2027: deadline early May, notification late September, estimates from 2026 (6 May and 24 September). ICASSP 2028 remains only if its deadline falls after the decision; both are estimates.
    EMNLP 2027 through ARR May: deadline late May, commitment early August, notification about late August, estimates from 2026 (25 May, 2 August, 20 August). ICASSP 2028 remains.
    AAAI-28: deadline late July; phase-1 rejections late September and final decisions about the end of November, estimates from AAAI-27 (24 September and 30 November). Only a phase-1 rejection could still reach ICASSP 2028, and only if its deadline falls later.
    ICASSP 2028: deadline mid or late September 2027 and notification about mid-January 2028, estimates from ICASSP 2027 (23 September 2026 and 13 January 2027). After that come later cycles, outside this plan.
    The practical consequence: ACM MM and ASLI compete for the April window; NeurIPS and EMNLP compete for May; after the July and August decisions, ICASSP 2028 is the next step.
    Checked 14 November 2026: ACL 2026 and 2027 calls, Interspeech 2027 call, ACM MM 2026 important dates, ASLI 2027 important dates and call, NeurIPS 2026 calls, EMNLP 2026 site, AAAI-27 page, ICASSP 2026 announcement and ICASSP 2027 important dates; the parallel-submission rules from each venue's call or policy page.`);
  }
};
