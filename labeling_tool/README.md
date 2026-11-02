# labeling_tool

A small tool for labeling the emotion of audio clips by hand: one HTML file that runs in the browser, offline, with
nothing to install. Several people can label the same clips; each person's labels go to their own CSV file.

## For the people who label (no setup needed)

1. Open `labeler.html` in **Chrome or Edge** (double-click it).
2. Click **Choose data folder** and pick the folder you were given: the one that *holds* the `audio/` folder, not
   `audio/` itself. Allow the browser to view and save files in it.
3. Enter your name or nickname and choose your listener number (`listener1` … `listener4`).
4. Each clip plays by itself. Choose the emotion you hear: click it, or press its number key. The next clip opens.
5. A clip that cannot be judged (silence, noise, cut off, several people talking): **Bad clip — skip** (key `0`).

Your labels are saved after every click, into `annotations/annotations_<listener>_<name>.csv` inside the data folder.
Close the browser at any time: next time, **Reopen “…”** brings you back where you stopped. The progress bar shows
how many clips are done and, after a few clips, the time left.

| Key | Action |
|---|---|
| `Space` | play / pause |
| `R` | replay from the start |
| `1` … `8` | choose an emotion |
| `0` | bad clip, skip |
| `←` `→` | previous / next clip |
| `U` | next unlabeled clip |

Firefox and Safari can read the folder but cannot write into it: the page then says so, keeps the labels in the
browser, and **Export CSV** saves the file, which you send on.

### 中文快速说明

1. 用 **Chrome 或 Edge** 双击打开 `labeler.html`。
2. 点 **Choose data folder（选择数据文件夹）**，选择发给你的文件夹：是包含 `audio/` 的那一层，不是 `audio/` 本身。允许浏览器查看和保存文件。
3. 输入你的名字或昵称，选择你的标注员编号（`listener1` … `listener4`）。
4. 每条音频会自动播放。听完点击情绪，或按对应的数字键，然后自动进入下一条。
5. 无法判断的音频（没声音、噪声、被截断、多人同时说话）：点 **Bad clip — skip（坏数据）**，或按 `0`。

每点一次就会自动保存到数据文件夹的 `annotations/annotations_<编号>_<名字>.csv`。随时可以关闭浏览器，下次点
**Reopen（重新打开）** 就能接着标。界面语言在 `config/labels.json` 的 `language` 里切换：`en`、`zh` 或 `both`（中英双语）。

## For the person who prepares the data

### Data folder

```
emotion_labeling/
├── config/
│   └── labels.json               # the labels and options (optional: built-in defaults otherwise)
├── audio/
│   ├── clip_0001.wav
│   ├── clip_0001.txt             # optional: the transcript of clip_0001 (UTF-8 text)
│   ├── clip_0002.mp3             # wav, mp3, flac, ogg, m4a
│   └── ...
├── predictions/                  # optional: what models predicted, one CSV per model
│   ├── emotion2vec_plus_large.csv
│   └── gemini.csv
└── annotations/                  # written by the tool: one CSV per listener
    ├── annotations_listener1_Alice.csv
    └── annotations_listener2_Bob.csv
```

Copy `config/labels.json` from this folder into the data folder and edit it there; each batch of data can have its
own labels. Put the data folder on each person's computer, or on a shared drive (OneDrive, a network folder): every
listener writes only their own file, so several people can label at the same time.
[`demo_data/`](demo_data/) is a complete example with six public clips: open `labeler.html` and choose it to try the
tool.

### `config/labels.json`

| Field | Default | Meaning |
|---|---|---|
| `version` | `emotion8-v1` | name of this label set; written into every CSV row (`labels_version`). Change it when you change the labels |
| `language` | `en` | `en`, `zh` or `both` (English and Chinese side by side) |
| `listeners` | `listener1` … `listener4` | the listener numbers to choose from |
| `randomize` | `true` | each listener gets the clips in their own shuffled order (always the same for that listener number and `version`) |
| `autoplay` | `true` | play each clip when it opens |
| `auto_advance` | `true` | go to the next unlabeled clip after a label is chosen |
| `audio_folder` | `audio` | folder of the clips inside the data folder (if it does not exist, clips are taken from the data folder itself) |
| `annotations_folder` | `annotations` | where the CSV files are written |
| `predictions_folder` | `predictions` | where the model predictions are (optional; see below) |
| `predictions_mode` | `show` | `show`: predictions on screen with each clip; `click`: hidden until "Show model predictions" is clicked; `off`: never shown |
| `models` | `[]` | which prediction files to show and their display names, in this order: `[{"id": ..., "name": ..., "file": ...}]`. Empty: every CSV in `predictions_folder`, named after the file |
| `labels` | the 7 emotions of emotion2vec+ and `unknown` | one entry per button: `id` (written to the CSV), `en`, `zh`, `emoji`, `color`, `key` |
| `bad_clip` | `bad_clip`, key `0` | the "bad clip, skip" button; remove the field to hide it |

Default labels: `angry`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, `surprised`, `unknown` (Unknown / Other).

### Model predictions (optional)

When models have already labeled the clips (e.g. emotion2vec+, Gemini), the tool shows their answers under the
transcript as guidance, and marks the emotion buttons they chose with 🤖 and the number of models. Without a
`predictions/` folder nothing changes. One CSV per model:

```csv
file,label,score
clip_0001.wav,angry,0.93
clip_0002.mp3,neutral,0.61
clip_0003.wav,happy,
```

- `file`: the clip file name (the name without extension also works).
- `label`: matched to the label set by `id`, English or Chinese name, case-insensitive (emotion2vec+'s `生气/angry`
  also works). A label that is not in the set is shown as plain text.
- `score` (optional): the model's confidence, 0-1, shown as a percentage.
- A clip that is missing from a file shows "no prediction for this clip" for that model.

Name the models in `config/labels.json`, for example:

```json
"models": [
  {"id": "emotion2vec_plus_large", "name": "emotion2vec+ large", "file": "emotion2vec_plus_large.csv"},
  {"id": "gemini", "name": "Gemini 2.5 Pro", "file": "gemini.csv"}
]
```

Predictions can pull the listeners towards the model's answer. When the labels must be independent of the models,
use `"predictions_mode": "click"` or `"off"`.

### Output: `annotations/annotations_<listener>_<name>.csv`

One row per labeled clip (UTF-8 with a byte-order mark, so Excel shows Chinese correctly):

| Column | Content |
|---|---|
| `file` | clip file name |
| `transcript` | its transcript, if there is one |
| `label` | the chosen label `id`, or `bad_clip` |
| `listener_id`, `listener_name` | who labeled it |
| `labeled_at` | time of the (last) choice, ISO 8601, UTC |
| `play_count` | how often the clip was played |
| `seconds_spent` | time spent on the clip |
| `labels_version` | `version` of the label set |

Collect the `annotations/*.csv` files of all listeners for the next step (e.g. majority vote).

### Sharing the tool

- Send `labeler.html` (it is self-contained), or keep it next to the data folder on the shared drive.
- It can also be served from GitHub Pages: it has no backend, and the audio is still read from each person's own
  computer. Only `labeler.html` (and the README) would be published, never data.
- The page makes no network requests at all (a Content-Security-Policy forbids them), so audio, transcripts and
  labels never leave the computer.

## Files

| File | Content |
|---|---|
| `labeler.html` | the tool (HTML, CSS and JavaScript in one file) |
| `config/labels.json` | the default label set, to copy into a data folder |
| `demo_data/` | an example data folder: six clips of the public CREMA-D corpus, four with a transcript, and the predictions of two models |
