# Custom speech emotion classification template

[中文](template_configuration.md) | English · [User guide](README_en.md)

The **Audio/Speech Processing** template list in the local Label Studio **1.23.2** instance does not show Audio Classification.
Use **Custom template → Code** in the lower-left corner, paste the complete configuration, and click **Save**.
The official [Audio Classification](https://labelstud.io/templates/audio_classification.html) template is an equivalent
reference; it is not a visible button in this local template list. Do not select Speech Transcription or Speaker
Segmentation: this project labels the entire clip.

All fixed interface text is in English. The interface has two sections:

- **Emotion**, at the top: 8 options, comprising 7 emotions and Unknown / Other.
- **Unusable audio**, below: a separate Bad clip rejection option that assigns no emotion.

These two paths are mutually exclusive. **Bad clip is not a ninth emotion.** Both sections share a single `decision`
radio group internally, ensuring that each clip receives only one decision.

## 1. Create the project and paste the configuration

1. Create a project, such as `SER Emotion Labeler`, and click **Custom template** in the lower-left corner of **Labeling Setup**.
2. Open **Code** and replace its contents with the complete XML below. Do not append it to an existing template.
3. Check the player, the 8 emotion cards in the upper section, the separate Bad clip option below, and the ⓘ help in both sections. Then click **Save**.
4. For an existing project, open **Settings → Labeling Interface → Code**.

If an existing project already contains annotations, Label Studio may prevent removal of fields used by those results.
Create a clean project for a new annotation protocol and preserve the old project's history.
You can also copy the complete code directly from [config_audio_only.xml](config_audio_only.xml):

```xml
<View>
  <Style>
    .ser-shell { max-width: 1040px; margin: 8px auto; padding: 20px; border: 1px solid #e0e1e6; border-radius: 18px; background: #fff; box-shadow: 0 4px 20px rgba(28,32,36,.05); color: #1c2024; }
    .ser-title h3 { margin: 0 0 5px; font-size: 22px !important; font-weight: 700; }
    .ser-subtitle h3 { margin: 0 0 16px; font-size: 14px !important; line-height: 1.5; font-weight: 400; color: #60646c; }
    .ser-player { padding: 12px; border-radius: 12px; background: #f6f7f9; border: 1px solid #e9ebf0; margin-bottom: 18px; }
    .ser-emotions { padding: 18px; border: 1px solid #dfe5f1; border-radius: 14px; background: #fff; }
    .ser-section-head { display: flex; align-items: center; gap: 8px; margin: 0 0 5px; }
    .ser-section-title { font-size: 16px; font-weight: 650; color: #1c2024; }
    .ser-section-note { margin: 0 0 14px; font-size: 13px; color: #60646c; line-height: 1.5; }
    .ser-info { position: relative; display: inline-flex; align-items: center; justify-content: center; color: #7b8497; font-size: 16px; cursor: help; outline-offset: 3px; }
    .ser-tooltip { visibility: hidden; opacity: 0; position: absolute; z-index: 50; left: 0; bottom: calc(100% + 8px); width: 280px; padding: 10px 12px; background: #202735; color: #fff; border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,.16); font-size: 13px; font-weight: 400; line-height: 1.5; pointer-events: none; text-align: left; }
    .ser-info:hover .ser-tooltip, .ser-info:focus .ser-tooltip { visibility: visible; opacity: 1; }
    .ser-emotion-grid { display: grid !important; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 12px; }
    .ser-choices .lsf-choice { display: block !important; min-width: 0; }
    .ser-choices .lsf-choice__item { margin: 0 !important; min-width: 0; }
    .ser-choices .ant-radio-wrapper { position: relative; display: flex !important; align-items: center; justify-content: center; width: 100%; height: 100px; margin: 0 !important; padding: 10px; border: 2px solid #e0e1e6; border-radius: 12px; background: #fff; cursor: pointer; transition: border-color .15s,background .15s,transform .1s; }
    .ser-choices .ant-radio-wrapper:hover { border-color: var(--ser-accent,#3e63dd); background: var(--ser-tint,#edf2fe); transform: translateY(-1px); }
    .ser-choices .ant-radio-wrapper-checked { border-color: var(--ser-accent,#3e63dd); background: var(--ser-tint,#edf2fe); box-shadow: 0 0 0 1px var(--ser-accent,#3e63dd); }
    .ser-choices .ant-radio-wrapper:focus-within { outline: 3px solid #b5c7ff; outline-offset: 3px; }
    .ser-choices .ant-radio { position: absolute; opacity: 0; }
    .ser-choices .ant-radio-wrapper > span:last-child { width: 100%; padding: 0; }
    .ser-card { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 7px; text-align: center; }
    .ser-emoji { font-size: 34px; line-height: 1.1; }
    .ser-label { font-size: 19px; font-weight: 600; line-height: 1.2; color: #1c2024; }
    .ser-choices .lsf-hint { position: absolute; top: 7px; right: 9px; font-size: 11px; color: #8b8d98; }
    .ser-reject { display: grid; grid-template-columns: 1fr 230px; align-items: center; gap: 18px; margin-top: 20px; padding: 16px 18px; border: 1px solid #e8dfe2; border-radius: 14px; background: #faf7f8; }
    .ser-reject .ser-section-note { margin-bottom: 0; }
    .ser-reject .ant-radio-wrapper { height: 58px; border: 1px dashed #dc9da3; background: #fff; }
    .ser-reject .ser-card { flex-direction: row; gap: 9px; }
    .ser-reject .ser-emoji { font-size: 24px; }
    .ser-reject .ser-label { font-size: 17px; color: #bc3543; }
    .ser-transcript { margin: 0 0 18px; padding: 12px 14px; border-left: 3px solid #3e63dd; border-radius: 0 10px 10px 0; background: #f6f7f9; font-size: 16px; }
    .ser-transcript h3 { margin: 0 0 8px; font-size: 13px !important; font-weight: 600; color: #60646c; }
    @media (max-width: 1000px) { .ser-shell { padding: 16px; } .ser-emotion-grid { grid-template-columns: repeat(2,minmax(0,1fr)); } .ser-reject { grid-template-columns: 1fr; gap: 12px; } }
  </Style>
  <View className="ser-shell">
    <View className="ser-title"><Header value="Emotion Labeler"/></View>
    <View className="ser-subtitle"><Header value="Listen to the clip, then make one selection."/></View>
    <View className="ser-player"><Audio name="audio" value="$audio"/></View>
    <View className="ser-choices">
      <Choices name="decision" toName="audio" choice="single-radio" required="true" showInline="false">
        <View className="ser-emotions">
          <HyperText name="emotion_help" inline="true" selectionEnabled="false" value='&lt;div class="ser-section-head"&gt;&lt;span class="ser-section-title"&gt;Emotion&lt;/span&gt;&lt;span class="ser-info" tabindex="0" title="Use this section when the audio can be judged. Choose one emotion or Unknown / Other. Unknown means the audio is usable but its emotion is uncertain." aria-label="Emotion help"&gt;ⓘ&lt;span class="ser-tooltip"&gt;Use this section when the audio can be judged. Choose one emotion or Unknown / Other. Unknown means the audio is usable but its emotion is uncertain.&lt;/span&gt;&lt;/span&gt;&lt;/div&gt;&lt;p class="ser-section-note"&gt;Choose exactly one of the 8 options.&lt;/p&gt;'/>
          <View className="ser-emotion-grid">
            <Choice value="angry" hotkey="1" hint="Choose this emotion. Select only one of the 8 emotion options." style="--ser-accent: #e5484d; --ser-tint: #fff1f2;" html='&lt;div class="ser-card"&gt;&lt;span class="ser-emoji"&gt;😠&lt;/span&gt;&lt;span class="ser-label"&gt;Angry&lt;/span&gt;&lt;/div&gt;'/>
            <Choice value="disgusted" hotkey="2" hint="Choose this emotion. Select only one of the 8 emotion options." style="--ser-accent: #46a758; --ser-tint: #effaf1;" html='&lt;div class="ser-card"&gt;&lt;span class="ser-emoji"&gt;🤢&lt;/span&gt;&lt;span class="ser-label"&gt;Disgusted&lt;/span&gt;&lt;/div&gt;'/>
            <Choice value="fearful" hotkey="3" hint="Choose this emotion. Select only one of the 8 emotion options." style="--ser-accent: #8e4ec6; --ser-tint: #f7f0ff;" html='&lt;div class="ser-card"&gt;&lt;span class="ser-emoji"&gt;😨&lt;/span&gt;&lt;span class="ser-label"&gt;Fearful&lt;/span&gt;&lt;/div&gt;'/>
            <Choice value="happy" hotkey="4" hint="Choose this emotion. Select only one of the 8 emotion options." style="--ser-accent: #f5a524; --ser-tint: #fff8e8;" html='&lt;div class="ser-card"&gt;&lt;span class="ser-emoji"&gt;😄&lt;/span&gt;&lt;span class="ser-label"&gt;Happy&lt;/span&gt;&lt;/div&gt;'/>
            <Choice value="neutral" hotkey="5" hint="Choose this emotion. Select only one of the 8 emotion options." style="--ser-accent: #8b8d98; --ser-tint: #f4f5f7;" html='&lt;div class="ser-card"&gt;&lt;span class="ser-emoji"&gt;😐&lt;/span&gt;&lt;span class="ser-label"&gt;Neutral&lt;/span&gt;&lt;/div&gt;'/>
            <Choice value="sad" hotkey="6" hint="Choose this emotion. Select only one of the 8 emotion options." style="--ser-accent: #3e63dd; --ser-tint: #eef2ff;" html='&lt;div class="ser-card"&gt;&lt;span class="ser-emoji"&gt;😢&lt;/span&gt;&lt;span class="ser-label"&gt;Sad&lt;/span&gt;&lt;/div&gt;'/>
            <Choice value="surprised" hotkey="7" hint="Choose this emotion. Select only one of the 8 emotion options." style="--ser-accent: #12a594; --ser-tint: #eafbf7;" html='&lt;div class="ser-card"&gt;&lt;span class="ser-emoji"&gt;😲&lt;/span&gt;&lt;span class="ser-label"&gt;Surprised&lt;/span&gt;&lt;/div&gt;'/>
            <Choice value="unknown" hotkey="8" hint="The audio is usable, but the emotion is uncertain or outside the listed categories." style="--ser-accent: #6f6e77; --ser-tint: #f4f3f7;" html='&lt;div class="ser-card"&gt;&lt;span class="ser-emoji"&gt;❓&lt;/span&gt;&lt;span class="ser-label"&gt;Unknown / Other&lt;/span&gt;&lt;/div&gt;'/>
          </View>
        </View>
        <View className="ser-reject">
          <HyperText name="reject_help" inline="true" selectionEnabled="false" value='&lt;div class="ser-section-head"&gt;&lt;span class="ser-section-title"&gt;Unusable audio&lt;/span&gt;&lt;span class="ser-info" tabindex="0" title="Use Bad clip only when the audio itself cannot be reliably judged, for example severe noise, silence, or overlapping speakers. It clears any emotion choice." aria-label="Unusable audio help"&gt;ⓘ&lt;span class="ser-tooltip"&gt;Use Bad clip only when the audio itself cannot be reliably judged, for example severe noise, silence, or overlapping speakers. It clears any emotion choice.&lt;/span&gt;&lt;/span&gt;&lt;/div&gt;&lt;p class="ser-section-note"&gt;No emotion will be assigned.&lt;/p&gt;'/>
          <Choice value="bad_clip" hotkey="0" hint="Reject unusable audio. This clears the emotion choice; no emotion will be assigned." style="--ser-accent: #e5484d; --ser-tint: #fff1f2;" html='&lt;div class="ser-card"&gt;&lt;span class="ser-emoji"&gt;🚫&lt;/span&gt;&lt;span class="ser-label"&gt;Bad clip — skip&lt;/span&gt;&lt;/div&gt;'/>
        </View>
      </Choices>
    </View>
  </View>
</View>
```

## 2. What each field controls

| Configuration | Purpose |
|---|---|
| Outermost `<View>` | Wraps the entire labeling interface |
| `<Style>` | CSS that controls heading sizes, card dimensions, the grid, sections, colors, and tooltips; it adds no annotation fields |
| `<View className="ser-shell">` and similar elements | Apply CSS classes to layout sections |
| `<Header value="…"/>` | Fixed English instructions, such as Emotion Labeler; not a result field |
| `<Audio name="audio" value="$audio"/>` | `Audio` is the player, `name="audio"` is its object name, and `$audio` reads the task's `data.audio` URL |
| `<Choices name="decision" toName="audio" …>` | The native annotation field is `decision`, associated with the player named `audio` |
| `choice="single-radio"` | All decisions share one radio group, so an emotion and Bad clip are mutually exclusive |
| `required="true"` | Prevents submission without a selection |
| `showInline="false"` | Disables the default inline arrangement; nested Views and CSS determine the layout |
| `<View className="ser-emotions">` and `ser-emotion-grid` | Display the upper Emotion section and its 8 options; these are visual groupings only |
| `<View className="ser-reject">` | Displays the separate rejection section below; it remains inside the same Choices group and does not create a second annotation field |
| `<Choice value="angry" …/>` and other choices | `value` is the native decision value; emotions use stable IDs, while `bad_clip` represents rejection |
| `html="…"` | Sets the card's emoji and English name; display content is separate from the exported `value` |
| `hint="…"` | English help shown when hovering over a choice |
| `hotkey="1"` and other hotkeys | Selects an option by keyboard; **Submit** is still required |
| `style="--ser-accent: …; --ser-tint: …;"` | Sets an option's accent and pale background colors; affects appearance only |
| `<HyperText name="emotion_help" …>` and `reject_help` | Display English section titles, instructions, and ⓘ help; require no input and add no human decision fields |
| No `perRegion="true"` | Labels the entire clip without requiring a time-range selection |

Nested Views do not split the radio group into two groups. Switching Angry → Bad clip → Unknown retains only the
current decision. Hover over either section's **ⓘ** for English explanations, or focus it using the Tab key.
The cards also provide hover help. There is no additional questionnaire or text input.

The `&lt;` and `&gt;` sequences inside `html` attributes are XML escapes. Leave them unchanged when copying.
Adjust CSS or `html` to change appearance; do not casually change `name`, `toName`, or `Choice value` after formal
annotation begins. Confirm the final emotion label set after the pilot, then freeze the configuration version.

## 3. Distinguish emotion labels from rejection

The upper Emotion section contains:

| English card | Native decision value | Shortcut | Meaning |
|---|---|---|---|
| 😠 Angry | `angry` | `1` | Anger |
| 🤢 Disgusted | `disgusted` | `2` | Disgust |
| 😨 Fearful | `fearful` | `3` | Fear |
| 😄 Happy | `happy` | `4` | Happiness |
| 😐 Neutral | `neutral` | `5` | Sufficient evidence supports a neutral judgment |
| 😢 Sad | `sad` | `6` | Sadness |
| 😲 Surprised | `surprised` | `7` | Surprise |
| ❓ Unknown / Other | `unknown` | `8` | Usable audio, but the emotion is uncertain or outside the current categories |

The lower Unusable audio section contains only **🚫 Bad clip — skip**, with shortcut `0`.
Choose it when severe noise, silence, overlapping speakers, or similar issues make the audio itself unreliable to
judge. It clears any selected emotion. Selecting any emotion instead cancels Bad clip. Click **Submit** after selecting.

Unknown does not mean Neutral and does not mean the audio is unusable. Bad clip is a separate rejection decision;
normalized results assign it no emotion. The word skip in the button means rejecting this clip. **Submit** still
saves that decision; the task is not silently discarded.

## 4. Display an existing transcript if needed

Replace the editor contents with the complete [config_with_transcript.xml](config_with_transcript.xml) configuration.
Do not concatenate the two XML files. It retains the same `decision` radio group and adds only an English Transcript
section after the player:

```xml
<View className="ser-transcript">
  <Header value="Transcript"/>
  <Text name="transcript" value="$transcript" selectionEnabled="false"/>
</View>
```

`$transcript` reads the same task's `data.transcript`. Text displays the existing content as read-only; it does not
require transcription or another ASR run. Use the same audio-only or with-transcript protocol for all annotators in
a batch. When transcript assistance is allowed, listen first and then consult the text.

## 5. Match the configuration to task JSON

`tasks.json` is an array of tasks. This example demonstrates field mapping and contains no actual company call content:

```json
[
  {
    "data": {
      "audio": "/data/local-files/?d=audio/clip_0001.wav",
      "transcript": "Existing clip transcript.",
      "clip_id": "clip_0001"
    }
  }
]
```

| Task field | Purpose |
|---|---|
| `data.audio` | Corresponds to `<Audio value="$audio"/>`; the player loads this URL |
| `data.transcript` | Corresponds to `<Text value="$transcript"/>`; displayed only by the with-transcript protocol |
| `data.clip_id` | Stable clip ID retained in exports and used to join results to the data preparation manifest |

The local public sample set contains six clips, `clip_0001` through `clip_0006`, with original files in
`labeling_tool/demo_data/audio/`. Two clips have no `.txt` file, so their task `transcript` is an empty string.
An empty transcript display is expected for those tasks. See the [data import guide](data_import_en.md) for preparing
batch directories and audio URLs.

## 6. Native exports and business fields

The native JSON downloaded from Label Studio stores a decision result as follows:

```json
{
  "from_name": "decision",
  "to_name": "audio",
  "type": "choices",
  "value": {"choices": ["bad_clip"]}
}
```

Here, `bad_clip` is a rejection decision, not an emotion label. After running [normalize_export.py](normalize_export.py):

| Native `decision` | Business `status` | Business `emotion` |
|---|---|---|
| One of the 8 emotion values, including `unknown` | `accepted` | The corresponding emotion value |
| `bad_clip` | `bad_clip` | JSON `null` |

Keep native and normalized JSON separately. Do not treat the native file as already-converted business data.
Commands and output examples are in the [export normalization guide](data_import_en.md#4-read-and-normalize-exports).

## 7. Check before saving

- All fixed text is English. The upper Emotion section has 8 emoji cards, and the separate lower Unusable audio section provides rejection.
- Both ⓘ icons show explanations on hover or keyboard focus. The Bad clip card also has English help.
- Switching between an emotion and Bad clip leaves only one current selection; no other fields need to be filled in.
- Audio-only does not display a transcript; with-transcript adds only a read-only Transcript section.
- Play a clip, select an option, click **Submit**, and export native JSON. Check `from_name="decision"` and a single decision in `value.choices`.
- A normalized Bad clip result has `emotion: null`; it does not become a ninth emotion.

Official field references: [Choices](https://labelstud.io/tags/choices), [Choice](https://labelstud.io/tags/choice),
[Text](https://labelstud.io/tags/text), and [View](https://labelstud.io/tags/view).
