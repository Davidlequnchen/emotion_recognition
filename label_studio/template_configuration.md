# 自定义语音情绪分类模板

中文 | [English](template_configuration_en.md)

本机 Label Studio **1.23.2** 的 **Audio/Speech Processing** 列表没有显示 Audio Classification。
实际路径是点击左下角 **Custom template（创建自定义模板）→ Code**，粘贴完整配置，再 **Save**。
官方 [Audio Classification](https://labelstud.io/templates/audio_classification.html) 是语义等价的参考，
不是本机列表中可见的按钮。不要选择 Speech Transcription 或 Speaker Segmentation；当前标注整个 clip。

固定界面文字全部为英文。界面分为两区：

- 上方 **Emotion**：8 个选项，包含 7 种情绪和 Unknown / Other。
- 下方 **Unusable audio**：独立的 Bad clip 弃用入口，表示不赋予情绪。

这两条路径互斥，**Bad clip 不是第 9 种情绪**。内部共用一个 `decision` 单选组，确保每条 clip 只有一个决策。

## 1. 创建并粘贴配置

1. 建立项目，例如 `SER Emotion Labeler`，在 **Labeling Setup** 点击左下角 **Custom template**。
2. 打开 **Code**，复制下方完整 XML，替换原有内容，不要追加在旧模板后面。
3. 检查播放器、上区 8 个情绪卡片、下区独立 Bad clip 和两区的 ⓘ 帮助，再点击 **Save**。
4. 已建项目的入口是 **Settings → Labeling Interface → Code**。

如果旧项目已有标注，Label Studio 可能不允许删除正在被结果使用的字段。新协议应另建干净项目，保留旧项目历史。
下面的完整代码也可直接从 [config_audio_only.xml](config_audio_only.xml) 复制：

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

## 2. 字段分别控制什么

| 配置 | 作用 |
|---|---|
| 最外层 `<View>` | 包裹整个标注界面 |
| `<Style>` | CSS：控制标题字号、卡片尺寸、网格、分区、颜色及提示框；不新增标注字段 |
| `<View className="ser-shell">` 等 | 将 CSS 类绑定到布局区域 |
| `<Header value="…"/>` | 固定英文提示，例如 Emotion Labeler；不是结果字段 |
| `<Audio name="audio" value="$audio"/>` | `Audio` 是播放器，`name="audio"` 是对象名，`$audio` 读取任务的 `data.audio` 地址 |
| `<Choices name="decision" toName="audio" …>` | 原生标注字段是 `decision`，关联上方名为 `audio` 的播放器 |
| `choice="single-radio"` | 所有决策共用一个单选组，情绪与 Bad clip 原生互斥 |
| `required="true"` | 没有选择时不能提交 |
| `showInline="false"` | 不使用默认的同行排列；具体布局由嵌套 View 与 CSS 决定 |
| `<View className="ser-emotions">`、`ser-emotion-grid` | 显示上方 Emotion 区和 8 个情绪选项；只有视觉分组作用 |
| `<View className="ser-reject">` | 显示下方独立弃用区；仍位于同一个 Choices 内，不构成第二个标注字段 |
| `<Choice value="angry" …/>` 等 | `value` 是原生决策值；情绪采用稳定 ID，`bad_clip` 代表弃用路由 |
| `html="…"` | 决定卡片的 emoji 和英文名称；显示内容与导出的 `value` 分开 |
| `hint="…"` | 选项 hover 时的英文帮助 |
| `hotkey="1"` 等 | 快捷选择；选中后仍需 Submit |
| `style="--ser-accent: …; --ser-tint: …;"` | 为选项设置强调色和浅背景色，仅影响外观 |
| `<HyperText name="emotion_help" …>`、`reject_help` | 显示两区的英文标题、说明和 ⓘ 帮助，不要求输入，也不新增人工决策字段 |
| 没有 `perRegion="true"` | 标记整个 clip，不要求拖选时间区间 |

嵌套 View 不会把单选组拆成两个。切换 Angry → Bad clip → Unknown 时，分别只保留当前一个决策。
鼠标移到两区的 **ⓘ** 可看英文解释，也可用键盘 Tab 聚焦 ⓘ；卡片本身也有 hover 帮助。
没有额外问卷或文字输入。

`html` 属性中的 `&lt;`、`&gt;` 是 XML 转义，复制时保持原样。
修改外观可调整 CSS 或 `html`；正式标注后不要随意修改 `name`、`toName` 或 `Choice value`。
最终情绪标签集仍需在 pilot 后确认，再冻结配置版本。

## 3. 情绪与弃用怎么区分

上方 Emotion 区：

| 英文卡片 | 原生决策值 | 快捷键 | 含义 |
|---|---|---|---|
| 😠 Angry | `angry` | `1` | 生气 |
| 🤢 Disgusted | `disgusted` | `2` | 厌恶 |
| 😨 Fearful | `fearful` | `3` | 恐惧 |
| 😄 Happy | `happy` | `4` | 开心 |
| 😐 Neutral | `neutral` | `5` | 有足够线索支持中性判断 |
| 😢 Sad | `sad` | `6` | 难过 |
| 😲 Surprised | `surprised` | `7` | 吃惊 |
| ❓ Unknown / Other | `unknown` | `8` | 音频可用，但情绪不确定或不属于当前类别 |

下方 Unusable audio 区只有 **🚫 Bad clip — skip**，快捷键为 `0`。
当严重噪声、静音或多人重叠等使音频本身无法可靠判断时选它；这会取消已有情绪。
反过来选择任一情绪，会取消 Bad clip。选择后点击 **Submit**。

Unknown 不等于 Neutral，也不代表音频不可用。Bad clip 是独立弃用决策，标准化结果中不赋予情绪。
按钮名称中的 skip 指弃用这条音频；仍通过 Submit 保存这个决策，而不是静默丢弃任务。

## 4. 如需显示已有转写

使用 [config_with_transcript.xml](config_with_transcript.xml) 的完整配置替换编辑器内容，不要拼接两份 XML。
它保留相同的 `decision` 单选组，只在播放器后新增英文 Transcript 区域：

```xml
<View className="ser-transcript">
  <Header value="Transcript"/>
  <Text name="transcript" value="$transcript" selectionEnabled="false"/>
</View>
```

`$transcript` 读取同一任务的 `data.transcript`。Text 只读显示已有内容，不要求重新转写或重新 ASR。
同一批标注员使用一致的 audio-only 或 with-transcript 协议；允许文字辅助时先听声音、再参考文字。

## 5. 配置如何对应任务 JSON

`tasks.json` 是任务数组。以下仅演示字段配对，不是实际公司的通话内容：

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

| 任务字段 | 作用 |
|---|---|
| `data.audio` | 对应 `<Audio value="$audio"/>`，播放器加载这个地址 |
| `data.transcript` | 对应 `<Text value="$transcript"/>`，仅 with-transcript 协议显示 |
| `data.clip_id` | 稳定的片段 ID，保留在导出中，用于关联数据准备 manifest |

本机公开样本有六个 clip，原始文件位于 `labeling_tool/demo_data/audio/`，编号 `clip_0001` 至 `clip_0006`。
其中两个没有 `.txt`，任务的 `transcript` 为空字符串，带转写界面中显示为空是预期行为。
批次目录和音频地址准备见 [数据导入接口](data_import.md)。

## 6. 原生导出与业务字段

Label Studio 下载的原生 JSON 中，决策结果是：

```json
{
  "from_name": "decision",
  "to_name": "audio",
  "type": "choices",
  "value": {"choices": ["bad_clip"]}
}
```

这里的 `bad_clip` 是弃用决策，不是 emotion 标签。运行 [normalize_export.py](normalize_export.py) 后：

| 原生 `decision` | 业务 `status` | 业务 `emotion` |
|---|---|---|
| 8 个情绪值之一，包括 `unknown` | `accepted` | 对应情绪值 |
| `bad_clip` | `bad_clip` | JSON `null` |

原生 JSON 和标准化 JSON 分别保存，不能把原生文件误当成已经转换的业务数据。
命令与输出示例见 [标准化导出说明](data_import.md#4-读取并标准化导出)。

## 7. 保存前自查

- 固定文字全部英文；上方 Emotion 为 8 个 emoji 情绪卡片，下方独立 Unusable audio 弃用区。
- 两区的 ⓘ 可 hover 或键盘聚焦查看说明；Bad clip 卡片也有英文帮助。
- 情绪和 Bad clip 切换后只保留一个当前选择，不需要填写其他字段。
- audio-only 不显示转写；with-transcript 只增加只读 Transcript。
- 试播、选择、Submit 后导出原生 JSON，确认 `from_name="decision"` 和 `value.choices` 中唯一的决策值。
- 标准化 Bad clip 结果为 `emotion: null`，不会成为第 9 种情绪。

官方字段依据：[Choices](https://labelstud.io/tags/choices)、[Choice](https://labelstud.io/tags/choice)、
[Text](https://labelstud.io/tags/text)、[View](https://labelstud.io/tags/view)。
