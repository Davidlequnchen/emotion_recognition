# 数据准备人员：导入接口

中文 | [English](data_import_en.md)

[prepare_tasks.py](prepare_tasks.py) 从多个批次中挑出一轮 1,000 个客户 clip，复制音频，并生成 Label Studio 任务文件；
转写直接读取 manifest 的 `text` 列。挑选规则（分层、配额、补齐）见[操作手册第 3 节](README.md#3-如何挑选待标注片段)。
本文说明输入、设置和输出，以及三种导入方式；不涉及 Label Studio 的安装或启动。

## 1. 生成一轮任务

### 1.1 输入

```text
/absolute/path/to/curated_dataset/              # curated_root
├── batch_01/
│   ├── audio/                                  # 音频，平铺，例如 <UUID>.wav
│   └── manifest_with_predictions.parquet       # 原 manifest + 三个预测列
├── batch_02/
└── …                                           # 约 40 个随机批次
```

每个 `manifest_with_predictions.parquet` 需要以下列，列名可在 YAML 的 `columns` 中修改：

| 默认列名 | 内容 | `columns` 中的键 |
|---|---|---|
| `clip_id` | 稳定 ID，写入 `data.clip_id`；在所有批次中唯一 | `clip_id` |
| `audio_clip_file` | 音频文件名或路径 | `audio` |
| `text` | 已有转写，写入 `data.transcript`；空白或空值写为 `""` | `text` |
| `is_longer_than_10_seconds` | `true` 排除，`false` 保留 | `long_clip` |
| `external_call_id` | 通话 ID，用于"每通电话最多 3 段" | `call_id` |
| `pred_e2v`、`pred_gpt`、`pred_gemini` | emotion2vec、GPT Realtime、Gemini 的预测 | `predictions` |

**预测标签格式**：每格一个标签，不区分大小写，忽略两端空白。

- 有效票：`angry`、`disgusted`、`fearful`、`happy`、`sad`、`surprised`、`neutral`。
- 不算票：空值（null、NaN、空字符串）、`invalid`、`other`、`unknown`、`none`。
- 其他任何值都会报错，并按列列出最多 10 个意外值，便于发现合并错误。

manifest 的其他列原样保留到输出的 `manifest.parquet`。

### 1.2 设置并运行

设置写在 [config/prepare_tasks.yaml](config/prepare_tasks.yaml)，不用命令行参数：

```yaml
curated_root: /absolute/path/to/curated_dataset
batches: [batch_01, batch_02, batch_03]          # 参与挑选的批次文件夹
manifest_name: manifest_with_predictions.parquet
output_root: /absolute/path/to/selected_data_for_human_labeling
columns:
  clip_id: clip_id
  audio: audio_clip_file
  text: text
  long_clip: is_longer_than_10_seconds
  call_id: external_call_id
  predictions: [pred_e2v, pred_gpt, pred_gemini]
quotas:                  # 每轮片段数 = 四项之和 = 1,000
  random: 150
  emotion: 560
  conflict: 140
  neutral: 150
max_clips_per_call: 3
files_per_folder: 100    # 每个 part 文件夹的文件数 = 每次上传的文件数
seed: 42
```

在仓库根目录、用已安装 [requirements.txt](requirements.txt) 的 Python 运行（`emotion_recognition_llm` 环境已包含）：

```bash
python3 label_studio/prepare_tasks.py
```

每次运行生成下一轮（`tasks_001`、`tasks_002` …），并排除前面各轮 `tasks_NNN/tasks_NNN.json` 中已有的片段。
随机数种子是 `seed` 加轮号，所以同样的数据和设置总是得到同一轮结果。

### 1.3 检查规则

脚本先做完全部检查，才复制文件；任何一项不通过都会报错，并且不创建任何文件。

- `output_root` 不能在 `curated_root` 里面，反过来也不行。
- 长片段标记可以是布尔值、数值 `1/0`，或字符串 `true/false`、`1/0`（忽略大小写和两端空白）；缺失或无效会报错。
  筛选只看这个标记，不读取音频测量时长。
- `clip_id` 在所有列出的批次中必须唯一；合格片段的通话 ID 不能为空。
- 音频路径：只有文件名时在 `<批次>/audio/` 中查找；带目录的相对路径相对批次文件夹；绝对路径必须在 `curated_root` 内。
  指向 `curated_root` 外的 symlink 会被拒绝。支持 WAV、MP3、FLAC、OGG、M4A。
- 文件名在所有合格片段中必须唯一（忽略大小写和 Label Studio 会删掉的字符），且不超过 70 个字符：
  服务器版 Label Studio 按文件名配对任务，过长的文件名会被截短。
- 入选的音频必须都存在，否则列出缺失的文件。

运行时依次显示：各批次的行数、长片段、前面各轮已用和合格数；每个模型的票数分布（含不算票）；各分层的片段数；
各层的目标与实际数量；缺口去向；最后是下一步操作。

### 1.4 输出

```text
/absolute/path/to/selected_data_for_human_labeling/   # output_root
├── tasks_001/                        # 第 1 轮
│   ├── audio/part_01/ … part_10/     # 音频副本，每个文件夹 100 个，按任务顺序；原文件不变
│   ├── tasks_001.json                # 任务文件
│   ├── manifest.parquet              # 入选行：原 manifest 全部列 + 来源和位置
│   └── summary.json                  # 本轮记录
└── tasks_002/                        # 下一次运行生成第 2 轮
```

- `manifest.parquet` 在原有列之后加上 `source_batch`、`layer`（random、emotion、conflict、neutral、fill）、
  `target_class`、`stratum`（如 `agree3:angry`）、`task_index`、`part` 和 `copied_audio`。
- `summary.json` 记录设置快照、轮号、seed、各分层片段数、目标与实际数量、缺口去向、空转写数和生成时间。
- 先写入 `tasks_NNN.partial/`，全部完成后才改名为 `tasks_NNN/`；中途出错会删掉 `.partial`。已有的轮次不会被覆盖。

任务 JSON 的形态：

```json
[
  {
    "data": {
      "audio": "/data/local-files/?d=tasks_001/audio/part_01/<clip>.wav",
      "transcript": "已有的 clip 转写",
      "clip_id": "<clip_id>"
    }
  }
]
```

只有这三个字段：预测、分层和票数不进任务 JSON，标注员看不到。脚本对 URL 中的特殊字符编码。
`?d=` 后是相对 `output_root` 的路径，所以本机 Label Studio 的 document root 设为 `output_root`。
不要使用 `file://` 或裸本机绝对路径。公司音频、转写、任务 JSON 和导出均保留在公司授权目录，不写入本仓库。

## 2. 用 Local Files / Tasks 同步导入

本机 Label Studio 的 Local Files **document root** 设为 `output_root`。项目内进入
**Settings → Cloud Storage → Add Source Storage → Local Files**：

1. Source path / Absolute local path 填 `/absolute/path/to/selected_data_for_human_labeling/tasks_001`。
   它必须是 document root 的严格子目录，**不能与 document root 相等**。
2. 点击 **Test Connection**；成功后点 **Next**。
3. **Import Method** 选 **Tasks**，文件过滤填 `^tasks_001\.json$`。
4. 点击 **Load Preview**，确认预览读取到任务而不是把 JSON 当成音频文件。
5. 点击 **Next → Save & Sync**，回到数据页核对任务数，并试播一条。

下一轮另加一个 Source Storage：Source path 为 `…/tasks_002`，过滤条件为 `^tasks_002\.json$`。

本次公开演示采用的就是 Tasks 同步导入。下图中的文件名是 `import_tasks.json`，所以过滤条件实际为
`import_tasks\.json$`；使用上面的轮次目录时，请对应使用 `^tasks_001\.json$`。

![Local Files 向导：Tasks、文件过滤和成功加载的预览](screenshots/04-import-settings.jpg)

仅同步过滤后的任务 JSON，不要再用 **Files** 模式扫描音频目录，否则会绕过挑选结果。
更新 JSON 不会自动删除此前已导入的任务。

## 3. 或通过 Import 上传 JSON

如果负责人已配置好音频的本地文件访问，建议在项目数据页用 **Import → Upload Files** 逐轮上传：

1. 先只上传 `tasks_001/tasks_001.json`，完成这一轮标注。
2. 在同一个项目里上传 `tasks_002/tasks_002.json`，继续下一轮；现有标注结果保留。
3. 按需要继续，做到几千条后可以停止。

上传 JSON 只导入任务字段，不会上传 JSON 中引用的音频。
**每份 JSON 在 Tasks Sync 和 Upload Files 中二选一，且只导入一次**，不要重复导入。

## 4. 服务器版 Label Studio：直接上传音频（不用 Local Files / GCS）

**适用情况**：只能使用公司服务器上的 Label Studio Enterprise（LSE），不能用 Local Files，GCS 也来不及配置。
这时负责人通过 **Import → Upload Files** 直接上传音频。`tasks_NNN.json` **不导入**：它只是"要上传哪些音频"的清单，
也是 `transcript` 和 `clip_id` 的来源。

**为什么要切换模板**：with-transcript 模板有 `$audio`、`$transcript` 两个数据键，直接上传音频会被拒绝：

> Your label config has more than one data key and direct file upload supports only one data key.
> To import data with multiple data keys, use a JSON or CSV file.

所以先用只有 `$audio` 的 audio-only 模板上传，再用脚本补上转写和 `clip_id`，最后切换到 with-transcript。

| 步骤 | 操作 | 检查 |
|---|---|---|
| 1 | 本机运行 `prepare_tasks.py`（[第 1 节](#1-生成一轮任务)），音频已分到 `tasks_NNN/audio/part_01/`、`part_02/` … | 各 part 的文件数 |
| 2 | 在 LSE 建项目，模板用 [config_audio_only.xml](config_audio_only.xml) | 预览只有音频和选项 |
| 3 | 打开 **Import**，每次选中一个 part 文件夹里的全部文件拖进去 | 任务数等于已上传的文件数 |
| 4 | 运行 `attach_task_data.py`，先试运行（`write: false`） | 没有报错；`To update` 等于文件数 |
| 5 | 改为 `write: true`（`limit: 3`），只写 3 条 | Data Manager 出现 `transcript`、`clip_id` 列，这 3 条有值 |
| 6 | 改为 `limit: null`，写入全部 | 提示 `All tasks of this file are done` |
| 7 | **Settings → Labeling Interface → Code** 换成 [config_with_transcript.xml](config_with_transcript.xml) | 打开一条任务：能播放，文字对应 |
| 8 | 这时才让标注员开始 | |

两个脚本都在仓库根目录、用已安装 [requirements.txt](requirements.txt) 的 Python 运行（`emotion_recognition_llm` 环境已包含），
不用命令行参数，设置写在 `label_studio/config/` 的 YAML 中。

- 每个 part 最多 `files_per_folder` 个文件。开源版每次上传默认最多 100 个文件（`DATA_UPLOAD_MAX_NUMBER_FILES`），
  LSE 以管理员设置为准。上传被拒时调小这个值，删掉这一轮的文件夹后重跑：设置不变时会选出同样的片段，只是 part 更小。
- Label Studio 会把空格改成 `_`、删掉 `#` 等符号，并在文件名前加 8 位随机前缀；`attach_task_data.py` 会自动对应。
  `prepare_tasks.py` 已检查文件名唯一且不超过 70 个字符。

### 4.1 补上转写和 clip_id

`label_studio/config/attach_task_data.yaml`：

```yaml
project_id: 1      # 项目网址中的数字：https://<your-label-studio-host>/projects/<id>/data
tasks_file: /absolute/path/to/selected_data_for_human_labeling/tasks_001/tasks_001.json
write: false       # false = 试运行，只读取和匹配；true = 写入
limit: 3           # 每次最多写入几条；检查无误后改为 null = 全部
```

服务器地址和 token 只放在环境变量中，不写进 YAML（仓库是公开的）：

```bash
export LABEL_STUDIO_URL="https://<your-label-studio-host>"
read -rs LABEL_STUDIO_API_KEY && export LABEL_STUDIO_API_KEY   # 粘贴 token 后回车：不显示，也不进 shell 历史
python3 label_studio/attach_task_data.py
```

- token 位置：右上角头像 → **Account & Settings** → **Legacy Token**。也可用 Personal Access Token，脚本会自动换取短期 access token。
- token 等同密码：不要写进文件或发给别人。泄露后立即在同一页面重置。
- 返回 401 时先检查 token；若管理员关闭了 Legacy Token，改用 Personal Access Token。
- 公司网络使用自有证书时，先设置 `export REQUESTS_CA_BUNDLE=/absolute/path/to/company-ca.pem`。
- 脚本只打印数量、文件名和 `clip_id`，不打印转写或 token。
- 找不到对应文件、同一文件上传了两次等问题，都会在写入前停止并列出任务。重复上传时，在 Data Manager
  删掉多余的任务后重跑。已完成的任务会跳过，可以放心重跑。

### 4.2 同一项目的后续轮次

只在两轮之间、没有人标注时进行：

1. 模板换回 `config_audio_only.xml`。两份模板的选项相同，已有标注保留。
2. 再运行一次 `prepare_tasks.py`，生成下一轮 `tasks_002/`，上传它的各 part 文件夹。
3. 把 `attach_task_data.yaml` 的 `tasks_file` 改为 `…/tasks_002/tasks_002.json`，按上表第 4–6 步运行。
   前面轮次的任务显示为 `other batches (ignored)`，不会被修改。
4. 换回 `config_with_transcript.xml`，再让标注员继续。

## 5. 读取并标准化导出

先在 Label Studio **Export → JSON** 下载并保留完整原生文件。以 `data.clip_id` 关联 manifest。
原生 `result` 中 `from_name="decision"`、`type="choices"` 的 `value.choices` 保存唯一决策，
包括 8 个情绪值（含 `other`）、`unknown` 不确定判断，或 `bad_clip` 弃用决策；Bad clip 不是情绪类别。

使用本项目标准化脚本生成业务数据，保留原生文件：

```bash
python3 label_studio/normalize_export.py \
  --input /absolute/path/to/exports/label-studio.json \
  --output /absolute/path/to/exports/emotion-votes.json
```

脚本按每条 annotation 输出一行记录，**保留每位标注员的 vote，不聚合、不覆盖为一个最终标签**。
保留 `clip_id`、`task_id`、`annotation_id`、`annotator_id`、`lead_time`、`decision`、`status` 和 `emotion`。
`completed_by` 是用户对象时只取其 ID，不输出邮箱。

- 8 个情绪决策（包括 `other`）或 `unknown` 不确定判断：`status="accepted"`，`emotion` 为相应原始值。
- Bad clip 决策：`status="bad_clip"`，`emotion` 为 JSON `null`，例如：

```json
[
  {
    "clip_id": "clip_0001",
    "task_id": 1,
    "annotation_id": 1,
    "annotator_id": 1,
    "lead_time": 3.5,
    "decision": "bad_clip",
    "status": "bad_clip",
    "emotion": null
  }
]
```

`accepted` 只表示选择了情绪分支，不代表该 vote 已经多人裁决为 gold；`unknown` 仍单独保留。
`other` 表示明确的其他情绪，`unknown` 表示不确定。旧版界面曾将 `unknown` 显示为 Unknown / Other，
因此历史 `unknown` 不能自动改成 `other`。新批次单独记录配置版本，保留原始导出。
音频可用但情绪不确定时选 Unknown；音频本身无法可靠判断时选 Bad clip，后处理排除这类音频。

取消的 annotation（`was_cancelled`）和无 annotation 的任务会跳过，并报告数量。
Label Studio 默认导出已标注任务，未提交任务不默认包含在内。
缺失 `clip_id` 或 `decision`、多个决策/多个选择、未知值均会报错，验证全部成功后才写输出；已有输出不会被覆盖。

官方依据：[本地存储](https://labelstud.io/guide/storage_local)、[Choices](https://labelstud.io/tags/choices)、
[只读 Text](https://labelstud.io/tags/text)、[导出字段](https://labelstud.io/guide/export.html)。
