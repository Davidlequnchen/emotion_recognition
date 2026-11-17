# 数据准备人员：导入接口

中文 | [English](data_import_en.md)

从批次的 `manifest.parquet` 为每个符合条件的客户 clip 建一条任务；转写直接读取 manifest 的 `text` 列，
不需要单独的转写文件夹。合格片段用固定 seed 打乱后，拆成互不重叠的任务 JSON，每份最多 1,000 条。
本机服务的 document root 已由负责人配置；下方使用通用路径说明，不涉及安装或启动步骤。

## 1. 准备一个批次目录

将批次放在 document root 的**严格子目录**中，例如：

```text
/absolute/path/to/dataset/        # 服务已配置的 document root
└── batch/
    ├── audio/                   # clip_0001.wav、clip_0002.mp3 …
    ├── manifest.parquet         # clip ID、音频文件、文字稿、长音频标记等
    └── tasks/                   # 生成的任务文件目录；Local Files 的 Source path
        ├── tasks_001.json       # 最多 1,000 条
        ├── tasks_002.json       # 另外最多 1,000 条
        └── …
```

Local Files 的 Source path 必须是 document root 内的严格子目录，**不能与 document root 相等**。
上面的 `batch/tasks/` 满足这一要求。

在已安装 `pyarrow` 的 Python 环境中运行（项目的 `emotion_recognition_llm` 环境已包含它）。
独立使用时，先在 Python 3.11 环境中从仓库根目录安装 `python3 -m pip install -r label_studio/requirements.txt`。
然后在仓库根目录生成任务：

```bash
python3 label_studio/prepare_tasks.py \
  --manifest /absolute/path/to/dataset/batch/manifest.parquet \
  --document-root /absolute/path/to/dataset \
  --output-dir /absolute/path/to/dataset/batch/tasks \
  --batch-size 1000 \
  --seed 42
```

`--output-dir` 替代原来的单文件 `--output`。`--batch-size` 默认 `1000`，`--seed` 默认 `42`，两者均可省略。
脚本先完成筛选和验证，按 `clip_id` 排序，再用该 seed 打乱一次并连续分块。每个合格片段恰好出现一次，
各文件之间没有重叠，最后一份可以不足 1,000 条；例如 2,007 条会生成 1,000、1,000、7 条三个文件。
相同合格片段集合、seed 和 batch size 会得到相同分组，即使 manifest 行顺序改变也不影响分组。
入选片段集合或 seed 改变后可能重新分组；开始标注后请继续使用首次生成的文件。

默认读取以下列；其他 metadata 保留在 manifest 中，通过 `clip_id` 与导出结果关联：

| manifest 列 | 用途 | 修改列名的参数 |
|---|---|---|
| `clip_id` | 原有稳定 ID，写入 `data.clip_id`；整数 ID 转成字符串，字符串 ID 保留原值 | `--clip-id-column` |
| `audio_clip_file` | 音频文件名或文件路径，转换为 `data.audio` | `--audio-column` |
| `text` | 已有文字稿，写入 `data.transcript` | `--text-column` |
| `is_longer_than_10_seconds` | `true` 排除，`false` 保留 | `--long-clip-column` |

**筛选完全依据 manifest 标记，不读取音频来重新测量时长。** 当前列名表示 10 秒，使用的是 10 秒标记，
不是此前讨论的 15 秒规则。如果以后 manifest 提供了 15 秒标记列，可用
`--long-clip-column is_longer_than_15_seconds` 指定该列。布尔值、数值 `1/0` 和字符串
`true/false`、`1/0` 均可使用；字符串忽略大小写和两端空白。标记缺失或无效时会报错，不会猜测。

`audio_clip_file` 的路径解析规则：

- 仅文件名（如 `clip_0001.wav`）：默认从 manifest 旁的 `audio/` 中查找；可用 `--audio-dir` 指定其他目录。
- 带目录的相对路径（如 `audio/clip_0001.wav`）：相对 manifest 所在的批次目录解析。
- 绝对路径：直接使用，但必须位于 document root 内。

支持 WAV、MP3、FLAC、OGG、M4A。脚本只读取 manifest 中保留行引用的音频，不扫描音频目录，也不复制、
删除或修改音频。未列入 manifest 或被标记排除的文件可以留在 `audio/` 中；被排除的行不检查音频是否存在。
全部行被排除时不生成任何批次 JSON。空白、null 或 NaN 转写写为空字符串并报告数量；其他非文本转写会报错。
保留行的 clip ID 和音频真实路径必须各自唯一，缺列、缺失音频或重复记录都会报错。
manifest、音频目录及引用音频的真实路径必须位于 document root 内；指向根目录外的 symlink 被拒绝。
全部验证成功后才写输出；若输出目录已有 `tasks_*.json`，脚本会报错，避免覆盖或混入旧分组。
重复生成时请选择新的输出目录。运行结束会显示 seed、各文件任务数及排除数量。

每份任务 JSON 的形态不变：

```json
[
  {
    "data": {
      "audio": "/data/local-files/?d=batch/audio/clip_0001.wav",
      "transcript": "已有的 clip 转写",
      "clip_id": "clip_0001"
    }
  }
]
```

脚本对 URL 中的特殊字符编码。`?d=` 后是相对 **document root** 的路径，不是相对 Source path；
因此这里必须保留 `batch/`。不要使用 `file://` 或裸本机绝对路径。
原始 call/turn/segment 时间戳保留在数据准备 manifest 中，通过稳定的 `clip_id` 与标注导出关联。
公司音频、转写、任务 JSON 和导出均保留在公司授权目录，不写入本仓库。

## 2. 用 Local Files / Tasks 同步导入

项目内进入 **Settings → Cloud Storage → Add Source Storage → Local Files**：

1. Source path / Absolute local path 填 `/absolute/path/to/dataset/batch/tasks`，是 document root 的严格子目录。
2. 点击 **Test Connection**；成功后点 **Next**。
3. **Import Method** 选 **Tasks**，先只导入第一份，文件过滤填 `^tasks_001\.json$`。
4. 点击 **Load Preview**，确认预览读取到任务而不是把 JSON 当成音频文件。
5. 点击 **Next → Save & Sync**，回到数据页核对任务数，并试播一条。

本次公开演示采用的就是 Tasks 同步导入。下图中的文件名是 `import_tasks.json`，所以过滤条件实际为
`import_tasks\.json$`；使用上面的分批目录时，请对应使用 `^tasks_001\.json$`。

![Local Files 向导：Tasks、文件过滤和成功加载的预览](screenshots/04-import-settings.jpg)

仅同步过滤后的任务 JSON，不要再用 **Files** 模式扫描整个音频目录，否则会绕过 manifest 的筛选。
更新 JSON 不会自动删除此前已导入的任务。

## 3. 或通过 Import 上传 JSON

如果负责人已配置好音频的本地文件访问，建议在项目数据页用 **Import → Upload Files** 逐批上传：

1. 先只上传 `tasks_001.json`，完成这一批标注。
2. 在同一个项目里上传 `tasks_002.json`，继续下一批；现有标注结果保留。
3. 按需要继续，做到几千条后可以停止，其余文件无需导入。

上传 JSON 只导入任务字段，不会上传 JSON 中引用的音频。不要一次同步所有分批文件，否则全部任务都会进入项目。
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
| 1 | 本机运行 `link_upload_audio.py`，把一份 `tasks_NNN.json` 的音频硬链接到 `part_01/`、`part_02/` … | 各 part 的文件数 |
| 2 | 在 LSE 建项目，模板用 [config_audio_only.xml](config_audio_only.xml) | 预览只有音频和选项 |
| 3 | 打开 **Import**，每次选中一个 part 文件夹里的全部文件拖进去 | 任务数等于已上传的文件数 |
| 4 | 运行 `attach_task_data.py`，先试运行（`write: false`） | 没有报错；`To update` 等于文件数 |
| 5 | 改为 `write: true`（`limit: 3`），只写 3 条 | Data Manager 出现 `transcript`、`clip_id` 列，这 3 条有值 |
| 6 | 改为 `limit: null`，写入全部 | 提示 `All tasks of this file are done` |
| 7 | **Settings → Labeling Interface → Code** 换成 [config_with_transcript.xml](config_with_transcript.xml) | 打开一条任务：能播放，文字对应 |
| 8 | 这时才让标注员开始 | |

两个脚本都在仓库根目录、用已安装 [requirements.txt](requirements.txt) 的 Python 运行（`emotion_recognition_llm` 环境已包含），
不用命令行参数，设置写在 `label_studio/config/` 的 YAML 中。

### 4.1 建立上传文件夹（硬链接）

`label_studio/config/link_upload_audio.yaml`：

```yaml
tasks_file: /absolute/path/to/dataset/batch/tasks/tasks_001.json
document_root: /absolute/path/to/dataset
upload_dir: /absolute/path/to/dataset/batch/audio/upload_001  # 新的或空的文件夹
files_per_folder: 100
```

```bash
python3 label_studio/link_upload_audio.py
```

- 上传文件夹里是**硬链接**：不占额外磁盘空间。Finder 显示的文件夹大小可能仍是完整大小，但磁盘用量不变，
  对整个批次文件夹运行 `du -sh` 只算一次。原音频不移动、不修改，manifest 和任务 JSON 的路径照常有效。
- 上传后可以删除上传文件夹，原文件不受影响。**不要编辑上传文件夹里的文件**，那样原文件也会一起改变。
- 硬链接要求与音频在同一磁盘。exFAT 和部分网络盘不支持；脚本会报错并清理已建的链接，不会改为复制。
- 每个 part 最多 `files_per_folder` 个文件。开源版每次上传默认最多 100 个文件（`DATA_UPLOAD_MAX_NUMBER_FILES`），
  LSE 以管理员设置为准；上传被拒时调小这个值。
- Label Studio 会把空格改成 `_`、删掉 `#` 等符号，并在文件名前加 8 位随机前缀；脚本会自动对应。
  因此文件名必须唯一（忽略大小写和被删的符号），且不超过 70 个字符，否则服务器会截短文件名。

### 4.2 补上转写和 clip_id

`label_studio/config/attach_task_data.yaml`：

```yaml
project_id: 1      # 项目网址中的数字：https://<your-label-studio-host>/projects/<id>/data
tasks_file: /absolute/path/to/dataset/batch/tasks/tasks_001.json
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

### 4.3 同一项目的后续批次

只在两批之间、没有人标注时进行：

1. 模板换回 `config_audio_only.xml`。两份模板的选项相同，已有标注保留。
2. 对 `tasks_002.json` 运行 `link_upload_audio.py`（新的 `upload_dir`，例如 `upload_002`），上传各 part。
3. 把 `attach_task_data.yaml` 的 `tasks_file` 改为 `tasks_002.json`，按上表第 4–6 步运行。
   前面批次的任务显示为 `other batches (ignored)`，不会被修改。
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
