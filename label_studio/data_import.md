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

## 4. 读取并标准化导出

先在 Label Studio **Export → JSON** 下载并保留完整原生文件。以 `data.clip_id` 关联 manifest。
原生 `result` 中 `from_name="decision"`、`type="choices"` 的 `value.choices` 保存唯一决策，
包括 8 个情绪值或 `bad_clip` 弃用决策；Bad clip 不是情绪类别。

使用本项目标准化脚本生成业务数据，保留原生文件：

```bash
python3 label_studio/normalize_export.py \
  --input /absolute/path/to/exports/label-studio.json \
  --output /absolute/path/to/exports/emotion-votes.json
```

脚本按每条 annotation 输出一行记录，**保留每位标注员的 vote，不聚合、不覆盖为一个最终标签**。
保留 `clip_id`、`task_id`、`annotation_id`、`annotator_id`、`lead_time`、`decision`、`status` 和 `emotion`。
`completed_by` 是用户对象时只取其 ID，不输出邮箱。

- 8 个情绪决策，包括 `unknown`：`status="accepted"`，`emotion` 为相应情绪值。
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
音频可用但情绪不确定时选 Unknown；音频本身无法可靠判断时选 Bad clip，后处理排除这类音频。

取消的 annotation（`was_cancelled`）和无 annotation 的任务会跳过，并报告数量。
Label Studio 默认导出已标注任务，未提交任务不默认包含在内。
缺失 `clip_id` 或 `decision`、多个决策/多个选择、未知值均会报错，验证全部成功后才写输出；已有输出不会被覆盖。

官方依据：[本地存储](https://labelstud.io/guide/storage_local)、[Choices](https://labelstud.io/tags/choices)、
[只读 Text](https://labelstud.io/tags/text)、[导出字段](https://labelstud.io/guide/export.html)。
