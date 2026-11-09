# 2026-11-09：清理主仓库的数据与模型历史

主仓库以前直接保存过数据、权重及 90 MiB 分片。现在数据和模型分别由两个 submodule 管理，大文件放在各自仓库的 GitHub Releases，主仓库无需再保留旧副本。

## 这次改了什么

使用 `git filter-repo` 从历史中移除以下目录的文件内容，包含重命名前的旧路径：

```text
data/
models/
emotion_recognition/data/
emotion_recognition/models/
```

精确过滤表达式为 `^(?:emotion_recognition/)?(?:data|models)/`。末尾的斜杠用于保留当前 `emotion_recognition/data` 和 `emotion_recognition/models` 这两个 `160000` submodule 引用；`.gitmodules` 也保留。

代码、配置、研究文档、`outputs/`、`reports/`、网页和演示文件均保留。只有数据/模型改动、过滤后为空的提交会被移除，其余提交的 ID 会相应变化。旧 ID 到新 ID 的映射见 [commit map](git_history_commit_map.txt)；值全为零表示该提交被移除。数据和模型子仓库本身没有类似的大文件 Git 历史，本次未重写它们。

## 验证与空间

- 改写前后最新提交的文件树 hash 完全相同：`d1ef5ef5016735d96a2a4241770f08e928e88b6a`。
- 两个 submodule 的路径、mode 和当时固定的 commit 相同；原始模型、数据、features 与输出文件留在本机。
- 615 个已跟踪工作文件在切换本地 refs 前后 SHA-256 相同，未提交的修改也保留。
- 当前分支及本地 Codex checkpoint 中的旧数据/模型内容均清除；checkpoint 的其余文件保留。
- 独立过滤副本、本地仓库和从 GitHub 全新克隆的副本均通过 `git fsck`；新的可达对象清单不含上述目录内部文件。
- 主 Git 对象包从 **15.83 GiB** 降至 **113.85 MiB**。包含子仓库 Git 元数据的本机 `.git` 从约 **17.06 GB** 降至 **0.17 GB**。远端全新 bare clone 的对象包实测 **120.06 MiB**；打包方式不同会导致少量差异。

主分支使用精确旧 SHA 的 `--force-with-lease` 更新，避免覆盖另一个会话的新提交；随后同步本地 refs、清理旧 reflog 和不可达对象。GitHub 后台仍可能暂时显示旧的仓库大小，不能把页面计数当作此次 fresh-clone 验证结果。此次检查没有发现远端 PR refs 或额外分支/标签。

## 其他电脑怎样同步

**如果是历史改写前的旧 clone，先保存本地工作与实验结果，不要直接把旧历史 merge 回来。** 公司服务器上的 features、CSV、JSON、JSONL 也应保留；本次不要求重跑实验。

最容易核对的方式是在新目录 clone，原目录先保留：

```bash
git clone --recurse-submodules https://github.com/Davidlequnchen/emotion_recognition.git emotion_recognition-clean
```

这只获取代码和 submodule 元数据，不自动下载 Release 大文件。再按需要恢复模型/数据，或者核验后复用原目录里已有的大文件。不要把旧仓库的 `.git` 复制进新目录。

```bash
cd emotion_recognition-clean
python3 emotion_recognition/scripts/common/setup_models.py gemma4_e4b  # 示例：只恢复一个模型
python3 emotion_recognition/scripts/common/setup_datasets.py meld    # 示例：只恢复一个数据集
```

已完成历史切换的 clone 可正常 `git pull`。后续继续把权重和音频保存在 Releases，主仓库仅更新代码、结果和 submodule 指针。
