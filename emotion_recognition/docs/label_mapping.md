# Label mapping（类别对应表）

| Project class（= emotion2vec+ 的类别名） | emotion2vec+ large | MERaLiON-SER-v1 | Melati | Vox-Profile Whisper / WavLM | CREMA-D | MCAE-SPPS | MESS | RAVDESS | TESS | JL-Corpus | MEAD | MELD（测试集） | EMOVO | Emozionalmente | SAVEE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| angry | 生气/angry | angry | angry | Anger | ANG | 3 Anger | A (angry) | 05 angry | angry | angry | angry | anger | rab (rabbia) | anger | a (anger) |
| disgusted | 厌恶/disgusted | disgusted | disgusted | Disgust | DIS | 6 Disgust | – | 07 disgust | disgust | – | disgusted | disgust | dis (disgusto) | disgust | d (disgust) |
| fearful | 恐惧/fearful | fearful | fearful | Fear | FEA | 4 Fear | – | 06 fearful | fear | – | fear | fear | pau (paura) | fear | f (fear) |
| happy | 开心/happy | happy | happy | Happiness | HAP | 2 Happiness | H (happy) | 03 happy | happy | happy；excited（预测 happy 算对） | happy | joy | gio (gioia) | joy | h (happiness) |
| neutral | 中立/neutral | neutral | neutral | Neutral | NEU | 1 Neutral | C (calm) | 01 neutral；02 calm | neutral | neutral | neutral（只有一个强度） | neutral | neu (neutrale) | neutrality | n (neutral) |
| sad | 难过/sad | sad | sad | Sadness | SAD | 5 Sadness | S (sad) | 04 sad | sad | sad | sad | sadness | tri (tristezza) | sadness | sa (sadness) |
| surprised | 吃惊/surprised | surprised | surprised | Surprise | – | 7 Surprise | – | 08 surprised | ps (pleasant surprise) | excited（预测 surprised 也算对） | surprised | surprise | sor (sorpresa) | surprise | su (surprise) |
| 不参与评估 | 其他/other；`<unk>`（从不作为预测） | – | – | Contempt；Other（从不作为预测） | – | – | – | – | – | anxious、apologetic、assertive、concerned、encouraging（这 1,200 条片段不评估） | contempt（不评估，同 THAI-SER 的 frustrated） | `dia38_utt4`、`dia220_utt0`（时间戳错误，每条含 4–5 分钟剧集，不评估） | – | – | – |

新增数据集的说明（2026-11-05）：

- **MEAD**：情绪来自文件夹名，三个强度（level_1 弱、level_2 中、level_3 强）全部保留；强度写在加载器的 clip 字典里（`intensity`），报告不用。contempt（轻蔑）没有对应类别，不评估。自 2026-11-06 起只评估 8,000 条的分层子集（见下）。
- **MELD**：只用官方测试集（2,610 条，评估 2,608 条）。标签是三位标注者看视频和字幕后的多数票，不是演员按要求表演的情绪；train/dev 被多个模型用于训练，所以不用。
- **EMOVO**：文件名第一段是意大利语情绪缩写；官方网站已失效，数据来自 DagsHub 上的非官方副本。
- **Emozionalmente**：`acted` 是说话人想表达的情绪（`emotion_expressed`），`listener` 是 5 位听众的多数票（平票时列出所有并列情绪，同 CREMA-D）。

新增数据集的说明（2026-11-06）：

- **SAVEE**：文件名里说话人后面的字母是情绪（`DC_a01.wav` = 说话人 DC、anger、第 1 句；`sa`/`su` 是两个字母）；4 位男性说话人，480 条（neutral 120 条，其他每类 60 条）。数据来自 Kaggle 上的非官方副本（官方版需注册，许可禁止再分发），所以 `public_audio` 为 False，不放公开网页；和除 ESD 外的所有数据集一样，研究评估时会发给 API 模型（`api_audio`）。

评估子集（2026-11-06 决定，为了节省算力；`config/evaluation.yaml` 的 `subsets`，写在加载器里，所有模型和脚本用同样的片段）：

- **THAI-SER**：只用一致率 `agreement` ≥ 0.71 的片段（数据集作者推荐的高一致子集），4 类共 11,848 条（全部 22,584 条）。标签映射不变；注意一致率是评分者之间的一致，不是与指定情绪的一致：其中 2,698 条的多数票仍不同于 `assigned_emo`（多为 frustrated 或 neutral）。
- **MEAD**：8,000 条分层子集（全部 27,514 条）：按演员、情绪、强度、句子编号排序后等间隔取，每个演员/情绪/强度格子按比例保留。标签映射不变。
