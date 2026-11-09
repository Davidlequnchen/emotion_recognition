# Prepared context experiments

The current EmotionTalk pilot contains **prepared inputs only**, not model predictions.
Each versioned folder freezes the selected train targets, histories, controls and prompts.
An explicit `context_pilot.py run` is required to create `runs/*/responses.jsonl` and analysis outputs.
See [the explanation and commands](../../docs/context_pilot.md).

These input artifacts contain selected EmotionTalk transcripts, released gold labels and derived metadata.
EmotionTalk is **CC BY-NC-SA 4.0**; retain that licence and attribution when sharing these data-derived files.
Original audio is not copied into this folder. Selection, bracket-annotation removal and condition packaging
are this project's processing steps, described in the guide and manifests.

Source: Haoqin Sun, Jinghua Zhao, Xuechen Wang, Shiwan Zhao, Jiaming Zhou, Hui Wang, Xi Yang,
Yequan Wang and Yonghua Lin. *EmotionTalk: An Interactive Chinese Multimodal Emotion Dataset
With Rich Annotations*. Findings of ACL 2026.
[Paper](https://aclanthology.org/2026.findings-acl.440/),
[official data](https://huggingface.co/datasets/BAAI/Emotiontalk),
[licence](https://creativecommons.org/licenses/by-nc-sa/4.0/).
