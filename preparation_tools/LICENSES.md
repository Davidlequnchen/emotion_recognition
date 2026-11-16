# Licences of the downloaded components

Everything here was picked so it can be used inside the company (commercial, internal use). The wheels and the
checkpoint are kept unchanged and are not stored in Git; they are mirrored unchanged, together with this file, in
the GitHub Release `preparation-tools-segmenter-v1` of this repository.
[`dependencies/manifest.json`](dependencies/manifest.json) and [`model_weights/manifest.json`](model_weights/manifest.json)
record their source URLs, sizes and SHA-256.
Licences were read from each wheel's `METADATA`/licence files and from torchaudio 2.7.1's model definition on
2026-11-13.

## Wheels (`dependencies/`)

| Package | Version | Platforms | Licence | Notes |
|---|---|---|---|---|
| silero-vad | 6.2.3 | any | MIT | The JIT model `silero_vad/data/silero_vad.jit` (2.27 MB) is inside the wheel, together with ONNX variants that are not used; no download at run time |
| soundfile | 0.14.0 | Linux x86_64, macOS arm64 | BSD-3-Clause | Bundles libsndfile 1.2.2 (LGPL-2.1) with libmpg123 (LGPL-2.1) and libmp3lame (LGPL-2.0+) for MP3, and libFLAC, libogg, libvorbis, libopus (BSD-3-Clause). Loaded as a separate shared library, unmodified; the wheel's `licensing/license_notes.md` lists the sources |
| cffi | 2.1.1 | Linux x86_64, macOS arm64 | MIT-0 (as declared in the wheel's METADATA) | Needed by soundfile |
| pycparser | 3.12 | any | BSD-3-Clause | Needed by cffi |
| PyYAML | 6.0.3 | Linux x86_64, macOS arm64 | MIT | Reads the YAML settings |
| packaging | 26.3 | any | Apache-2.0 OR BSD-2-Clause | Needed by silero-vad |

Not included on purpose: `onnxruntime` and `torchcodec` (optional silero-vad extras; the segmenter uses the JIT
model and soundfile).

## Prerequisites (not duplicated here)

| Package | Version | Licence | Where it comes from |
|---|---|---|---|
| torch | 2.7.1 | BSD-3-Clause | Linux: `emotion_recognition/dependencies/torch-cu118-cp311-linux_x86_64/`; Mac: `emotion_recognition/requirements-llm-mac.txt` |
| torchaudio | 2.7.1 | BSD-2-Clause | as above; provides `forced_align` and the model definition |
| numpy | 2.4.6 in both environments | BSD-3-Clause | comes with the torch installation |
| typing-extensions | 4.16.0 on Linux | PSF-2.0 | comes with the torch installation |

## Model weights (`model_weights/`)

| Model | File | Size | Licence |
|---|---|---|---|
| torchaudio `WAV2VEC2_ASR_BASE_960H` (94.4M parameters, English, fine-tuned on LibriSpeech 960 h) | `wav2vec2_fairseq_base_ls960_asr_ls960.pth` from download.pytorch.org | 377,664,473 bytes | MIT: "Originally published by the authors of wav2vec 2.0 under MIT License and redistributed with the same license" (torchaudio 2.7.1 bundle documentation; fairseq licence at commit `ce6c9ee`) |

Not used, because they are not licensed for commercial use: torchaudio `MMS_FA` (CC BY-NC 4.0) and the VoxPopuli
wav2vec 2.0 models (CC BY-NC 4.0).

Candidate for later Spanish alignment (not downloaded): `jonatasgrosman/wav2vec2-large-xlsr-53-spanish`,
Apache-2.0 (check again before use).

## Test data

The integration check (`tests/integration_dailytalk.py`) reads the local public DailyTalk copy
(CC BY-SA 4.0) and writes its synthetic turns only to the git-ignored `local/` folder. The presentation includes
11 adapted DailyTalk audio clips and derived figures under CC BY-SA 4.0, with attribution retained in the deck;
see [the presentation's data and licences](presentation/README.md#data-and-licences). No internal call data are
part of this subproject.
