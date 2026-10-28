# JL-Corpus

This note was added by this project; it is not part of the dataset.

**JL-Corpus**, an emotional speech corpus of New Zealand English, by Jesin James, Li Tian and Catherine Watson.

- Source: Kaggle, https://www.kaggle.com/datasets/tli725/jl-corpus (uploaded by the authors; see also
  https://github.com/tli725/JL-Corpus)
- Licence: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain dedication), as set on Kaggle.
- Paper: Jesin James, Li Tian, Catherine Watson. An Open Source Emotional Speech Corpus for Human Robot Interaction
  Applications. Interspeech 2018.
- `scripts/download_data.py` checked the Kaggle zip and unpacked it into `audio/`. The evaluation uses
  `audio/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/`: 2,400 WAV files (44.1 kHz) with their transcripts.

Only that folder, `Format_Intro.txt` next to it and the authors' `README.md` belong in git. Leave out the rest of
the zip: the perception-test folder holds a sheet with personal details of the listeners, and `Picture Stimuli/`
holds stock photos whose copyright belongs to others.
