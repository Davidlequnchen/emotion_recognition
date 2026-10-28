---
language:
- pl
license: cc-by-nc-sa-4.0
multilinguality: monolingual
size_categories:
- 1K<n<10K
task_categories:
- audio-classification
- automatic-speech-recognition
- text-to-speech
task_ids:
- audio-emotion-recognition
- audio-language-identification
- sentiment-analysis
- speaker-identification
pretty_name: nEMO
dataset_info:
  features:
  - name: file_id
    dtype: string
  - name: audio
    dtype: audio
  - name: emotion
    dtype: string
  - name: raw_text
    dtype: string
  - name: normalized_text
    dtype: string
  - name: speaker_id
    dtype: string
  - name: gender
    dtype: string
  - name: age
    dtype: int32
  splits:
  - name: train
    num_bytes: 531802170.875
    num_examples: 4481
  download_size: 531224302
  dataset_size: 531802170.875
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
# nEMO: Dataset of Emotional Speech in Polish

## Dataset Description

nEMO is a simulated dataset of emotional speech in the Polish language. The corpus contains over 3 hours of samples recorded with the participation of nine actors portraying six emotional states: anger, fear, happiness, sadness, surprise, and a neutral state. The text material used was carefully selected to represent the phonetics of the Polish language. The corpus is available for free under the Creative Commons license (CC BY-NC-SA 4.0).

### Example Usage

The nEMO dataset can be loaded and processed using the datasets library:

```python
from datasets import load_dataset

nemo = load_dataset("amu-cai/nEMO", split="train")
```

### Supported Tasks

- `audio-classification`: This dataset was mainly created for the task of speech emotion recognition. Each recording is labeled with one of six emotional states (anger, fear, happiness, sadness, surprised, and neutral). Additionally, each sample is labeled with speaker id and speaker gender. Because of that, the dataset can also be used for different audio classification tasks.
- `automatic-speech-recognition`: The dataset includes orthographic and normalized transcriptions for each audio recording, making it a useful resource for automatic speech recognition (ASR) tasks. The sentences were carefully selected to cover a wide range of phonemes in the Polish language.
- `text-to-speech`: The dataset contains emotional audio recordings with transcriptions, which can be valuable for developing TTS systems that produce emotionally expressive speech.

### Languages

nEMO contains audio and transcription in Polish language.

## Dataset Structure

### Data Instances

```python
{
  'file_id': 'WR0_surprised_64',
  'audio': {
    'path': None,
    'array': array([ 6.10351562e-05, -3.05175781e-05, -3.05175781e-05, ...,
      6.10351562e-05, -1.22070312e-04,  1.83105469e-04]),
    'sampling_rate': 24000
  },
  'emotion': 'surprised',
  'raw_text': 'Ucho wykrywa dźwięki o różnej częstotliwości.',
  'normalized_text': 'ucho wykrywa dźwięki o różnej częstotliwości',
  'speaker_id': 'WR0',
  'gender': 'male',
  'age': 23
}
```

### Data Fields

- `file_id` (string) - id of file, i.e. `{speaker_id}_{emotion}_{sentence_id}`,
- `audio` (audio) - dictionary containing audio array, path and sampling rate,
- `emotion` (string) - label corresponding to emotional state,
- `raw_text` (string) - original (orthographic) transcription of the audio,
- `normalized_text` (string) - normalized transcription of the audio,
- `speaker_id` (string) - id of speaker,
- `gender` (string) - gender of the speaker,
- `age` (int32) - age of the speaker.

### Data Splits

The nEMO dataset is provided as a whole, without predefined training and test splits. This allows researchers and developers flexibility in creating their splits based on the specific needs.

| Emotion   | Number of samples |
| --------- | ----------------- |
| anger     | 749               |
| fear      | 736               |
| happiness | 749               |
| neutral   | 809               |
| sadness   | 769               |
| surprised | 669               |
| **Total** | 4,481             |

## Additional Information

### Licensing Information

The dataset is available under the Creative Commons license (CC BY-NC-SA 4.0).

### Citation Information

You can access the nEMO paper at [ACL Anthology](https://aclanthology.org/2024.lrec-main.1059/). Please cite the paper when referencing the nEMO dataset as:

```
@inproceedings{christop-2024-nemo-dataset,
    title = "n{EMO}: Dataset of Emotional Speech in {P}olish",
    author = "Christop, Iwona",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1059",
    pages = "12111--12116",
    abstract = "Speech emotion recognition has become increasingly important in recent years due to its potential applications in healthcare, customer service, and personalization of dialogue systems. However, a major issue in this field is the lack of datasets that adequately represent basic emotional states across various language families. As datasets covering Slavic languages are rare, there is a need to address this research gap. This paper presents the development of nEMO, a novel corpus of emotional speech in Polish. The dataset comprises over 3 hours of samples recorded with the participation of nine actors portraying six emotional states: anger, fear, happiness, sadness, surprise, and a neutral state. The text material used was carefully selected to represent the phonetics of the Polish language adequately. The corpus is freely available under the terms of a Creative Commons license (CC BY-NC-SA 4.0).",
}
```

### Contributions

Thanks to [@iwonachristop](https://github.com/iwona-christop) for adding this dataset.