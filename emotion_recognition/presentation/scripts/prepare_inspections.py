"""Select traceable ten-model listening cases from the completed benchmark.

This is a read-only analysis of stored predictions, not a model run. The output
manifest records the source hashes, selection rule, and exact audio for replay.
Audio is converted to mono 16 kHz without trimming or loudness normalization.
"""
import collections
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common import CREMA_CODE, CREMA_SENT, RAVDESS_SENT, MCAE_COLS, crema_votes, mcae_info

DECK = Path(__file__).resolve().parents[1]
ROOT = DECK.parent
BUILD = DECK / '.inspection-build'
ASSETS = DECK / 'assets' / 'inspections'
MODELS = ['emotion2vec_plus_large', 'meralion_ser_v1', 'melati_emotion',
          'vox_profile_whisper', 'vox_profile_wavlm', 'qwen2_5_omni_7b',
          'qwen2_audio_7b', 'gemma4_e4b', 'gpt_realtime', 'gemini-3.8-flash']
DATASETS = ['meld', 'emotiontalk', 'dailytalk', 'multidialog', 'crema_d',
            'ravdess', 'savee', 'jl_corpus', 'tess', 'mead', 'mess',
            'mcae_spps', 'thai_ser', 'nemo', 'emovo', 'emozionalmente']


def csv_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(args):
    return subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def duration(path):
    return float(checked(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                          '-of', 'default=noprint_wrappers=1:nokey=1', str(path)]))


def make_audio(case):
    source = ROOT / case['audio_path']
    stem = case['dataset_key'] + '__' + Path(case['audio_path']).stem
    audio = ASSETS / 'audio' / (stem + '.mp3')
    waveform = ASSETS / 'waveforms' / (stem + '.png')
    checked(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-ac', '1', '-ar', '16000',
             '-c:a', 'libmp3lame', '-b:a', '64k', '-map_metadata', '-1', str(audio)])
    raw = checked(['ffmpeg', '-v', 'error', '-i', str(source), '-ac', '1', '-ar', '16000',
                   '-f', 'f32le', '-'])
    samples = np.frombuffer(raw, dtype='<f4')
    # A shared absolute amplitude scale preserves the relative levels between takes.
    w, h = 1000, 100
    image = Image.new('RGB', (w, h), '#F4F6F9')
    draw = ImageDraw.Draw(image)
    draw.line((0, h//2, w, h//2), fill='#D5DAE1')
    bins = np.linspace(0, len(samples), w+1, dtype=int)
    for i in range(w):
        segment = samples[bins[i]:max(bins[i]+1, bins[i+1])]
        if len(segment):
            lo, hi = np.clip([segment.min(), segment.max()], -1, 1)
            draw.line((i, h/2-hi*46, i, h/2-lo*46), fill='#0072B2')
    image.save(waveform)
    case.update(audio_asset=str(audio.relative_to(DECK)),
                waveform_asset=str(waveform.relative_to(DECK)),
                audio_sha256=sha(source), duration_s=round(len(samples)/16000, 3))


def main():
    for directory in [BUILD, ASSETS/'audio', ASSETS/'waveforms']:
        directory.mkdir(parents=True, exist_ok=True)
    (BUILD/'.gitignore').write_text('*\n')
    analysis_path = ROOT.parent / 'research_paper/output/cross_dataset_analysis.json'
    analysis = json.loads(analysis_path.read_text())
    selected = {(x['model'], x['dataset']): x for x in analysis['manifest']['selection']
                if x['status'] == 'selected'}
    assert len(selected) == 160 and set(analysis['manifest']['full_models']) == set(MODELS)
    sources, datasets, loaded = [], [], {}
    for ds in DATASETS:
        rows_by_model, run_by_model = {}, {}
        for model in MODELS:
            json_path = ROOT / selected[model, ds]['path']
            run = json.loads(json_path.read_text())
            csv_path = json_path.with_suffix('.csv')
            rows = csv_rows(csv_path)
            assert len(rows) == selected[model, ds]['clips'] == run['clips']
            rows_by_model[model] = {x['audio_path']: x for x in rows}
            assert len(rows_by_model[model]) == len(rows)
            run_by_model[model] = run
            for file in [json_path, csv_path]:
                sources.append({'path': str(file.relative_to(ROOT)), 'sha256': sha(file)})
        canonical = rows_by_model[MODELS[0]]
        for model in MODELS:
            assert set(rows_by_model[model]) == set(canonical), (model, ds, 'cohort mismatch')
            assert all(row['acted_emotion'] == rows_by_model[model][path]['acted_emotion']
                       for path, row in canonical.items())
            acc = sum(row['correct'].lower() in ('true', 'yes', '1')
                      for row in rows_by_model[model].values()) / len(canonical)
            saved = next(x for x in analysis['model_rows'] if x['model'] == model and x['dataset'] == ds)
            assert abs(acc - saved['accuracy']) < 1e-8, (model, ds, 'accuracy mismatch')
        loaded[ds] = (rows_by_model, run_by_model)
        print(f'Verified {ds}: {len(canonical):,} identical clip IDs across 10 models', flush=True)

    voice_votes = crema_votes()
    mcae = mcae_info()
    dialogue = {ds: {row['audio_path']: row for row in csv_rows(ROOT/'data'/ds/'utterances.csv')}
                for ds in ['emotiontalk', 'dailytalk', 'multidialog']}
    emo_meta = {row['file_name']: row for row in csv_rows(
        ROOT/'data/emozionalmente/audio/emozionalmente/metadata/samples.csv')}
    emo_votes = collections.defaultdict(collections.Counter)
    emo_map = dict(zip(['anger', 'disgust', 'fear', 'joy', 'neutrality', 'sadness', 'surprise'],
                       ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']))
    for row in csv_rows(ROOT/'data/emozionalmente/audio/emozionalmente/metadata/evaluations.csv'):
        emo_votes[row['file_name']][emo_map[row['emotion_recognized']]] += 1
    thai = {row['audio_id']: row for row in csv_rows(ROOT/'data/thai_ser/labels.csv')}
    meld = {f"dia{x['Dialogue_ID']}_utt{x['Utterance_ID']}": x
            for x in csv_rows(ROOT/'data/meld/test_sent_emo.csv')}

    def case_info(ds, path):
        rows_by_model, runs = loaded[ds]
        row = rows_by_model[MODELS[0]][path]
        name = Path(path).stem
        result = {'dataset_key': ds, 'clip_id': name, 'audio_path': path,
                  'reference': row['acted_emotion'], 'listener_majority': row.get('listener_vote', ''),
                  'text': '', 'speaker': '', 'listener_votes': {}, 'extra': '',
                  'predictions': {m: {'label': rows_by_model[m][path]['prediction'],
                                     'correct': rows_by_model[m][path]['correct'].lower() in ('true','yes','1')}
                                  for m in MODELS}}
        if ds == 'crema_d':
            actor, sentence, emotion, intensity = name.split('_')
            result.update(text=CREMA_SENT[sentence], speaker=actor, extra=f'Intensity: {intensity}',
                          listener_votes={CREMA_CODE[k]: v for k,v in voice_votes[name].items()
                                          if k in CREMA_CODE and v})
        elif ds == 'ravdess':
            parts = name.split('-')
            result.update(text=RAVDESS_SENT[parts[4]], speaker='Actor '+parts[6],
                          extra='Intensity: '+('strong' if parts[3] == '02' else 'normal'))
        elif ds == 'tess':
            speaker, word, _ = name.split('_')
            result.update(text=f'Say the word {word}.', speaker=speaker)
        elif ds == 'jl_corpus':
            result.update(text=(ROOT/path).with_suffix('.txt').read_text().strip(), speaker=name.split('_')[0])
        elif ds == 'mcae_spps':
            meta = mcae[name]
            result.update(text=meta['sentence content'].strip(), speaker=name[0],
                          listener_votes={v: int(meta[k]) for k,v in MCAE_COLS.items() if meta[k]})
        elif ds == 'mess':
            result.update(speaker=name[1:3], extra='Corpus applies loudness equalization')
        elif ds == 'mead':
            parts = Path(path).parts
            result.update(speaker=parts[-4], extra='Intensity: '+parts[-2].replace('_', ' '))
        elif ds == 'nemo':
            result.update(speaker=name.split('_')[0])
        elif ds == 'savee':
            result.update(speaker=name.split('_')[0])
        elif ds == 'emovo':
            result.update(speaker=name.split('-')[1], extra='Sentence ID: '+name.split('-')[-1])
        elif ds == 'emozionalmente':
            meta = emo_meta[Path(path).name]
            result.update(text=meta['sentence'], speaker=meta['actor'][:8],
                          listener_votes=dict(emo_votes[Path(path).name]))
        elif ds == 'thai_ser':
            meta = thai[Path(path).name]
            result.update(speaker=meta['actor_id'], text=meta['script_sent'],
                          extra=f"Rater agreement: {100*float(meta['agreement']):.0f}%")
        elif ds == 'meld':
            meta = meld[name]
            result.update(text=meta['Utterance'], speaker=meta['Speaker'])
        elif ds in dialogue:
            rel = path.removeprefix(f'data/{ds}/')
            meta = dialogue[ds][rel]
            result.update(text=meta['text'], speaker=meta['speaker_id'])
            if ds == 'emotiontalk':
                result['listener_votes'] = json.loads(meta['votes_json'])
        result['ser_correct'] = sum(result['predictions'][m]['correct'] for m in MODELS[:5])
        result['llm_correct'] = sum(result['predictions'][m]['correct'] for m in MODELS[5:])
        return result

    # First inspect a contrast with strong SER performance, then a different
    # emotion where LLMs do better. Fall back to a split case if no such clip exists.
    for ds in DATASETS:
        models, runs = loaded[ds]
        candidates = []
        for path in models[MODELS[0]]:
            c = case_info(ds, path)
            labels = [p['label'] for p in c['predictions'].values()]
            c['distinct_predictions'] = len(set(labels))
            c['tie_break'] = hashlib.sha256(path.encode()).hexdigest()
            if len(c['text']) <= 180 and c['distinct_predictions'] >= 2:
                candidates.append(c)
        picks = []
        for index in range(2):
            def score(c):
                a,b = c['ser_correct'],c['llm_correct']
                if index == 0:
                    return (a-b, a, c['distinct_predictions'], c['tie_break'])
                return (b-a, b, c['distinct_predictions'], c['tie_break'])
            ranked = sorted(candidates, key=score, reverse=True)
            for c in ranked:
                if picks and (c['reference'] == picks[0]['reference'] or c['speaker'] == picks[0]['speaker']):
                    continue
                f = ROOT / c['audio_path']
                if not f.is_file():
                    continue
                d = duration(f)
                if not 1.5 <= d <= 10:
                    continue
                c['selection'] = 'largest SER-minus-LLM correctness gap' if index == 0 else 'largest LLM-minus-SER correctness gap with a different emotion and speaker'
                make_audio(c)
                picks.append(c)
                break
            assert len(picks) == index+1, (ds, 'no playable case')
        for c in picks:
            c.pop('tie_break')
        run = runs[MODELS[0]]
        datasets.append({'key': ds, 'name': run['dataset'], 'language': run['language'],
                         'clips': run['clips'], 'accept': run.get('accept', {}),
                         'ground_truth_description': run['columns']['acted_emotion'], 'cases': picks})
        print(ds, [(c['clip_id'], c['reference'], c['ser_correct'], c['llm_correct']) for c in picks], flush=True)

    # Fixed matched-sentence illustration discovered from the current runs.
    semantic = [case_info('crema_d', f'data/crema_d/AudioWAV/1002_DFA_{emo}_XX.wav')
                for emo in ['ANG','DIS','FEA','HAP']]
    for c in semantic:
        assert c['predictions']['gemini-3.8-flash']['label'] == 'happy'
        assert c['predictions']['gpt_realtime']['label'] == 'neutral'
        assert c['listener_majority'] == c['reference']
        make_audio(c)
    manifest = {
        'date': '2026-11-16', 'scope': 'Internal research inspection copy; source audio terms still apply',
        'analysis_sha256': sha(analysis_path), 'model_order': MODELS,
        'selection_rule': 'Two illustrative extremes per corpus: maximize SER-minus-LLM correct count, then LLM-minus-SER correct count with a different reference emotion and speaker; 1.5–10 s, transcript at most 180 characters, deterministic SHA-256 tie break. Not a representative sample.',
        'audio_processing': 'Full clip, mono 16 kHz MP3, 64 kbit/s. No trimming or loudness normalization. Waveforms share an absolute amplitude scale, with separate time axes.',
        'datasets': datasets, 'same_sentence': semantic,
        'same_sentence_other_takes': [case_info('crema_d', f'data/crema_d/AudioWAV/1002_DFA_{e}_XX.wav') for e in ['NEU','SAD']],
        'sources': sources,
    }
    (ASSETS/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(f'Saved 32 inspection clips + 4 matched-sentence clips: {ASSETS}', flush=True)


if __name__ == '__main__':
    if '--semantic-only' in sys.argv:
        for directory in [BUILD, ASSETS/'audio', ASSETS/'waveforms']:
            directory.mkdir(parents=True, exist_ok=True)
        (BUILD/'.gitignore').write_text('*\n')
        models = ['gemini-3.8-flash', 'gpt_realtime', 'emotion2vec_plus_large']
        rows = {m: {r['audio_path']: r for r in csv_rows(ROOT/'outputs'/m/'crema_d_predictions.csv')}
                for m in models}
        votes = crema_votes()
        cases = []
        for code, reference in [('ANG','angry'),('DIS','disgusted'),('FEA','fearful'),('HAP','happy')]:
            stem = f'1002_DFA_{code}_XX'
            path = f'data/crema_d/AudioWAV/{stem}.wav'
            c = {'dataset_key': 'crema_d', 'clip_id': stem, 'audio_path': path,
                 'reference': reference,
                 'listener_votes': {CREMA_CODE[k]: v for k,v in votes[stem].items() if k in CREMA_CODE and v},
                 'predictions': {m: {'label': rows[m][path]['prediction']} for m in models}}
            assert all(rows[m][path]['acted_emotion'] == reference for m in models)
            assert rows[models[0]][path]['prediction'] == 'happy'
            assert rows[models[1]][path]['prediction'] == 'neutral'
            assert rows[models[2]][path]['prediction'] == reference
            make_audio(c)
            cases.append(c)
        (ASSETS/'semantic.json').write_text(json.dumps({'same_sentence': cases}, indent=2)+'\n')
        print('Prepared four matched-sentence examples', flush=True)
    else:
        main()
