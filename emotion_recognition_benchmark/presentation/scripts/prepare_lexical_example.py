"""Read the matched JL-Corpus lexical/prosodic conflict from saved predictions."""
import collections
import csv
import hashlib
import json
import subprocess
from pathlib import Path

DECK=Path(__file__).resolve().parents[1]
ROOT=DECK.parent
OUT=DECK/'assets/inspections/lexical'
MODELS=['gemini-3.8-flash','gpt_realtime','emotion2vec_plus_large']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows,metadata,sources={},{},[]
    for model in MODELS:
        csv_path=ROOT/'outputs'/model/'jl_corpus_predictions.csv'
        json_path=csv_path.with_suffix('.json')
        with csv_path.open(encoding='utf-8-sig',newline='') as stream:
            rows[model]={r['audio_path']:r for r in csv.DictReader(stream)}
        metadata[model]=json.loads(json_path.read_text())
        if model in MODELS[:2]:
            assert metadata[model]['inference_protocol']['label_scope']=='dataset'
        for p in [csv_path,json_path]:sources.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
    base=ROOT/'data/jl_corpus/audio/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)'
    cases=[]
    sentence='Work hard or you lose.'
    for emotion in ['angry','neutral','happy','sad','excited']:
        clip_id=f'male2_{emotion}_12b_1'
        source=base/(clip_id+'.wav')
        transcript=source.with_suffix('.txt')
        assert transcript.read_text().strip()==sentence
        key=str(source.relative_to(ROOT))
        predictions={model:{'label':rows[model][key]['prediction'],
                            'correct':rows[model][key]['correct'].lower() in ('true','yes','1')}
                     for model in MODELS}
        assert all(rows[model][key]['acted_emotion']==emotion for model in MODELS)
        assert predictions[MODELS[0]]['label']==predictions[MODELS[1]]['label']=='angry'
        assert predictions[MODELS[2]]['label']==(emotion if emotion!='excited' else 'angry')
        target=OUT/(clip_id+'.mp3')
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(source),'-ac','1','-ar','16000',
                        '-c:a','libmp3lame','-b:a','64k','-map_metadata','-1',str(target)],check=True)
        duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration',
                     '-of','default=noprint_wrappers=1:nokey=1',str(source)]))
        cases.append({'clip_id':clip_id,'audio_path':key,'reference':emotion,'text':sentence,
                      'duration_s':duration,'source_audio_sha256':sha(source),
                      'source_transcript_sha256':sha(transcript),'audio_asset':str(target.relative_to(DECK)),
                      'predictions':predictions})
    sentence_rows=[]
    for key,row in rows[MODELS[0]].items():
        transcript=(ROOT/key).with_suffix('.txt')
        if transcript.read_text().strip()==sentence:
            sentence_rows.append(key)
    both_angry=sum(all(rows[m][key]['prediction']=='angry' for m in MODELS[:2]) for key in sentence_rows)
    assert len(sentence_rows)==80 and both_angry==33
    result={'date':'2026-11-16','dataset':'JL-Corpus','sentence':sentence,'speaker':'male2',
            'session':'b','repetition':1,'cases':cases,'accept':metadata[MODELS[0]]['accept'],
            'ground_truth':'Acted emotion encoded in the filename. No per-clip listener labels are published for these recordings.',
            'selection':'All five evaluated emotions for speaker male2, sentence 12, session b, repetition 1. Selected after inspecting saved model predictions. This is not a representative sample or a causal test.',
            'sentence_context':{'n':len(sentence_rows),'both_apis_angry':both_angry,
                                'gemini_labels':dict(collections.Counter(rows[MODELS[0]][key]['prediction'] for key in sentence_rows)),
                                'gpt_labels':dict(collections.Counter(rows[MODELS[1]][key]['prediction'] for key in sentence_rows))},
            'audio_processing':'Full recording, mono 16 kHz, 64 kbit/s MP3. No trimming or loudness normalization.',
            'inference_protocols':{m:metadata[m].get('inference_protocol') for m in MODELS},'sources':sources}
    (OUT/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'sentence':sentence,'clips':[x['clip_id'] for x in cases],
                      'both_apis_angry':both_angry,'sentence_recordings':len(sentence_rows)},indent=2))


if __name__=='__main__':main()
