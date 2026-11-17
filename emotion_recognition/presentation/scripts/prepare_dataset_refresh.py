"""Prepare the 16-dataset presentation inventory and four traceable listening clips each.

Read-only with respect to datasets and benchmark outputs. No model inference is
performed. Old six-corpus audio is reused; newly selected audio keeps the full
recording and is converted to 16 kHz mono MP3 without loudness normalization.
"""
import collections
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from common import CREMA_CODE, CREMA_SENT, RAVDESS_SENT, MCAE_COLS, crema_votes, mcae_info

DECK = Path(__file__).resolve().parents[1]
ROOT = DECK.parent
ASSETS = DECK / 'assets/dataset_refresh'
_source_tree=ast.parse((ROOT/'scripts/common/eval_datasets.py').read_text())
_source_node=next(n for n in _source_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCES' for t in n.targets))
SOURCES=eval(compile(ast.Expression(_source_node.value),'dataset-source-metadata','eval'),{'__builtins__':{},'CC':'https://creativecommons.org/licenses/'})

MODELS = ['emotion2vec_plus_large', 'meralion_ser_v1', 'melati_emotion',
          'vox_profile_whisper', 'vox_profile_wavlm', 'qwen2_5_omni_7b',
          'qwen2_audio_7b', 'gemma4_e4b', 'gpt_realtime', 'gemini-3.8-flash']
ORDER = ['crema_d','mcae_spps','mess','ravdess','tess','jl_corpus','mead','meld',
         'emovo','emozionalmente','savee','thai_ser','nemo','emotiontalk','dailytalk','multidialog']
OVERVIEW_ORDER = ['meld','emotiontalk','dailytalk','multidialog','crema_d','ravdess','savee','jl_corpus',
                  'tess','mead','mess','mcae_spps','thai_ser','nemo','emovo','emozionalmente']
SHORT = dict(zip(MODELS, ['e2v+','MERaLiON','Melati','Vox Whisper','Vox WavLM','Omni','Qwen-Audio','Gemma','GPT','Gemini']))

# Counts describe the current evaluation cohort. Full release counts are noted
# separately where they differ, rather than mixed into the clip totals.
FACTS = {
 'crema_d': dict(name='CREMA-D', language='US English', type='standalone', speakers='91 actors',
   parallel_text='12 sentences', parallel_detail='12 shared sentences acted in six emotions; the retained cohort is not a complete Cartesian grid.',
   label_summary='6 classes; actor + ≈9 voice raters/clip', label_detail='Acted intention is the benchmark reference. Voice-only crowd votes are an additional reference, approximately nine ratings per clip.',
   emotion_summary='6: angry, disgusted, fearful, happy, neutral, sad', licence='ODbL 1.0'),
 'mcae_spps': dict(name='MCAE-SPPS', language='Mandarin', type='standalone', speakers='6 actors',
   parallel_text='200 sentences', parallel_detail='200 shared sentence designs across seven acted emotions and six actors. Listener screening retains 6,675 recordings.',
   label_summary='7 classes; actor + 40 listeners/clip', label_detail='Acted intention plus 40 native listeners per clip. Retained recordings have the intended emotion as the top listener choice.',
   emotion_summary='All 7 canonical emotions', licence='CC BY-NC 4.0'),
 'mess': dict(name='MESS', language='US English', type='standalone', speakers='6 speakers',
   parallel_text='90 sentence IDs', parallel_detail='90 sentence designs, identified by the V/T/M sentence codes in the 1,800-file public package; speakers do not cover a full 90×4×6 grid.',
   label_summary='4 classes; acted labels only', label_detail='Acted intention is public. Listener ratings are described in the publication but are not part of the public per-clip release. Calm is mapped to neutral.',
   emotion_summary='4: angry, calm→neutral, happy, sad', licence='CC BY 4.0'),
 'ravdess': dict(name='RAVDESS', language='N. American English', type='standalone', speakers='24 actors',
   parallel_text='2 sentences', parallel_detail='Two fixed sentences, two repetitions and normal/strong intensity, except that neutral has one intensity.',
   label_summary='8→7 classes; acted + validation', label_detail='Acted emotion is the reference; calm maps to neutral. The original validation supplies ten ratings per file, but no reusable per-clip listener emotion vote is used here.',
   emotion_summary='8 original classes → 7; calm→neutral', licence='CC BY-NC-SA 4.0'),
 'tess': dict(name='TESS', language='Canadian English', type='standalone', speakers='2 actresses',
   parallel_text='200 carrier phrases', parallel_detail='“Say the word ___” with 200 target words, two actresses and seven emotions. OA/OAF filename prefixes denote the same older actress.',
   label_summary='7 classes; acted labels only', label_detail='Acted intention is the reference. No per-clip listener emotion votes are published. Pleasant surprise maps to surprised.',
   emotion_summary='7; pleasant surprise→surprised', licence='CC BY-NC 4.0'),
 'jl_corpus': dict(name='JL-Corpus', language='New Zealand English', type='standalone', speakers='4 speakers',
   parallel_text='15 evaluated sentences', parallel_detail='15 distinct transcripts in the 1,200-recording five-primary-emotion cohort; the raw 2,400-recording release also contains secondary emotions excluded here.',
   label_summary='5 primary; acted labels', label_detail='Five primary acted emotions are evaluated. Excited accepts happy or surprised. A perception study is reported, but these raw clips have no released per-clip listener gold.',
   emotion_summary='5 primary: angry, excited, happy, neutral, sad', licence='CC0 1.0'),
 'mead': dict(name='MEAD', language='English', type='standalone', speakers='48 actors (Part0)',
   parallel_text='Partly shared scripts', parallel_detail='About 30 sentences per emotional category and 40 neutral sentences; some are shared and some are emotion-related. Sentence numbers alone do not establish identical words across emotions. Current scope: stratified 8,000 of 31,729 valid Part0 WAVs.',
   label_summary='7 mapped classes; actor intent', label_detail='Acted intention comes from emotion folders. Contempt is excluded. Three intensity levels are retained. No per-clip listener labels are published.',
   emotion_summary='7; contempt excluded; all intensities', licence='Research data; no published data licence'),
 'meld': dict(name='MELD (test)', language='English', type='dialogue', speakers='100 role labels (test)',
   parallel_text='No controlled parallel set', parallel_detail='TV-dialogue utterances with varied scripts. The 2,608 evaluated recordings have 100 published speaker/role labels, including “All” and three multi-character labels, not 100 independently verified people.',
   label_summary='7 classes; 3 video/text annotators', label_detail='The emotion label is the majority of three annotators who saw video and transcript. These are performed TV dialogues, not instructed emotional takes, and the labels are not audio-only judgments.',
   emotion_summary='All 7 canonical emotions', licence='TV clips; repository GPL-3.0'),
 'emovo': dict(name='EMOVO', language='Italian', type='standalone', speakers='6 actors',
   parallel_text='14 sentences', parallel_detail='Fourteen fixed sentence IDs across six actors and seven emotions, 588 recordings. Some sentences are nonsense sentences; local WAV filenames, rather than an invented transcript, identify the selected prompt.',
   label_summary='7 classes; actor intent', label_detail='Acted intention is encoded in filenames. The authors validated selected sentences, but per-clip listener votes are not published in the local release.',
   emotion_summary='All 7 canonical emotions', licence='No licence with unofficial audio copy'),
 'emozionalmente': dict(name='Emozionalmente', language='Italian', type='standalone', speakers='431 speakers',
   parallel_text='18 sentences', parallel_detail='Eighteen fixed sentences recorded through a web application by 431 non-professional speakers; not every speaker–sentence–emotion combination is present.',
   label_summary='7 classes; intent + 5 listeners/clip', label_detail='The speaker selects an intended emotion. Five listeners supply perceived-emotion votes per recording; these remain separate from the intended-label benchmark reference.',
   emotion_summary='All 7 canonical emotions; 986 clips/class', licence='CC BY 4.0'),
 'savee': dict(name='SAVEE', language='British English', type='standalone', speakers='4 male speakers',
   parallel_text='3 common + emotion-specific', parallel_detail='Three sentences are shared across all emotions. Other sentences are emotion-specific: 15 per emotional category and 30 neutral sentences. Sentence text is not reconstructed from filenames for the selected clips.',
   label_summary='7 classes; actor intent + validation', label_detail='Acted intention is the reference. Ten evaluators validated recordings in the original study, but no per-clip listener labels are published with this copy.',
   emotion_summary='All 7 canonical emotions', licence='Research only; redistribution restricted'),
 'thai_ser': dict(name='THAI-SER (high agreement)', language='Thai', type='standalone', speakers='200 actors',
   parallel_text='3 scripts + improvised speech', parallel_detail='The retained high-agreement cohort contains three fixed script sentences plus improvised scenario utterances. It is a mixture of scripted and improvised stand-alone target clips.',
   label_summary='4 classes; assigned + crowd majority', label_detail='Assigned actor emotion is the primary reference. Crowd-majority labels and agreement are an additional reference. The cohort retains agreement ≥0.71 and excludes assigned frustrated; high listener agreement does not guarantee agreement with actor intention.',
   emotion_summary='4: angry, happy, neutral, sad', licence='CC BY-SA 4.0'),
 'nemo': dict(name='nEMO', language='Polish', type='standalone', speakers='9 speakers',
   parallel_text='90 sentence IDs', parallel_detail='Current file IDs contain 90 sentence IDs, 1–90. The source README defines file_id as speaker/emotion/sentence ID. Selected takes share that published ID; local prediction CSVs do not carry the Polish transcript.',
   label_summary='6 classes; actor intent', label_detail='Acted intention is the reference. Listeners screened recordings, but no per-clip listener emotion labels are published. Only three of nine participants are professional voice actors.',
   emotion_summary='6; no disgusted class', licence='CC BY-NC-SA 4.0'),
 'emotiontalk': dict(name='EmotionTalk (test)', language='Mandarin', type='dialogue', speakers='4 test actors (19 source)',
   parallel_text='No controlled parallel set', parallel_detail='Dialogue turns rather than a controlled same-sentence emotion grid. The evaluated test cohort has 1,929 clips and speaker IDs 02, 13, 20 and 21; the source release has 19 actors.',
   label_summary='7 classes; audio gold + 5 raters', label_detail='Published audio-perception emotion_result is the benchmark gold; five listener judgments and confidence weights are preserved. Gold is not always a simple majority vote and raters may replay preceding turns.',
   emotion_summary='All 7 canonical emotions', licence='CC BY-NC-SA 4.0'),
 'dailytalk': dict(name='DailyTalk (validation)', language='English', type='dialogue', speakers='2 actors',
   parallel_text='No emotion-parallel script', parallel_detail='Performed DailyDialog-derived conversations. The validation cohort contains 1,197 current utterances. Scripts are not a controlled grid of the same words across emotions.',
   label_summary='7 classes; inherited text categories', label_detail='Public ECSS categories equal the inherited DailyDialog text categories after none/no-emotion normalization. ECSS intensity is additional; these are not independently different audio-perception category labels.',
   emotion_summary='7 mapped classes; none→neutral', licence='Audio CC BY-SA 4.0'),
 'multidialog': dict(name='MultiDialog (gold_both)', language='English', type='dialogue', speakers='6 retained actors (12 source)',
   parallel_text='No controlled parallel set', parallel_detail='Performed TopicalChat conversations with prescribed emotion. Current gold_both plus whole-dialogue QC retains 7,672 utterances from 353 dialogues and six speaker IDs (a,b,c,f,g,i); the two source test splits contain twelve actor IDs.',
   label_summary='7 mapped; prescribed script emotion', label_detail='Labels are prescribed TopicalChat performance emotions. Curious to dive deeper maps to neutral. “Gold” means actor-level screening, not new per-utterance listener consensus. Whole-dialogue segmentation QC is applied.',
   emotion_summary='7; curious→neutral', licence='CC BY-NC 4.0'),
}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def flag(value):return str(value).lower() in ('true','yes','1')


def main():
    (ASSETS/'audio').mkdir(parents=True, exist_ok=True)
    legacy=json.loads((DECK/'assets/manifest.json').read_text())['audio']
    analysis_path=ROOT.parent/'research_paper/output/cross_dataset_analysis.json'
    analysis=json.loads(analysis_path.read_text())
    inventory_path=ROOT.parent/'research_paper/output/audio_inventory.csv'
    inventory={row['audio_path']:row for row in read_csv(inventory_path)}
    metadata_sources=[analysis_path, inventory_path, DECK/'assets/manifest.json',ROOT/'scripts/common/eval_datasets.py']
    dialogue={d:{x['audio_path']:x for x in read_csv(ROOT/'data'/d/'utterances.csv')} for d in ['emotiontalk','dailytalk','multidialog']}
    for d in dialogue:metadata_sources += [ROOT/'data'/d/'utterances.csv',ROOT/'data'/d/'README.md']
    crema=crema_votes();mcae=mcae_info()
    metadata_sources += [ROOT/'data/crema_d/processedResults/tabulatedVotes.csv',ROOT/'data/mcae_spps/Corpus information.xlsx']
    emo_dir=ROOT/'data/emozionalmente/audio/emozionalmente/metadata'
    emo_meta={x['file_name']:x for x in read_csv(emo_dir/'samples.csv')}
    emo_names=dict(zip(['anger','disgust','fear','joy','neutrality','sadness','surprise'],['angry','disgusted','fearful','happy','neutral','sad','surprised']))
    emo_votes=collections.defaultdict(collections.Counter)
    for row in read_csv(emo_dir/'evaluations.csv'):emo_votes[row['file_name']][emo_names[row['emotion_recognized']]]+=1
    metadata_sources += [emo_dir/'samples.csv',emo_dir/'evaluations.csv']
    thai={x['audio_id']:x for x in read_csv(ROOT/'data/thai_ser/labels.csv')}
    meld={f"dia{x['Dialogue_ID']}_utt{x['Utterance_ID']}":x for x in read_csv(ROOT/'data/meld/test_sent_emo.csv')}
    metadata_sources += [ROOT/'data/thai_ser/labels.csv',ROOT/'data/meld/test_sent_emo.csv',ROOT/'data/nemo/README.md']
    predictions_sources=[];datasets=[]
    expected_counts={x['dataset']:x['n'] for x in analysis['dataset_rows']}
    for key in ORDER:
        models={};runs={}
        for model in MODELS:
            path=ROOT/'outputs'/model/(key+'_predictions.csv')
            models[model]={x['audio_path']:x for x in read_csv(path)}
            runs[model]=json.loads(path.with_suffix('.json').read_text())
            predictions_sources += [{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [path,path.with_suffix('.json')]]
        canonical=models[MODELS[0]]
        assert len(canonical)==expected_counts[key]
        for m in MODELS:
            assert set(models[m])==set(canonical)
            assert all(models[m][p]['acted_emotion']==row['acted_emotion'] for p,row in canonical.items())
        run=runs[MODELS[0]]
        def describe(path):
            row=canonical[path];p=Path(path);stem=p.stem
            c={'id':stem,'audio_path':path,'reference':row['acted_emotion'],'speaker':'', 'text':'',
               'listener_votes':{},'listener_majority':row.get('listener_vote',''),
               'duration_s':float(inventory[path]['duration_seconds']),
               'predictions':{m:{'label':models[m][path]['prediction'],'correct':flag(models[m][path]['correct'])} for m in MODELS}}
            if key=='crema_d':
                c.update(speaker=stem.split('_')[0],text=CREMA_SENT[stem.split('_')[1]],
                         listener_votes={CREMA_CODE[k]:v for k,v in crema[stem].items() if k in CREMA_CODE and v})
            elif key=='mcae_spps':
                c.update(speaker=stem[0],text=mcae[stem]['sentence content'].strip(),
                         listener_votes={v:int(mcae[stem][k]) for k,v in MCAE_COLS.items() if mcae[stem][k]})
            elif key=='mess':c['speaker']=stem[1:3]
            elif key=='ravdess':c.update(speaker=stem.split('-')[-1],text=RAVDESS_SENT[stem.split('-')[4]])
            elif key=='tess':c.update(speaker={'OA':'OAF'}.get(stem.split('_')[0],stem.split('_')[0]),text='Say the word '+stem.split('_')[1]+'.')
            elif key=='jl_corpus':c.update(speaker=stem.split('_')[0],text=(ROOT/p).with_suffix('.txt').read_text().strip())
            elif key=='mead':c['speaker']=p.parts[-4]
            elif key=='savee':c['speaker']=stem.split('_')[0]
            elif key=='emovo':c.update(speaker=stem.split('-')[1],sentence_id=stem.split('-')[2])
            elif key=='nemo':c.update(speaker=stem.split('_')[0],sentence_id=stem.rsplit('_',1)[1])
            elif key=='emozionalmente':
                meta=emo_meta[p.name];c.update(speaker=meta['actor'],text=meta['sentence'],listener_votes=dict(emo_votes[p.name]))
            elif key=='thai_ser':
                meta=thai[p.name];c.update(speaker=meta['actor_id'],text=meta['script_sent'],listener_agreement=float(meta['agreement']))
                annotations=json.loads(meta['annotated'])
                if all(isinstance(v,list) and len(v)==1 for v in annotations):
                    c['listener_votes']=dict(collections.Counter(emo_names.get(v[0].lower(),v[0].lower()) for v in annotations))
            elif key=='meld':c.update(speaker=meld[stem]['Speaker'],text=meld[stem]['Utterance'])
            else:
                meta=dialogue[key][path.removeprefix(f'data/{key}/')]
                c.update(speaker=meta['speaker_id'],text=meta['text'])
                if key=='emotiontalk':c['listener_votes']=json.loads(meta['votes_json'])
            c['source_text']=c['text']
            c['text']=re.sub(r'\[/?over/?\]','',c['text']).strip()
            c['source_audio_sha256']=sha(ROOT/path)
            return c
        original=[x for x in legacy if x.get('group')=='dataset_example' and x.get('subgroup')==key]
        if original:
            cases=[]
            for old in original:
                c=describe(old['original_path']);c['audio_asset']='assets/'+old['path'];c['audio_processing']='Legacy presentation clip: trimmed/level-adjusted by scripts/make_audio.py; source WAV unchanged.'
                cases.append(c)
            matching='Same speaker and sentence, four acted emotions'
            assert len(cases)==4
        else:
            candidates=[describe(path) for path in canonical]
            usable=[c for c in candidates if 1.2<=c['duration_s']<=10 and len(c['text'])<=180 and not flag(inventory[c['audio_path']]['empty_audio'])]
            targets=['neutral','angry','happy','sad']
            groups=collections.defaultdict(list)
            if key in ['emovo','emozionalmente','thai_ser','nemo']:
                for c in usable:
                    prompt=c.get('sentence_id') if key in ['emovo','nemo'] else c['text']
                    if prompt:groups[(c['speaker'],prompt)].append(c)
            choices=[]
            for g,cs in groups.items():
                by={e:[c for c in cs if c['reference']==e] for e in targets}
                if all(by.values()):
                    picks=[min(by[e],key=lambda c:c['id']) for e in targets]
                    agreement=sum((c['listener_votes'].get(c['reference'],0)/sum(c['listener_votes'].values())) if c['listener_votes'] else 0 for c in picks)
                    choices.append((agreement,g,picks))
            if choices:
                _,g,cases=max(choices,key=lambda x:(x[0],str(x[1])))
                matching='Same speaker and published sentence ID, four acted emotions' if key in ['emovo','nemo'] else 'Same speaker and sentence, four acted emotions'
            else:
                cases=[]
                for emotion in targets:
                    pool=[c for c in usable if c['reference']==emotion]
                    assert pool,(key,emotion)
                    def rank(c):
                        votes=c['listener_votes'];fraction=votes.get(c['reference'],0)/sum(votes.values()) if votes else 0
                        return (-fraction,hashlib.sha256(c['audio_path'].encode()).hexdigest())
                    cases.append(min(pool,key=rank))
                matching='Four representative current-cohort clips; different words and speakers may occur'
            for c in cases:
                name=key+'__'+hashlib.sha256(c['audio_path'].encode()).hexdigest()[:8]+'__'+c['id']+'.mp3'
                out=ASSETS/'audio'/name
                subprocess.run(['ffmpeg','-v','error','-y','-i',str(ROOT/c['audio_path']),'-ac','1','-ar','16000',
                                '-c:a','libmp3lame','-b:a','64k','-map_metadata','-1',str(out)],check=True)
                c['audio_asset']=str(out.relative_to(DECK));c['audio_processing']='Full recording; 16 kHz mono MP3 at 64 kbit/s; no trimming or loudness normalization.'
        facts=dict(FACTS[key])
        facts.update(key=key,clips=len(canonical),classes=run['classes'],accept=run.get('accept',{}),cases=cases,matching_caption=matching,
                     hours=round(sum(float(inventory[p]['duration_seconds']) for p in canonical)/3600,3))
        facts['training_involvement']=', '.join(SHORT[x['model']] for x in analysis['model_rows'] if x['dataset']==key and x['training_status']!='not_established') or 'Not established'
        facts['source_notes']='\n'.join([FACTS[key]['parallel_detail'],FACTS[key]['label_detail'],
            'Counts describe the exact currently evaluated cohort. Dialogue denotes recording source; model input remains one utterance without prior turns.',
            'Training involvement is the current benchmark provenance status, including upstream lineage/development; it is not a clip-leakage claim.',
            'Source: scripts/common/eval_datasets.py; outputs/<model>/'+key+'_predictions.csv and matching JSON; research_paper/output/audio_inventory.csv.',
            'Corpus source: '+SOURCES[key]['url'], 'Credit: '+SOURCES[key]['credit'],'Terms: '+SOURCES[key]['licence']])
        # Independently check the precise speaker interpretation that is most
        # likely to be mistaken for a full-corpus actor count.
        if key=='meld':assert len({meld[Path(p).stem]['Speaker'] for p in canonical})==100
        if key=='nemo':assert len({Path(p).stem.rsplit('_',1)[1] for p in canonical})==90
        if key=='tess':assert len({{'OA':'OAF'}.get(Path(p).stem.split('_')[0],Path(p).stem.split('_')[0]) for p in canonical})==2
        datasets.append(facts)
        print(key,len(canonical),[c['id'] for c in cases],matching,flush=True)
    assert len(datasets)==16 and sum(d['clips'] for d in datasets)==67062
    result={'date':'2026-11-17','model_order':MODELS,'dataset_order':ORDER,'overview_order':OVERVIEW_ORDER,
            'datasets':datasets,'scope':'Internal research presentation. Source data terms are retained.',
            'source_hashes':predictions_sources+[{'path':str(p.relative_to(ROOT.parent)),'sha256':sha(p)} for p in dict.fromkeys(metadata_sources)]}
    (ASSETS/'data.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Saved 16 dataset records and 64 listening examples:',ASSETS/'data.json',flush=True)


if __name__=='__main__':main()
