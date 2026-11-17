"""Refresh the ten retained listening-error cases from all ten saved runs.

This preserves the existing audio selection and its actor/listener labels. It
does not run models or interpret one-hot LLM export fields as confidence.
"""
import csv
import hashlib
import json
from pathlib import Path

DECK=Path(__file__).resolve().parents[1]
ROOT=DECK.parent
OUT=DECK/'assets/error_refresh'
MODELS=['emotion2vec_plus_large','meralion_ser_v1','melati_emotion','vox_profile_whisper',
        'vox_profile_wavlm','qwen2_5_omni_7b','qwen2_audio_7b','gemma4_e4b',
        'gpt_realtime','gemini-3.8-flash']
NAMES=['emotion2vec+ large','MERaLiON-SER-v1','Melati','Vox-Profile Whisper',
       'Vox-Profile WavLM','Qwen2.5-Omni 7B','Qwen2-Audio 7B Instruct',
       'Gemma 4 E4B','GPT Realtime','Gemini 3.8 Flash']
GROUPS=[
 ('i_models_side_with_listeners','agreement','Error Analysis: Agreement with Listeners'),
 ('ii_confident_on_unseen_data','domain-1','Error Analysis: Domain Shift (1/2)'),
 ('ii_confident_on_unseen_data','domain-2','Error Analysis: Domain Shift (2/2)'),
 ('iii_models_disagree','disagreement','Error Analysis: Model Disagreement'),
 ('iv_questionable_ground_truth','ambiguous','Error Analysis: Ambiguous Ground Truth'),
]
EXPLANATIONS={
 '1068_IEO_HAP_HI':('Listeners vs actor','Five models agree with listener anger; only Vox-Whisper matches acted happy. The scored error depends on whether the reference is intended or perceived emotion.'),
 '1066_ITS_ANG_XX':('Listener agreement','Eight of ten listeners hear disgust. Three models select disgusted and six select neutral; none matches the acted angry label. Listener agreement changes the interpretation of these errors.'),
 'AM2V15_SCR':('A shared failure','Nine models predict neutral despite the acted angry label. Qwen2-Audio alone predicts angry. A high dedicated-SER score does not establish correctness on this corpus.'),
 '57175':('Surprise vs anger','Thirty-eight of forty listeners hear surprise, but only Vox-Whisper predicts surprised. Five models predict angry. This is shared category confusion with strong listener agreement.'),
 'YAF_read_neutral':('Two outliers','Eight models match the neutral reference. MERaLiON and Gemini predict sad. Agreement elsewhere in the panel helps localize the disagreement to those two systems.'),
 '03-01-08-02-02-01-17':('Model-specific errors','All five audio LLMs and three dedicated SER systems predict surprised. MERaLiON predicts fearful and Melati angry, so this failure is model-specific rather than universal.'),
 'SF3T02_SCR':('Family disagreement','All five audio LLMs and emotion2vec+ predict neutral. Four other dedicated SER systems predict sad. Model-family agreement can still disagree sharply with the acted label.'),
 '62191':('A perceptual split','Listeners split mainly between happy and neutral. MERaLiON and Melati choose happy; six models choose neutral. Model disagreement mirrors some, but not all, of the human ambiguity.'),
 '1006_IEO_ANG_MD':('A three-way tie','Listeners split equally across angry, happy and neutral. The ten predictions span those same categories, so scoring against angry alone conceals the three-way perceptual tie.'),
 '1030_WSI_FEA_XX':('No listener chose fear','No listener chooses the acted fearful label. MERaLiON and Qwen2-Audio match the actor, while the other predictions use categories heard by at least one listener.'),
}


def sha(file):return hashlib.sha256(file.read_bytes()).hexdigest()


def main():
    manifest_path=DECK/'assets/manifest.json'
    historical=json.loads(manifest_path.read_text())
    clips=[x for x in historical['audio'] if x['group']=='failure']
    assert len(clips)==10
    analysis_path=ROOT.parent/'research_paper/output/cross_dataset_analysis.json'
    analysis=json.loads(analysis_path.read_text())
    selected={(x['model'],x['dataset']):x for x in analysis['manifest']['selection'] if x['status']=='selected'}
    datasets={x['original_path'].split('/')[1] for x in clips}
    loaded={}
    sources=[{'path':str(manifest_path.relative_to(ROOT)),'sha256':sha(manifest_path)},
             {'path':str(analysis_path.relative_to(ROOT.parent)),'sha256':sha(analysis_path),'root':'repository'}]
    for ds in sorted(datasets):
        for model in MODELS:
            json_path=ROOT/selected[model,ds]['path']
            csv_path=json_path.with_suffix('.csv')
            run=json.loads(json_path.read_text())
            with csv_path.open(encoding='utf-8-sig',newline='') as fh:
                rows={x['audio_path']:x for x in csv.DictReader(fh)}
            assert len(rows)==run['clips']==selected[model,ds]['clips']
            if model in MODELS[5:]:assert run['inference_protocol']['label_scope']=='dataset'
            loaded[model,ds]=(rows,run)
            for file in [json_path,csv_path]:sources.append({'path':str(file.relative_to(ROOT)),'sha256':sha(file)})
    current=[]
    for old in clips:
        ds=old['original_path'].split('/')[1]
        clip_id=Path(old['original_path']).stem
        predictions={}
        for i,model in enumerate(MODELS):
            rows,run=loaded[model,ds]
            row=rows[old['original_path']]
            assert row['acted_emotion']==old['acted_label']
            score=float(row['p_'+row['prediction']]) if i<5 else None
            predictions[model]={'label':row['prediction'],
                                'correct':row['correct'].lower() in ('true','yes','1'),
                                'ser_score':score}
        audio=DECK/'assets'/old['path']
        assert audio.is_file()
        heading,explanation=EXPLANATIONS[clip_id]
        current.append({'id':clip_id,'dataset_key':ds,'dataset':old['dataset'],
                        'speaker':old['speaker'],'text':old['sentence'] or '',
                        'reference':old['acted_label'],'audio_path':old['original_path'],
                        'audio_asset':'assets/'+old['path'],'duration_s':old['duration_s'],
                        'audio_sha256':sha(audio),'listener_votes':old['listener_votes'] or {},
                        'listener_majority':old['listener_majority'] or '',
                        'listener_info':old['listener_info'],'subgroup':old['subgroup'],
                        'insight_heading':heading,'insight':explanation,'predictions':predictions})
    seen={}
    slides=[]
    for subgroup,key,title in GROUPS:
        cases=[x for x in current if x['subgroup']==subgroup]
        offset=seen.get(subgroup,0);seen[subgroup]=offset+2
        chosen=cases[offset:offset+2]
        assert len(chosen)==2
        slides.append({'kind':'error-analysis','key':key,'title':title,'cases':chosen})
    result={'model_order':MODELS,'model_names':dict(zip(MODELS,NAMES)),'slides':slides,
            'source_artifacts':sources,'prediction_cells':100,
            'audio_processing':'Existing presentation MP3s are retained byte-for-byte, including their earlier trimming and loudness processing. Predictions were generated from the original audio paths in saved runs.',
            'interpretation':'This is the retained illustrative selection, not a representative sample. Green matches the acted reference, blue matches the listener majority/tie, and red matches neither. Dedicated-SER scores are shown only for the five dedicated models and are not calibrated cross-model probabilities. LLM export one-hot fields are not confidence scores.',
            'provenance_note':'Corpus/teacher/development involvement follows docs/training_provenance.md. These examples do not establish exact clip leakage, prove a cause of domain shift, or certify unseen corpora.'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'data.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Prepared ten retained error cases, 100 current model labels and source hashes')


if __name__=='__main__':main()
