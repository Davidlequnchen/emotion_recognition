"""Extend the retained CREMA-D disputed-label native chart to ten models."""
import argparse
import copy
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import openpyxl
from refresh_main_deck import Editor,NS,EMU
from merge_inspections import DECK,BUILD,A,R,resolve,relfile,xml_bytes

C='http://schemas.openxmlformats.org/drawingml/2006/chart'
ASSETS=DECK/'assets/listener_plot_refresh'
MODELS=['emotion2vec_plus_large','meralion_ser_v1','melati_emotion','vox_profile_whisper','vox_profile_wavlm','qwen2_5_omni_7b','qwen2_audio_7b','gemma4_e4b','gpt_realtime','gemini-3.8-flash']
NAMES=['emotion2vec+ large','MERaLiON-SER-v1','Melati','Vox-Profile Whisper','Vox-Profile WavLM','Qwen2.5-Omni 7B','Qwen2-Audio 7B Instruct','Gemma 4 E4B','GPT Realtime','Gemini 3.8 Flash']
LABELS=['e2v+\nlarge','MERaLiON\nSER-v1','Melati','Vox\nWhisper','Vox\nWavLM','Qwen2.5\nOmni','Qwen2\nAudio','Gemma 4\nE4B','GPT\nRealtime','Gemini 3.8\nFlash']
SERIES=['the acted emotion',"the listeners' majority",'another emotion']
TITLE='Ten Models: Actor vs Listener Labels (CREMA-D)'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=DECK/'SER_evaluation.pptx');args=parser.parse_args()
    er=DECK.parent;repo=er.parent
    paths=[er/'outputs'/m/'crema_d_predictions.csv' for m in MODELS]
    predictions=[{Path(r['audio_path']).stem:r for r in read_csv(p)} for p in paths]
    baseline=predictions[0];assert len(baseline)==7442
    voice_path=er/'data/crema_d/processedResults/tabulatedVotes.csv'
    voice={r['fileName']:r for r in read_csv(voice_path) if r[''].startswith('1')}
    codes={'A':'angry','D':'disgusted','F':'fearful','H':'happy','N':'neutral','S':'sad'}
    for stem,r in baseline.items():
        votes=voice[stem];maximum=max(int(votes[k]) for k in codes)
        heard=' or '.join(sorted(codes[k] for k in codes if int(votes[k])==maximum))
        assert heard==r['listener_vote'],stem
    cohort=[stem for stem,r in baseline.items() if ' or ' not in r['listener_vote'] and r['listener_vote']!=r['acted_emotion']]
    assert len(cohort)==3699
    models=[]
    for key,name,label,recs in zip(MODELS,NAMES,LABELS,predictions):
        assert set(recs)==set(baseline),(key,'different source clips')
        for stem,r in recs.items():
            assert (r['acted_emotion'],r['listener_vote'])==(baseline[stem]['acted_emotion'],baseline[stem]['listener_vote'])
        counts=Counter('actor' if recs[stem]['prediction']==baseline[stem]['acted_emotion'] else 'listener' if recs[stem]['prediction']==baseline[stem]['listener_vote'] else 'other' for stem in cohort)
        counts={k:counts[k] for k in ['actor','listener','other']};assert sum(counts.values())==3699
        models.append({'key':key,'name':name,'axis_label':label,'counts':counts,'percentages':{k:100*v/3699 for k,v in counts.items()}})
    listener_counts=Counter(baseline[stem]['listener_vote'] for stem in cohort)
    sources=[{'path':str(p.relative_to(repo)),'sha256':digest(p)} for p in [*paths,voice_path]]
    data={'title':TITLE,'cohort_definition':'Same original CREMA-D cohort: one unambiguous voice-only listener plurality label, different from acted emotion. Ties excluded. All three categories are mutually exclusive. Invalid model answers count in another emotion.','full_cohort_count':7442,'disputed_cohort_count':3699,'cohort_ids':cohort,'cohort_sha256':hashlib.sha256('\n'.join(cohort).encode()).hexdigest(),'listener_label_counts':dict(listener_counts),'series':SERIES,'colors':['1B2A41','D55E00','C9CED6'],'models':models,'sources':sources,'template_sha256':digest(args.source)}
    ASSETS.mkdir(parents=True,exist_ok=True);(ASSETS/'data.json').write_text(json.dumps(data,indent=2)+'\n')
    editor=Editor(args.source)
    entry=next(e for e in editor.entries if editor.heading(e) in ['Selected SER Models: Actor vs Listener Labels (CREMA-D)',TITLE])
    owner=editor.part(entry);slide=editor.slide(entry)
    rels=E.fromstring(editor.parts[relfile(owner)])
    chart_path=resolve(owner,next(r for r in rels if r.get('Type')==R+'/chart').get('Target'))
    chart=E.fromstring(editor.parts[chart_path]);cns={'c':C,'a':A}
    for index,ser in enumerate(chart.findall('.//c:barChart/c:ser',cns)):
        # The two ~1% actor segments cannot hold a white whole-number label.
        # Keep their exact values in the native chart/workbook and speaker notes.
        labels=ser.find('c:dLbls',cns)
        for old in labels.findall('c:dLbl',cns):labels.remove(old)
        metric=['actor','listener','other'][index]
        for i,model in enumerate(models):
            if model['percentages'][metric]<3:
                hidden=E.Element(f'{{{C}}}dLbl');E.SubElement(hidden,f'{{{C}}}idx',val=str(i));E.SubElement(hidden,f'{{{C}}}delete',val='1');labels.insert(0,hidden)
        category=ser.find('c:cat/c:strRef',cns);category.find('c:f',cns).text='Sheet1!$A$2:$A$11'
        cache=category.find('c:strCache',cns);cache[:]=[];E.SubElement(cache,f'{{{C}}}ptCount',val='10')
        for i,label in enumerate(LABELS):
            point=E.SubElement(cache,f'{{{C}}}pt',idx=str(i));E.SubElement(point,f'{{{C}}}v').text=label
        values=ser.find('c:val/c:numRef',cns);values.find('c:f',cns).text=f'Sheet1!${chr(66+index)}$2:${chr(66+index)}$11'
        cache=values.find('c:numCache',cns);cache[:]=[];E.SubElement(cache,f'{{{C}}}formatCode').text='General';E.SubElement(cache,f'{{{C}}}ptCount',val='10')
        metric=['actor','listener','other'][index]
        for i,model in enumerate(models):
            point=E.SubElement(cache,f'{{{C}}}pt',idx=str(i));E.SubElement(point,f'{{{C}}}v').text=str(model['percentages'][metric])
    editor.parts[chart_path]=xml_bytes(chart)
    chart_rels=E.fromstring(editor.parts[relfile(chart_path)])
    workbook=resolve(chart_path,next(r for r in chart_rels if r.get('Type')==R+'/package').get('Target'))
    wb=openpyxl.load_workbook(io.BytesIO(editor.parts[workbook]));ws=wb['Sheet1'];ws.delete_rows(2,ws.max_row)
    for row,model in enumerate(models,2):
        ws.cell(row,1,model['axis_label'])
        for col,metric in enumerate(['actor','listener','other'],2):ws.cell(row,col,model['percentages'][metric])
    out=io.BytesIO();wb.save(out);editor.parts[workbook]=out.getvalue()
    # XLSX writers may serialize binary floats with a different last digit.
    # The chart gate compares exact worksheet decimals, so copy their literal
    # serialized representations into the chart cache after saving the workbook.
    sns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(io.BytesIO(out.getvalue())) as package:
        sheet=E.fromstring(package.read('xl/worksheets/sheet1.xml'))
    numeric={cell.get('r'):cell.find('s:v',sns).text for cell in sheet.findall('.//s:c',sns) if cell.find('s:v',sns) is not None}
    for col,ser in enumerate(chart.findall('.//c:barChart/c:ser',cns),2):
        for row,value in enumerate(ser.findall('c:val/c:numRef/c:numCache/c:pt/c:v',cns),2):
            value.text=numeric[f'{chr(64+col)}{row}']
    editor.parts[chart_path]=xml_bytes(chart)
    data['chart_numeric_strings']=[[numeric[f'{col}{row}'] for row in range(2,12)] for col in 'BCD']
    (ASSETS/'data.json').write_text(json.dumps(data,indent=2)+'\n')
    shapes=slide.findall('p:cSld/p:spTree/p:sp',NS)
    title=next(s for s in shapes if any(t.text and 'Actor vs Listener Labels' in t.text for t in s.findall('.//a:t',NS)))
    title.find('.//a:t',NS).text=TITLE
    summary=next(s for s in shapes if any(t.text and t.text.startswith(('Disputed clips:','The same 3,699 clips:')) for t in s.findall('.//a:t',NS)))
    summary.find('.//a:t',NS).text='The same 3,699 clips: a single listener-majority emotion differs from the acted label. Each bar shows what the model predicts.'
    summary.find('p:spPr/a:xfrm/a:ext',NS).set('cx',str(round(785*EMU)))
    frame=slide.find('p:cSld/p:spTree/p:graphicFrame',NS);frame.find('p:xfrm/a:ext',NS).set('cx',str(round(785*EMU)))
    right=next(s for s in shapes if any(t.text and ('Models trained on CREMA' in t.text or t.text=='Acted-label alignment') for t in s.findall('.//a:t',NS)))
    xfrm=right.find('p:spPr/a:xfrm',NS);xfrm.find('a:off',NS).set('x',str(round(928*EMU)));xfrm.find('a:ext',NS).set('cx',str(round(235*EMU)))
    # Reuse the original heading/subordinate bullet styles and point sizes.
    ps=right.findall('p:txBody/a:p',NS)
    strings=['Acted-label alignment','emotion2vec+ 82%, MERaLiON 77% and Qwen2-Audio 77% follow the actor.','Listener alignment','Omni 73%, Gemma 76%, GPT 77% and Gemini 52% match the listener majority.','Reference matters','77% of disputed listener labels are neutral. Matching them alone does not establish better perception.']
    assert len(ps)==len(strings)
    for para,value in zip(ps,strings):
        ts=para.findall('.//a:t',NS);assert len(ts)==1;ts[0].text=value
    slide.find('p:cSld',NS).set('name','ser-refresh:listener-plot:crema:whitespace90')
    editor.parts[owner]=xml_bytes(slide)
    notes=("This chart extends the earlier five-SER chart to all ten evaluated systems, using exactly the same 3,699 disputed clips and original three-way categorization. \n"+data['cohort_definition']+"\nFull model cohort: 7,442. Every prediction CSV has the same IDs, acted labels and listener labels. Voice-only labels were independently reconstructed from tabulatedVotes.csv for all 7,442 clips. A clear majority here means a unique modal category, not necessarily more than half of the votes.\nThe bar categories are the acted emotion, the unique listener plurality, or any other answer. Since the acted and listener labels differ, these categories cannot overlap. Percentages use the exact integer counts, rather than the earlier four-decimal summary. The y-axis still spans 0 to 100 and labels round to whole percentages, as in the original.\nDo not interpret these proportions as a causal effect of training or as class-balanced listener recognition. Of the 3,699 disputed clips, 2,857 (77.2%) have neutral as their listener majority. This imbalance can reward a neutral prediction.\n"+'\n'.join(m['name']+': '+json.dumps(m['counts'])+'; percentages '+json.dumps(m['percentages']) for m in models)+'\nSources and SHA256:\n'+'\n'.join(s['path']+' '+s['sha256'] for s in sources))
    note_rel=next(r for r in rels if r.get('Type')==R+'/notesSlide')
    note_path=resolve(owner,note_rel.get('Target'));note=E.fromstring(editor.parts[note_path])
    body=next(sp.find('p:txBody',NS) for sp in note.findall('p:cSld/p:spTree/p:sp',NS) if (ph:=sp.find('p:nvSpPr/p:nvPr/p:ph',NS)) is not None and ph.get('type')=='body')
    for old in body.findall('a:p',NS):body.remove(old)
    para=E.SubElement(body,f'{{{A}}}p');run=E.SubElement(para,f'{{{A}}}r');E.SubElement(run,f'{{{A}}}t').text=notes+'\nWhite chart labels are omitted only for the two actor segments below 3%; their values remain in the editable chart and workbook.'
    editor.parts[note_path]=xml_bytes(note)
    editor.entries=[entry];destination=BUILD/'listener-plot-refresh.pptx';editor.save(destination)
    (BUILD/'listener-plot-refresh-order.json').write_text(json.dumps([{'kind':'listener-plot','key':'crema','title':TITLE,'already_compact':True}],indent=2)+'\n')
    print(json.dumps({'pptx':str(destination),'disputed_clips':3699,'models':len(models),'source_hashes':len(sources)},indent=2))


if __name__=='__main__':main()
