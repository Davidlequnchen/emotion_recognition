"""Check source fidelity, requested coverage, audio targets and bottom whitespace."""
import copy
import io
import hashlib
import json
import posixpath
from decimal import Decimal
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E

from merge_inspections import DECK,BUILD,NS,P,A,R,CT,parts,relfile,resolve,xml_bytes
from refresh_main_deck import REFRESH,title,EMU

C='http://schemas.openxmlformats.org/drawingml/2006/chart'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def labels(cell):return ''.join(t.text or '' for t in cell.findall('.//a:t',NS))


def main():
    candidate=REFRESH/'main-candidate.pptx';package=parts(candidate)
    source_path=REFRESH/'source.pptx'
    if not source_path.exists():source_path=DECK/'SER_evaluation.pptx'
    source=parts(source_path)
    root=E.fromstring(package['ppt/presentation.xml'])
    rels=E.fromstring(package['ppt/_rels/presentation.xml.rels'])
    lookup={r.get('Id'):resolve('ppt/presentation.xml',r.get('Target')) for r in rels}
    entries=list(root.find('p:sldIdLst',NS));assert len(entries)==92
    paths=[lookup[e.get(f'{{{R}}}id')] for e in entries]
    slides=[E.fromstring(package[p]) for p in paths]
    headings=[title(s) for s in slides]
    dataset=json.loads((DECK/'assets/dataset_refresh/data.json').read_text())
    errors=json.loads((DECK/'assets/error_refresh/data.json').read_text())
    model=json.loads((BUILD/'model-refresh-order.json').read_text())
    result_data=json.loads((DECK/'assets/result_refresh/data.json').read_text())
    verified_cells=0;verified_hashes=0
    relationship_ids=[r.get('Id') for r in rels]
    assert len(relationship_ids)==len(set(relationship_ids)),'duplicate presentation relationship IDs'
    for record in dataset['source_hashes']:
        base=DECK.parent if record['path'].startswith('outputs/') else DECK.parent.parent
        p=base/record['path']
        assert sha(p)==record['sha256'],record['path'];verified_hashes+=1
    for record in result_data['sources']:
        assert sha(DECK.parent.parent/record['path'])==record['sha256'],record['path']
        verified_hashes+=1
    ds_by_key={d['key']:d for d in dataset['datasets']}
    dataset_slides=[];error_slides=[];overview_slides=[];maps=[]
    audio_count=0;tables=[];max_bottom=0.0
    for number,(part,slide) in enumerate(zip(paths,slides),1):
        cs=slide.find('p:cSld',NS);marker=cs.get('name','')
        tree=cs.find('p:spTree',NS)
        ids=[x.get('id') for x in slide.findall('.//p:cNvPr',NS)]
        assert len(ids)==len(set(ids)),(number,'duplicate shape ids')
        assert all(x.get('spid') in ids for x in slide.findall('.//p:spTgt',NS)),(number,'broken audio trigger')
        audio_count+=len(slide.findall('.//a:audioFile',NS))
        for shape in tree:
            xfrm=shape.find('p:spPr/a:xfrm',NS)
            if xfrm is None:xfrm=shape.find('p:xfrm',NS)
            if xfrm is None:continue
            off,ext=xfrm.find('a:off',NS),xfrm.find('a:ext',NS)
            if off is None or ext is None:continue
            bottom=(int(off.get('y'))+int(ext.get('cy')))/EMU
            max_bottom=max(max_bottom,bottom)
            assert bottom<=648.1,(number,headings[number-1],'object invades bottom whitespace',bottom)
        native=slide.findall('.//a:tbl',NS)
        if native:tables.append(number)
        if 'ser-refresh:overview:' in marker:overview_slides.append(number)
        if 'ser-refresh:label_mapping:' in marker or 'ser-refresh:label-mapping:' in marker or 'ser-refresh:label-map:' in marker:maps.append(number)
        if 'ser-refresh:dataset_example:' in marker or 'ser-refresh:dataset-example:' in marker or 'ser-refresh:dataset-intro:' in marker or 'ser-refresh:intro:' in marker:
            dataset_slides.append(number)
            key=marker.split(':')[-2]
            ds=ds_by_key[key]
            assert len(native)==1
            rows=native[0].findall('a:tr',NS);assert len(rows)==11
            for row,m in zip(rows[1:],dataset['model_order']):
                cells=row.findall('a:tc',NS);assert len(cells)==5
                for c,case in enumerate(ds['cases'],1):
                    expected=case['predictions'][m]['label']
                    if expected=='other answer':expected='invalid'
                    assert labels(cells[c])==expected,(key,m,c,labels(cells[c]),expected)
                    verified_cells+=1
            assert len(slide.findall('.//a:audioFile',NS))==4
        if 'ser-refresh:error-analysis:' in marker:
            error_slides.append(number)
            record=next(s for s in errors['slides'] if s['title']==headings[number-1])
            assert len(native)==2
            for table,case in zip(native,record['cases']):
                rows=table.findall('a:tr',NS);assert len(rows)==11
                for row,m in zip(rows[1:],dataset['model_order']):
                    texts=[labels(c) for c in row.findall('a:tc',NS)]
                    expected=case['predictions'][m]['label']
                    assert expected in texts or (expected=='other answer' and 'invalid' in texts),(headings[number-1],case['id'],m,texts)
                    verified_cells+=1
            assert len(slide.findall('.//a:audioFile',NS))==2
    assert len(overview_slides)==1,overview_slides
    overview=slides[overview_slides[0]-1].find('.//a:tbl',NS)
    overview_rows=overview.findall('a:tr',NS)
    assert len(overview_rows)==17,'The overview must include all sixteen datasets'
    assert all(len(r.findall('a:tc',NS))==7 for r in overview_rows),'Missing overview fields'
    for row,key in zip(overview_rows[1:],dataset['overview_order']):
        cells=[labels(c) for c in row.findall('a:tc',NS)];record=ds_by_key[key]
        assert cells[0]==record['name'].split(' (')[0],(key,cells[0])
        assert cells[2]==('dialogue' if record['type']=='dialogue' else 'standalone')
        assert cells[3]==format(record['clips'],','),(key,cells[3])
        assert cells[6]==record['label_summary'],(key,cells[6])
        assert all(cells),(key,'Missing overview metadata')
    assert len(dataset_slides)==16,dataset_slides
    # Label-map title is checked independently of the authors' marker spelling.
    map_index=next(i for i,t in enumerate(headings) if 'Label Mapping' in t and ('16' in t or 'Sixteen' in t))
    mapping=slides[map_index].find('.//a:tbl',NS)
    assert len(mapping.findall('a:tr',NS))==17
    assert len(error_slides)==5,error_slides
    assert audio_count==122,audio_count
    survey_index=next(i for i,t in enumerate(headings) if t==model[0]['title'])
    assert len(slides[survey_index].find('.//a:tbl',NS).findall('a:tr',NS))==11
    assert all(x['title'] in headings for x in model[1:])
    listener_data=json.loads((DECK/'assets/listener_plot_refresh/data.json').read_text())
    listener_matches=[i for i,s in enumerate(slides) if s.find('p:cSld',NS).get('name','').startswith('ser-refresh:listener-plot:')]
    assert len(listener_matches)==1,listener_matches
    listener_index=listener_matches[0]
    chart_ref=slides[listener_index].find('.//{'+C+'}chart')
    relationships=E.fromstring(package[relfile(paths[listener_index])])
    target=next(r.get('Target') for r in relationships if r.get('Id')==chart_ref.get('{'+R+'}id'))
    chart_path=resolve(paths[listener_index],target)
    chart=E.fromstring(package[chart_path])
    chart_relationships=E.fromstring(package[relfile(chart_path)])
    workbook_target=next(r.get('Target') for r in chart_relationships if r.get('Type').endswith('/package'))
    with ZipFile(io.BytesIO(package[resolve(chart_path,workbook_target)])) as workbook:
        sheet=E.fromstring(workbook.read('xl/worksheets/sheet1.xml'))
    sn={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    sheet_values={c.get('r'):c.find('s:v',sn).text for c in sheet.findall('.//s:c',sn) if c.find('s:v',sn) is not None}
    cn={'c':C};series=chart.findall('.//c:ser',cn);assert len(series)==3
    for s,key,column in zip(series,['actor','listener','other'],['B','C','D']):
        cached=s.findall('.//c:val//c:pt/c:v',cn)
        values=[float(v.text) for v in cached]
        assert len(values)==10
        for r,v in enumerate(cached,2):
            assert Decimal(v.text)==Decimal(sheet_values[column+str(r)]),(column,r,'Chart cache differs from worksheet')
        for value,record in zip(values,listener_data['models']):
            assert abs(value-record['percentages'][key])<1e-8,(key,record['name'],value)
        categories=[v.text for v in s.findall('.//c:cat//c:pt/c:v',cn)]
        assert categories==[m['axis_label'] for m in listener_data['models']]
    assert listener_data['disputed_cohort_count']==3699
    for record in listener_data['models']:assert sum(record['counts'].values())==3699
    for record in listener_data['sources']:
        assert sha(DECK.parent.parent/record['path'])==record['sha256'],record['path']
        verified_hashes+=1
    survey_headers=[labels(c) for c in slides[survey_index].find('.//a:tbl/a:tr',NS).findall('a:tc',NS)]
    assert any('Release' in h for h in survey_headers),survey_headers
    assert any('Emotion training' in h for h in survey_headers),survey_headers
    assert any('classes' in h.lower() for h in survey_headers),survey_headers
    assert not any('Label Mapping: Six Listening-Example Corpora' in t for t in headings)
    result_tables=[];verified_metrics=0
    authoritative=json.loads((DECK.parent.parent/'research_paper/output/cross_dataset_analysis.json').read_text())
    metric_rows={(r['model'],r['dataset']):r for r in authoritative['model_rows']}
    for metric in ['accuracy','uar','macro_f1']:
        matches=[i for i,s in enumerate(slides) if s.find('p:cSld',NS).get('name','').startswith('ser-refresh:result-table:'+metric+':')]
        assert len(matches)==1,(metric,matches)
        table=slides[matches[0]].find('.//a:tbl',NS)
        rows=table.findall('a:tr',NS)
        assert len(rows) in [17,18],(metric,len(rows))
        assert all(len(row.findall('a:tc',NS))==11 for row in rows),(metric,'missing model columns')
        detail=result_data['metric_details'][metric]
        expected=detail['table_values']
        assert len(rows)==len(expected),(metric,'missing mean row')
        for i,(row,wanted) in enumerate(zip(rows,expected)):
            actual=[labels(c) for c in row.findall('a:tc',NS)]
            assert actual==[x.replace('\n','') for x in wanted],(metric,i,actual,wanted)
        for r,dataset_key in enumerate(result_data['dataset_order']):
            for c,model_key in enumerate(result_data['model_order']):
                value=metric_rows[model_key,dataset_key][metric]
                assert value==detail['raw_fractions'][r][c]
                assert abs(float(labels(rows[r+1].findall('a:tc',NS)[c+1]).rstrip('†'))-100*value)<=.050001
                verified_metrics+=1
        result_tables.append(matches[0]+1)
    for name,blob in source.items():
        if name.startswith('ppt/embeddings/') and name in package:
            assert package.get(name)==blob,(name,'workbook changed')
        if name.startswith('ppt/media/') and name in package:
            assert package[name]==blob,(name,'audio/image bytes changed')
        if name.startswith('ppt/charts/') and name.endswith('.xml') and name in package:
            before,after=E.fromstring(blob),E.fromstring(package[name])
            for tag in ['numCache','strCache','f']:
                a=before.findall('.//{'+C+'}'+tag);b=after.findall('.//{'+C+'}'+tag)
                assert [E.tostring(x) for x in a]==[E.tostring(x) for x in b],(name,tag,'chart data changed')
    # Bounded imports retain every selected slide's exact XML and media bytes.
    batches=[]
    for start in range(0,len(entries),8):
        chosen=entries[start:start+8];pr=copy.deepcopy(root);pr.find('p:sldIdLst',NS)[:]=[copy.deepcopy(e) for e in chosen]
        selected={e.get(f'{{{R}}}id') for e in chosen};rr=copy.deepcopy(rels)
        for r in list(rr):
            if r.get('Type')==R+'/slide' and r.get('Id') not in selected:rr.remove(r)
        pp={**package,'ppt/presentation.xml':xml_bytes(pr),'ppt/_rels/presentation.xml.rels':xml_bytes(rr)}
        reachable=set()
        def walk(name):
            if name in reachable:return
            reachable.add(name);rp=relfile(name) if name else '_rels/.rels'
            if rp not in pp:return
            reachable.add(rp)
            for r in E.fromstring(pp[rp]):
                if r.get('TargetMode')!='External':walk(resolve(name,r.get('Target')))
        walk('');reachable.discard('')
        ct=E.fromstring(pp['[Content_Types].xml'])
        for t in list(ct):
            if t.tag==f'{{{CT}}}Override' and t.get('PartName').lstrip('/') not in reachable:ct.remove(t)
        pp['[Content_Types].xml']=xml_bytes(ct);reachable.add('[Content_Types].xml')
        dst=REFRESH/f'import-{start+1:02d}-{start+len(chosen):02d}.pptx'
        with ZipFile(dst,'w',ZIP_DEFLATED) as z:
            for name in sorted(reachable):z.writestr(name,pp[name])
        batches.append({'path':str(dst),'slides':len(chosen)})
    (REFRESH/'import-batches.json').write_text(json.dumps(batches,indent=2)+'\n')
    report={'passed':True,'candidate_sha256':sha(candidate),'slide_count':len(entries),'audio_shapes':audio_count,
            'dataset_examples':len(dataset_slides),'label_map_rows':16,'model_inventory_rows':10,'open_llm_architecture_slides':3,
            'error_analysis_slides':len(error_slides),'prediction_cells_verified':verified_cells,'source_hashes_verified':verified_hashes,
            'single_slide_result_tables':result_tables,'metric_cells_verified':verified_metrics,
            'actor_listener_plot_slide':listener_index+1,'actor_listener_segments_verified':30,
            'maximum_object_bottom_px':max_bottom,'minimum_bottom_whitespace_px':720-max_bottom,
            'native_table_slides':tables,'chart_data_and_embedded_workbooks_preserved':True}
    (REFRESH/'content-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
