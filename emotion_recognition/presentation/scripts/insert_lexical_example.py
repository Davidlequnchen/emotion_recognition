"""Insert the lexical-conflict slide immediately after the first sentence example.

Keep the private 84-slide intermediate as the source, retain all original media/chart
bytes, and validate the 85-slide ordering and the new saved-prediction table.
"""
import copy
import hashlib
import json
import posixpath
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E

from merge_inspections import DECK,BUILD,NS,P,A,R,CT,REL,parts,relfile,resolve,xml_bytes


def main():
    source_path=BUILD/'SER_evaluation_extended-candidate.pptx'
    source=parts(source_path)
    out=dict(source)
    imported=parts(BUILD/'lexical-candidate.pptx')
    types=E.fromstring(out['[Content_Types].xml'])
    imported_types=E.fromstring(imported['[Content_Types].xml'])
    overrides={x.get('PartName').lstrip('/'):x.get('ContentType') for x in imported_types if x.tag.endswith('Override')}
    known={x.get('PartName') for x in types if x.tag.endswith('Override')}
    exts={x.get('Extension') for x in types if x.tag.endswith('Default')}
    for x in imported_types:
        if x.tag.endswith('Default') and x.get('Extension') not in exts:
            types.append(copy.deepcopy(x));exts.add(x.get('Extension'))
    mapping={'ppt/slides/slide1.xml':'ppt/slides/slide85.xml'}
    copied=set()

    def cp(src):
        if src.startswith('ppt/slideLayouts/'):return 'ppt/slideLayouts/slideLayout1.xml'
        if src.startswith('ppt/notesMasters/'):return 'ppt/notesMasters/notesMaster1.xml'
        dst=mapping.setdefault(src,posixpath.join(posixpath.dirname(src),'lexical_'+posixpath.basename(src)))
        if src in copied:return dst
        copied.add(src)
        out[dst]=imported[src]
        if src in overrides and '/'+dst not in known:
            E.SubElement(types,f'{{{CT}}}Override',PartName='/'+dst,ContentType=overrides[src]);known.add('/'+dst)
        rp=relfile(src)
        if rp in imported:
            rels=E.fromstring(imported[rp])
            for rel in rels:
                if rel.get('TargetMode')!='External':
                    target=resolve(src,rel.get('Target'))
                    rel.set('Target',posixpath.relpath(cp(target),posixpath.dirname(dst)))
            out[relfile(dst)]=xml_bytes(rels)
        return dst

    new_slide=cp('ppt/slides/slide1.xml')
    root=E.fromstring(out['ppt/presentation.xml'])
    rels=E.fromstring(out['ppt/_rels/presentation.xml.rels'])
    entries=list(root.find('p:sldIdLst',NS))
    assert len(entries)==84
    lookup={x.get('Id'):resolve('ppt/presentation.xml',x.get('Target')) for x in rels}
    previous=E.fromstring(out[lookup[entries[38].get(f'{{{R}}}id')]])
    assert any('Same Words, Different Emotions' in (t.text or '') for t in previous.findall('.//a:t',NS))
    rid='rIdLexicalConflictExample'
    assert rid not in lookup
    E.SubElement(rels,f'{{{REL}}}Relationship',Id=rid,Type=R+'/slide',Target='slides/slide85.xml')
    new_id=max(int(x.get('id')) for x in entries)+1
    entries.insert(39,E.Element(f'{{{P}}}sldId',{'id':str(new_id),f'{{{R}}}id':rid}))
    root.find('p:sldIdLst',NS)[:]=entries
    lookup[rid]=new_slide
    footer=None
    for sp in previous.findall('p:cSld/p:spTree/p:sp',NS):
        off=sp.find('p:spPr/a:xfrm/a:off',NS)
        texts=sp.findall('.//a:t',NS)
        if off is not None and int(off.get('x'))>10_000_000 and int(off.get('y'))>6_000_000 and len(texts)==1 and (texts[0].text or '').isdigit():
            footer=copy.deepcopy(sp);break
    assert footer is not None
    page=0
    for number,entry in enumerate(entries,1):
        name=lookup[entry.get(f'{{{R}}}id')]
        slide=E.fromstring(out[name])
        texts=slide.findall('.//a:t',NS)
        if number==37:
            old='One CREMA-D actor says the same sentence in different emotions. Compare Gemini 3.8 Flash and GPT Realtime with the acted labels and listener votes.'
            matches=[t for t in texts if t.text==old]
            assert len(matches)==1
            matches[0].text='Same-speaker emotion contrasts in CREMA-D and JL-Corpus. Compare both APIs with the acted labels and available listener votes.'
        if number==1 or any('Thank you' in (t.text or '') for t in texts):continue
        page+=1
        if number<40 and number!=37:continue
        current=None
        for sp in slide.findall('p:cSld/p:spTree/p:sp',NS):
            off=sp.find('p:spPr/a:xfrm/a:off',NS)
            ts=sp.findall('.//a:t',NS)
            if off is not None and int(off.get('x'))>10_000_000 and int(off.get('y'))>6_000_000 and len(ts)==1 and (ts[0].text or '').isdigit():
                current=sp;break
        if current is None:
            current=copy.deepcopy(footer)
            current.find('p:nvSpPr/p:cNvPr',NS).set('id',str(max(int(x.get('id')) for x in slide.findall('.//p:cNvPr',NS))+1))
            slide.find('p:cSld/p:spTree',NS).append(current)
        current.find('.//a:t',NS).text=str(page)
        out[name]=xml_bytes(slide)
    out['ppt/presentation.xml']=xml_bytes(root)
    out['ppt/_rels/presentation.xml.rels']=xml_bytes(rels)
    out['[Content_Types].xml']=xml_bytes(types)
    app=E.fromstring(out['docProps/app.xml'])
    for child in app:
        if E.QName(child).localname=='Slides':child.text='85'
    out['docProps/app.xml']=xml_bytes(app)

    manifest=json.loads((DECK/'assets/inspections/lexical/manifest.json').read_text())
    for record in manifest['sources']:
        assert hashlib.sha256((DECK.parent/record['path']).read_bytes()).hexdigest()==record['sha256']
    new_xml=E.fromstring(out[new_slide])
    rows=new_xml.find('.//a:tbl',NS).findall('a:tr',NS)
    assert len(rows)==6
    for row,case in zip(rows[1:],manifest['cases']):
        cells=row.findall('a:tc',NS)
        for col,model in enumerate(['gemini-3.8-flash','gpt_realtime','emotion2vec_plus_large'],3):
            value=''.join(t.text or '' for t in cells[col].findall('.//a:t',NS))
            assert value==case['predictions'][model]['label']
        assert ''.join(t.text or '' for t in cells[2].findall('.//a:t',NS))==case['clip_id']
    for name,blob in source.items():
        if name.startswith(('ppt/media/','ppt/charts/','ppt/embeddings/')):
            assert out[name]==blob,(name,'source asset changed')
    audio_count=0
    for entry in entries:
        slide=E.fromstring(out[lookup[entry.get(f'{{{R}}}id')]])
        ids=[x.get('id') for x in slide.findall('.//p:cNvPr',NS)]
        assert len(set(ids))==len(ids)
        assert all(t.get('spid') in ids for t in slide.findall('.//p:spTgt',NS))
        audio_count+=len(slide.findall('.//a:audioFile',NS))
    assert audio_count==82
    destination=BUILD/'SER_evaluation_complete-candidate.pptx'
    with ZipFile(destination,'w',ZIP_DEFLATED) as z:
        for name,blob in out.items():z.writestr(name,blob)
    receipt={'source':str(source_path),'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
             'candidate_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
             'slide_count':85,'new_slide':40,'new_table_prediction_cells_verified':15,
             'audio_shapes':audio_count,'source_media_charts_workbooks_unchanged':True,
             'saved_prediction_sources_unchanged':True}
    (BUILD/'lexical-insertion-validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
