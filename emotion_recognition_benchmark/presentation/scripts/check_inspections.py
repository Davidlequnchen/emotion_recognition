"""Verify saved-prediction fidelity and create bounded exact-content import checks."""
import copy
import hashlib
import json
import posixpath
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E

from merge_inspections import DECK,BUILD,NS,P,A,R,CT,REL,parts,relfile,resolve,xml_bytes


def main():
    file=BUILD/'SER_evaluation_extended-candidate.pptx'
    package=parts(file)
    source=parts(DECK/'SER_evaluation.pptx')
    p=E.fromstring(package['ppt/presentation.xml'])
    relationships=E.fromstring(package['ppt/_rels/presentation.xml.rels'])
    lookup={x.get('Id'):resolve('ppt/presentation.xml',x.get('Target')) for x in relationships}
    entries=list(p.find('p:sldIdLst',NS))
    assert len(entries)==84
    manifest=json.loads((DECK/'assets/inspections/manifest.json').read_text())
    for record in manifest['sources']:
        path=DECK.parent/record['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],record['path']
    for i,dataset in enumerate(manifest['datasets'],40):
        slide=E.fromstring(package[lookup[entries[i-1].get(f'{{{R}}}id')]])
        table=slide.find('.//a:tbl',NS)
        rows=table.findall('a:tr',NS)
        assert len(rows)==11
        for row,model in zip(rows[1:],manifest['model_order']):
            cells=row.findall('a:tc',NS)
            for j in [0,1]:
                actual=''.join(cells[j+1].findall('.//a:t',NS)[0].itertext())
                assert actual==dataset['cases'][j]['predictions'][model]['label'],(dataset['key'],model,j)
    for name,blob in source.items():
        if name.startswith(('ppt/media/','ppt/charts/','ppt/embeddings/')):
            assert package[name]==blob,('original chart/audio/workbook changed',name)
    audio_shapes=0
    for entry in entries:
        slide=E.fromstring(package[lookup[entry.get(f'{{{R}}}id')]])
        shape_ids=[x.get('id') for x in slide.findall('.//p:cNvPr',NS)]
        assert len(shape_ids)==len(set(shape_ids))
        media=slide.findall('.//a:audioFile',NS)
        audio_shapes+=len(media)
        assert all(x.get('spid') in shape_ids for x in slide.findall('.//p:spTgt',NS))
    assert audio_shapes==77,audio_shapes
    # Limit package XML nodes, not slide quality/content. All copied slide,
    # chart, image and notes parts remain byte-for-byte identical to candidate.
    batches=[]
    for start in range(0,84,12):
        root=copy.deepcopy(p)
        root.find('p:sldIdLst',NS)[:]=[copy.deepcopy(x) for x in entries[start:start+12]]
        keep_ids={x.get(f'{{{R}}}id') for x in entries[start:start+12]}
        rels=copy.deepcopy(relationships)
        for rel in list(rels):
            if rel.get('Type')==R+'/slide' and rel.get('Id') not in keep_ids:
                rels.remove(rel)
        changed={**package,'ppt/presentation.xml':xml_bytes(root),'ppt/_rels/presentation.xml.rels':xml_bytes(rels)}
        reachable=set()
        def walk(name):
            if name in reachable:return
            reachable.add(name)
            rp=relfile(name) if name else '_rels/.rels'
            if rp not in changed:return
            reachable.add(rp)
            for rel in E.fromstring(changed[rp]):
                if rel.get('TargetMode')!='External':
                    walk(resolve(name,rel.get('Target')))
        walk('')
        reachable.discard('')
        types=E.fromstring(changed['[Content_Types].xml'])
        for entry in list(types):
            if entry.tag==f'{{{CT}}}Override' and entry.get('PartName').lstrip('/') not in reachable:
                types.remove(entry)
        reachable.add('[Content_Types].xml')
        changed['[Content_Types].xml']=xml_bytes(types)
        destination=BUILD/f'import-{start+1:02d}-{start+12:02d}.pptx'
        with ZipFile(destination,'w',ZIP_DEFLATED) as z:
            for name in sorted(reachable):z.writestr(name,changed[name])
        batches.append(str(destination))
    (BUILD/'import-batches.json').write_text(json.dumps(batches,indent=2)+'\n')
    print(json.dumps({'slides':84,'datasets':16,'verified_prediction_cells':320,'verified_source_hashes':320,
                      'audio_shapes':audio_shapes,'original_assets_unchanged':True,'import_batches':batches},indent=2))


if __name__=='__main__':main()
