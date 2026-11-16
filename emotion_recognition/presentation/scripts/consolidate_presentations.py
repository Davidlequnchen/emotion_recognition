"""Keep one complete deck and one two-example semantics/prosody deck.

All intermediates and validation receipts stay private. Publication copies only
the two validated packages to stable filenames in updated/.
"""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E

from merge_inspections import DECK,BUILD,NS,P,A,R,CT,parts,relfile,resolve,xml_bytes

FULL='SER_evaluation_extended.pptx'
EXAMPLES='Semantic_vs_Vocal_Emotion.pptx'
VALIDATED=BUILD/'consolidated'


def slide_inventory(package):
    root=E.fromstring(package['ppt/presentation.xml'])
    rels=E.fromstring(package['ppt/_rels/presentation.xml.rels'])
    lookup={r.get('Id'):resolve('ppt/presentation.xml',r.get('Target')) for r in rels}
    entries=list(root.find('p:sldIdLst',NS))
    paths=[lookup[x.get(f'{{{R}}}id')] for x in entries]
    audio=sum(len(E.fromstring(package[p]).findall('.//a:audioFile',NS)) for p in paths)
    return root,rels,entries,paths,audio


def publish():
    expected={FULL:(85,82),EXAMPLES:(2,9)}
    for name,(count,audio_count) in expected.items():
        package=parts(VALIDATED/name)
        _,_,entries,_,audio=slide_inventory(package)
        assert (len(entries),audio)==(count,audio_count)
        receipt=json.loads((BUILD/('consolidated__'+name+'.validation.json')).read_text())
        digest=hashlib.sha256((VALIDATED/name).read_bytes()).hexdigest()
        assert receipt['finalSha256']==digest
    updated=DECK/'updated'
    updated.mkdir(exist_ok=True)
    for name in expected:
        shutil.copyfile(VALIDATED/name,updated/name)
    for name in ['SER_evaluation_extended_v2.pptx','Emotionally_Loaded_Sentence.pptx']:
        (updated/name).unlink(missing_ok=True)
    assert {p.name for p in updated.glob('*.pptx')}==set(expected)
    print('Published only: '+', '.join(expected))


def prepare(source_path):
    source_path=source_path.resolve()
    source=parts(source_path)
    root,rels,entries,paths,audio=slide_inventory(source)
    assert (len(entries),audio)==(85,82)
    titles=['Same Words, Different Emotions','Emotionally Loaded Words']
    for index,title in zip([38,39],titles):
        assert any(title in (t.text or '') for t in E.fromstring(source[paths[index]]).findall('.//a:t',NS))
    full_candidate=BUILD/'SER_evaluation_complete-candidate.pptx'
    if source_path!=full_candidate.resolve():
        shutil.copyfile(source_path,full_candidate)
    chosen=entries[38:40]
    root.find('p:sldIdLst',NS)[:]=[copy.deepcopy(x) for x in chosen]
    keep={e.get(f'{{{R}}}id') for e in chosen}
    for rel in list(rels):
        if rel.get('Type')==R+'/slide' and rel.get('Id') not in keep:
            rels.remove(rel)
    changed=dict(source)
    changed['ppt/presentation.xml']=xml_bytes(root)
    changed['ppt/_rels/presentation.xml.rels']=xml_bytes(rels)
    for number,name in enumerate(paths[38:40],1):
        slide=E.fromstring(changed[name])
        for sp in slide.findall('p:cSld/p:spTree/p:sp',NS):
            off=sp.find('p:spPr/a:xfrm/a:off',NS)
            ts=sp.findall('.//a:t',NS)
            if off is not None and int(off.get('x'))>10_000_000 and int(off.get('y'))>6_000_000 and len(ts)==1 and (ts[0].text or '').isdigit():
                ts[0].text=str(number)
        changed[name]=xml_bytes(slide)
    app=E.fromstring(changed['docProps/app.xml'])
    for child in app:
        if E.QName(child).localname=='Slides':child.text='2'
    changed['docProps/app.xml']=xml_bytes(app)
    # Follow package relationships to include only the two slides' required
    # notes, images, media, theme, layout and master parts.
    reachable=set()
    def walk(name):
        if name in reachable:return
        reachable.add(name)
        rp=relfile(name) if name else '_rels/.rels'
        if rp not in changed:return
        reachable.add(rp)
        for rel in E.fromstring(changed[rp]):
            if rel.get('TargetMode')!='External':walk(resolve(name,rel.get('Target')))
    walk('');reachable.discard('')
    types=E.fromstring(changed['[Content_Types].xml'])
    for entry in list(types):
        if entry.tag==f'{{{CT}}}Override' and entry.get('PartName').lstrip('/') not in reachable:
            types.remove(entry)
    changed['[Content_Types].xml']=xml_bytes(types)
    reachable.add('[Content_Types].xml')
    destination=BUILD/'Semantic_vs_Vocal_Emotion-candidate.pptx'
    with ZipFile(destination,'w',ZIP_DEFLATED) as z:
        for name in sorted(reachable):z.writestr(name,changed[name])
    extracted=parts(destination)
    _,_,selected,_,extracted_audio=slide_inventory(extracted)
    assert (len(selected),extracted_audio)==(2,9)
    for name,blob in extracted.items():
        if name.startswith(('ppt/media/','ppt/charts/','ppt/embeddings/')):
            assert blob==source[name]
    VALIDATED.mkdir(exist_ok=True)
    print(json.dumps({'complete_slides':85,'example_slides':2,'example_audio_clips':9,
                      'complete_candidate':str(full_candidate),'examples_candidate':str(destination)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=BUILD/'SER_evaluation_complete-candidate.pptx')
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    publish() if args.publish else prepare(args.source)
