"""Publish only the canonical main deck and its two-page sentence excerpt."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E

from merge_inspections import DECK,BUILD,NS,P,A,R,CT,parts,relfile,resolve,xml_bytes

REFRESH=BUILD/'refresh'
VALIDATED=REFRESH/'validated-20261117-compact'
MAIN='SER_evaluation.pptx'
EXAMPLES='Semantic_vs_Vocal_Emotion.pptx'


def slide_inventory(package):
    root=E.fromstring(package['ppt/presentation.xml'])
    rels=E.fromstring(package['ppt/_rels/presentation.xml.rels'])
    lookup={r.get('Id'):resolve('ppt/presentation.xml',r.get('Target')) for r in rels}
    entries=list(root.find('p:sldIdLst',NS))
    paths=[lookup[x.get(f'{{{R}}}id')] for x in entries]
    audio=sum(len(E.fromstring(package[p]).findall('.//a:audioFile',NS)) for p in paths)
    return root,rels,entries,paths,audio


def publish():
    expected={MAIN:(92,122),EXAMPLES:(2,9)}
    for name,(count,audio_count) in expected.items():
        _,_,entries,_,audio=slide_inventory(parts(VALIDATED/name))
        assert (len(entries),audio)==(count,audio_count)
        receipt=json.loads((BUILD/(VALIDATED.name+'__'+name+'.validation.json')).read_text())
        assert receipt['finalSha256']==hashlib.sha256((VALIDATED/name).read_bytes()).hexdigest()
    proof=json.loads((REFRESH/'content-validation.json').read_text())
    assert proof['passed'] and proof['slide_count']==92
    assert proof['candidate_sha256']==hashlib.sha256((VALIDATED/MAIN).read_bytes()).hexdigest()
    for name in ['import-validation.json','visual-validation.json']:
        check=json.loads((REFRESH/name).read_text())
        assert check['passed'],name
        assert check.get('candidate_sha256',check.get('candidateSha256'))==proof['candidate_sha256'],name
    tables=json.loads((VALIDATED/(MAIN+'.table-validation.json')).read_text())
    assert tables['passed'] and tables['tableOwners']==proof['native_table_slides']
    covered=[]
    for check in tables['checks']:
        assert check['finalSha256']==proof['candidate_sha256']
        receipt=json.loads(Path(check['receiptPath']).read_text())
        assert receipt['finalSha256']==check['finalSha256']
        assert receipt['presentationLayout']['findingCount']==0
        covered.extend(check['owners'])
    assert covered==proof['native_table_slides']
    shutil.copyfile(VALIDATED/MAIN,DECK/MAIN)
    updated=DECK/'updated';updated.mkdir(exist_ok=True)
    # Keep an unchanged excerpt byte-stable when only ZIP timestamps differ.
    if not (updated/EXAMPLES).exists() or parts(VALIDATED/EXAMPLES)!=parts(updated/EXAMPLES):
        shutil.copyfile(VALIDATED/EXAMPLES,updated/EXAMPLES)
    for name in ['SER_evaluation_extended.pptx','SER_evaluation_extended_v2.pptx','Emotionally_Loaded_Sentence.pptx']:
        (updated/name).unlink(missing_ok=True)
    assert {p.name for p in updated.glob('*.pptx')}=={EXAMPLES}
    print('Published SER_evaluation.pptx and updated/Semantic_vs_Vocal_Emotion.pptx only')


def prepare(source_path):
    source=parts(source_path)
    root,rels,entries,paths,audio=slide_inventory(source)
    assert len(entries)==92 and audio==122
    selected=[]
    for i,path in enumerate(paths):
        ts=[t.text or '' for t in E.fromstring(source[path]).findall('.//a:t',NS)]
        if any(t.startswith('Same Words, Different Emotions: API Predictions') or t.startswith('Emotionally Loaded Words: Unchanged API Predictions') for t in ts):
            selected.append(i)
    assert len(selected)==2
    chosen=[entries[i] for i in selected]
    root.find('p:sldIdLst',NS)[:]=[copy.deepcopy(x) for x in chosen]
    keep={e.get(f'{{{R}}}id') for e in chosen}
    for r in list(rels):
        if r.get('Type')==R+'/slide' and r.get('Id') not in keep:rels.remove(r)
    changed=dict(source)
    changed['ppt/presentation.xml']=xml_bytes(root)
    changed['ppt/_rels/presentation.xml.rels']=xml_bytes(rels)
    app=E.fromstring(changed['docProps/app.xml'])
    for child in app:
        if E.QName(child).localname=='Slides':child.text='2'
    changed['docProps/app.xml']=xml_bytes(app)
    reachable=set()
    def walk(name):
        if name in reachable:return
        reachable.add(name)
        rp=relfile(name) if name else '_rels/.rels'
        if rp not in changed:return
        reachable.add(rp)
        for r in E.fromstring(changed[rp]):
            if r.get('TargetMode')!='External':walk(resolve(name,r.get('Target')))
    walk('');reachable.discard('')
    types=E.fromstring(changed['[Content_Types].xml'])
    for t in list(types):
        if t.tag==f'{{{CT}}}Override' and t.get('PartName').lstrip('/') not in reachable:types.remove(t)
    changed['[Content_Types].xml']=xml_bytes(types);reachable.add('[Content_Types].xml')
    REFRESH.mkdir(exist_ok=True);VALIDATED.mkdir(exist_ok=True)
    destination=REFRESH/'examples-candidate.pptx'
    with ZipFile(destination,'w',ZIP_DEFLATED) as z:
        for name in sorted(reachable):z.writestr(name,changed[name])
    _,_,ex,_,ea=slide_inventory(parts(destination));assert len(ex)==2 and ea==9
    print('Prepared two-page excerpt from main slides '+','.join(str(i+1) for i in selected))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=REFRESH/'main-candidate.pptx')
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--staging',type=Path,default=VALIDATED,help='Private validated output directory')
    a=parser.parse_args()
    VALIDATED=a.staging.resolve()
    publish() if a.publish else prepare(a.source)
