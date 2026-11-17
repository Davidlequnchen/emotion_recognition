"""Replace dataset/model/error sections and apply one presentation-wide layout.

The completed main deck has a single public path, SER_evaluation.pptx. Sources,
staging decks and older base packages remain private build inputs only.
"""
import argparse
import copy
import hashlib
import json
import posixpath
import re
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E

from merge_inspections import DECK,BUILD,P,A,R,REL,CT,NS,parts,relfile,resolve,xml_bytes

REFRESH=BUILD/'refresh'
EMU=9525
WIDTH,HEIGHT=1280,720
SCALE=.90


def title(root):
    ts=[t.text or '' for t in root.findall('.//a:t',NS)]
    return ts[1] if len(ts)>1 else (ts[0] if ts else '')


class Editor:
    def __init__(self,source):
        self.source=source
        self.parts=parts(source)
        self.root=E.fromstring(self.parts['ppt/presentation.xml'])
        self.rels=E.fromstring(self.parts['ppt/_rels/presentation.xml.rels'])
        self.types=E.fromstring(self.parts['[Content_Types].xml'])
        self.entries=list(self.root.find('p:sldIdLst',NS))
        first_paths={r.get('Id'):resolve('ppt/presentation.xml',r.get('Target')) for r in self.rels}
        first=E.fromstring(self.parts[first_paths[self.entries[0].get(f'{{{R}}}id')]])
        self.already_compact=first.find('p:cSld',NS).get('name','').endswith(':whitespace90')
        self.next_part=max(int(m.group(1)) for n in self.parts if (m:=re.fullmatch(r'ppt/slides/slide(\d+)\.xml',n)))+1
        self.next_id=max(int(x.get('id')) for x in self.entries)+1

    def paths(self):
        return {r.get('Id'):resolve('ppt/presentation.xml',r.get('Target')) for r in self.rels}

    def part(self,entry):return self.paths()[entry.get(f'{{{R}}}id')]
    def slide(self,entry):return E.fromstring(self.parts[self.part(entry)])
    def heading(self,entry):return title(self.slide(entry))

    def import_deck(self,file,prefix,order):
        # Rebuilding the canonical deck must not reuse relationship or media IDs.
        while any(r.get('Id','').startswith(prefix) for r in self.rels):
            prefix='next_'+prefix
        incoming=parts(file)
        root=E.fromstring(incoming['ppt/presentation.xml'])
        rels=E.fromstring(incoming['ppt/_rels/presentation.xml.rels'])
        incoming_paths={r.get('Id'):resolve('ppt/presentation.xml',r.get('Target')) for r in rels}
        slides=[incoming_paths[x.get(f'{{{R}}}id')] for x in root.find('p:sldIdLst',NS)]
        assert len(slides)==len(order),(file,len(slides),len(order))
        imported_types=E.fromstring(incoming['[Content_Types].xml'])
        overrides={t.get('PartName').lstrip('/'):t.get('ContentType') for t in imported_types if t.tag.endswith('Override')}
        known={t.get('PartName') for t in self.types if t.tag.endswith('Override')}
        extensions={t.get('Extension') for t in self.types if t.tag.endswith('Default')}
        for t in imported_types:
            if t.tag.endswith('Default') and t.get('Extension') not in extensions:
                self.types.append(copy.deepcopy(t));extensions.add(t.get('Extension'))
        mapping={}
        for src in slides:
            mapping[src]=f'ppt/slides/slide{self.next_part}.xml';self.next_part+=1
        copied=set()
        def cp(src):
            if src.startswith('ppt/slideLayouts/'):return 'ppt/slideLayouts/slideLayout1.xml'
            if src.startswith('ppt/notesMasters/'):return 'ppt/notesMasters/notesMaster1.xml'
            dst=mapping.setdefault(src,posixpath.join(posixpath.dirname(src),prefix+posixpath.basename(src)))
            if src in copied:return dst
            copied.add(src);self.parts[dst]=incoming[src]
            if src in overrides and '/'+dst not in known:
                E.SubElement(self.types,f'{{{CT}}}Override',PartName='/'+dst,ContentType=overrides[src]);known.add('/'+dst)
            if relfile(src) in incoming:
                rs=E.fromstring(incoming[relfile(src)])
                for r in rs:
                    if r.get('TargetMode')!='External':
                        r.set('Target',posixpath.relpath(cp(resolve(src,r.get('Target'))),posixpath.dirname(dst)))
                self.parts[relfile(dst)]=xml_bytes(rs)
            return dst
        entries=[]
        for i,(src,meta) in enumerate(zip(slides,order)):
            dst=cp(src)
            slide=E.fromstring(self.parts[dst])
            key=meta.get('key') or meta.get('dataset') or str(i+1)
            marker='ser-refresh:'+meta['kind']+':'+key
            if meta.get('already_compact'):marker+=':whitespace90'
            slide.find('p:cSld',NS).set('name',marker)
            self.parts[dst]=xml_bytes(slide)
            rid=prefix+'slide'+str(i+1)
            E.SubElement(self.rels,f'{{{REL}}}Relationship',Id=rid,Type=R+'/slide',Target=posixpath.relpath(dst,'ppt'))
            entry=E.Element(f'{{{P}}}sldId',{'id':str(self.next_id),f'{{{R}}}id':rid});self.next_id+=1
            entries.append((entry,meta))
        return entries

    def append_notes(self,owner,lines):
        if not lines:return
        rs=E.fromstring(self.parts[relfile(owner)])
        note_rel=next((r for r in rs if r.get('Type')==R+'/notesSlide'),None)
        assert note_rel is not None,owner
        note_path=resolve(owner,note_rel.get('Target'))
        note=E.fromstring(self.parts[note_path])
        bodies=[]
        for sp in note.findall('p:cSld/p:spTree/p:sp',NS):
            ph=sp.find('p:nvSpPr/p:nvPr/p:ph',NS)
            if ph is not None and ph.get('type')=='body':bodies.append(sp.find('p:txBody',NS))
        if not bodies:
            bodies=[sp.find('p:txBody',NS) for sp in note.findall('p:cSld/p:spTree/p:sp',NS) if sp.find('p:txBody',NS) is not None]
        assert bodies,note_path
        body=bodies[0]
        for line in ['Moved from slide footnotes / source lines:',*lines]:
            para=E.SubElement(body,f'{{{A}}}p');run=E.SubElement(para,f'{{{A}}}r');E.SubElement(run,f'{{{A}}}t').text=line
        self.parts[note_path]=xml_bytes(note)

    def whitespace(self,entry):
        owner=self.part(entry)
        slide=E.fromstring(self.parts[owner])
        cs=slide.find('p:cSld',NS)
        if cs.get('name','').endswith(':whitespace90'):return
        tree=cs.find('p:spTree',NS)
        notes=[]
        rs=E.fromstring(self.parts[relfile(owner)])
        urls={r.get('Id'):r.get('Target') for r in rs if r.get('TargetMode')=='External'}
        is_new=cs.get('name','').startswith('ser-refresh:')
        for shape in list(tree):
            transform=shape.find('p:spPr/a:xfrm',NS)
            if transform is None:transform=shape.find('p:xfrm',NS)
            if transform is None:transform=shape.find('p:grpSpPr/a:xfrm',NS)
            if transform is None:continue
            off,ext=transform.find('a:off',NS),transform.find('a:ext',NS)
            if off is None or ext is None:continue
            ts=shape.findall('.//a:t',NS)
            text=''.join(t.text or '' for t in ts)
            props=shape.find('p:nvSpPr/p:cNvPr',NS)
            name=(props.get('name','') if props is not None else '').lower()
            y=int(off.get('y'))/EMU
            sizes=[int(x.get('sz'))/100 for x in shape.findall('.//a:rPr',NS) if x.get('sz')]
            small=bool(sizes) and max(sizes)<=11
            footer=bool(text) and (y>=HEIGHT*.90 or (not is_new and y>=HEIGHT*.82 and small) or name in ['source','selection','retained'])
            if footer:
                if not text.strip().isdigit():
                    links=[urls.get(x.get(f'{{{R}}}id'),'') for x in shape.findall('.//a:hlinkClick',NS)]
                    notes.append(text+(' '+ ' '.join(x for x in links if x) if any(links) else ''))
                tree.remove(shape);continue
            # Keep script fallbacks explicit while retaining the original Latin face.
            for t in ts:
                if any('\u0e00'<=c<='\u0e7f' or '\u3000'<=c<='\u9fff' for c in (t.text or '')):
                    run=t.getparent();rpr=run.find('a:rPr',NS)
                    if rpr is None:rpr=E.Element(f'{{{A}}}rPr');run.insert(0,rpr)
                    for script in ['ea','cs']:
                        f=rpr.find(f'{{{A}}}{script}')
                        if f is None:f=E.SubElement(rpr,f'{{{A}}}{script}')
                        f.set('typeface','Arial Unicode MS')
            off.set('x',str(round((WIDTH/2+(int(off.get('x'))/EMU-WIDTH/2)*SCALE)*EMU)))
            off.set('y',str(round(int(off.get('y'))*SCALE)))
            ext.set('cx',str(round(int(ext.get('cx'))*SCALE)))
            ext.set('cy',str(round(int(ext.get('cy'))*SCALE)))
            for element in shape.iter():
                local=E.QName(element).localname
                if local in ['rPr','defRPr','endParaRPr'] and element.get('sz'):
                    element.set('sz',str(round(int(element.get('sz'))*SCALE)))
                if local=='gridCol' and element.get('w'):element.set('w',str(round(int(element.get('w'))*SCALE)))
                if local=='tr' and element.get('h'):element.set('h',str(round(int(element.get('h'))*SCALE)))
                if local in ['bodyPr','tcPr']:
                    for attr in ['lIns','rIns','tIns','bIns','marL','marR','marT','marB']:
                        if element.get(attr):element.set(attr,str(round(int(element.get(attr))*SCALE)))
                if local=='pPr':
                    for attr in ['marL','marR','indent']:
                        if element.get(attr):element.set(attr,str(round(int(element.get(attr))*SCALE)))
                if local=='spcPts' and element.get('val'):element.set('val',str(round(int(element.get('val'))*SCALE)))
                if local=='ln' and element.get('w'):element.set('w',str(round(int(element.get('w'))*SCALE)))
        self.append_notes(owner,notes)
        cs.set('name',cs.get('name','')+':whitespace90')
        self.parts[owner]=xml_bytes(slide)

    def save(self,destination):
        self.root.find('p:sldIdLst',NS)[:]=self.entries
        keep={x.get(f'{{{R}}}id') for x in self.entries}
        for r in list(self.rels):
            if r.get('Type')==R+'/slide' and r.get('Id') not in keep:self.rels.remove(r)
        self.parts['ppt/presentation.xml']=xml_bytes(self.root)
        self.parts['ppt/_rels/presentation.xml.rels']=xml_bytes(self.rels)
        app=E.fromstring(self.parts['docProps/app.xml'])
        for child in app:
            if E.QName(child).localname=='Slides':child.text=str(len(self.entries))
        self.parts['docProps/app.xml']=xml_bytes(app)
        reachable=set()
        def walk(name):
            if name in reachable:return
            reachable.add(name)
            rp=relfile(name) if name else '_rels/.rels'
            if rp not in self.parts:return
            reachable.add(rp)
            for rel in E.fromstring(self.parts[rp]):
                if rel.get('TargetMode')!='External':walk(resolve(name,rel.get('Target')))
        walk('');reachable.discard('')
        for t in list(self.types):
            if t.tag==f'{{{CT}}}Override' and t.get('PartName').lstrip('/') not in reachable:self.types.remove(t)
        self.parts['[Content_Types].xml']=xml_bytes(self.types);reachable.add('[Content_Types].xml')
        destination.parent.mkdir(parents=True,exist_ok=True)
        with ZipFile(destination,'w',ZIP_DEFLATED) as z:
            for name in sorted(reachable):z.writestr(name,self.parts[name])


def read_order(path):
    x=json.loads(path.read_text())
    return x['slides'] if isinstance(x,dict) and 'slides' in x else x


def save_refresh(editor):
    destination=REFRESH/'main-candidate.pptx';editor.save(destination)
    order=[{'slide':i+1,'part':editor.part(e),'title':editor.heading(e)} for i,e in enumerate(editor.entries)]
    (REFRESH/'slide-order.json').write_text(json.dumps(order,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'slides':len(order),'candidate':str(destination),'source_sha256':hashlib.sha256(editor.source.read_bytes()).hexdigest()},indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path)
    parser.add_argument('--survey-only',action='store_true',help='Replace only the ten-model survey, preserving all other slide XML')
    args=parser.parse_args()
    if args.source is None:
        args.source=DECK/'SER_evaluation.pptx' if args.survey_only else ((REFRESH/'source.pptx') if (REFRESH/'source.pptx').exists() else DECK/'SER_evaluation.pptx')
    editor=Editor(args.source)
    if args.survey_only:
        models=editor.import_deck(BUILD/'model-refresh.pptx','modelSurvey_',read_order(BUILD/'model-refresh-order.json'))
        survey=next(e for e,m in models if m['kind']=='model-survey')
        index=next(i for i,e in enumerate(editor.entries) if editor.slide(e).find('p:cSld',NS).get('name','').startswith('ser-refresh:model-survey'))
        editor.entries[index]=survey
        editor.whitespace(survey)
        save_refresh(editor)
        return
    dataset=editor.import_deck(BUILD/'dataset-refresh-audio.pptx','datasetRefresh_',read_order(BUILD/'dataset-refresh-order.json'))
    model=editor.import_deck(BUILD/'model-refresh.pptx','modelRefresh_',read_order(BUILD/'model-refresh-order.json'))
    errors=editor.import_deck(BUILD/'error-refresh-audio.pptx','errorRefresh_',read_order(BUILD/'error-refresh-order.json'))
    results=editor.import_deck(BUILD/'result-refresh.pptx','resultRefresh_',read_order(BUILD/'result-refresh-order.json'))
    listener=editor.import_deck(BUILD/'listener-plot-refresh.pptx','listenerPlot_',read_order(BUILD/'listener-plot-refresh-order.json'))
    original=editor.entries
    start=next(i for i,e in enumerate(original) if 'Sixteen Public Benchmark Datasets' in editor.heading(e) or ('ser-refresh:' in editor.slide(e).find('p:cSld',NS).get('name','') and 'overview' in editor.slide(e).find('p:cSld',NS).get('name','')))
    end=next(i for i,e in enumerate(original) if 'Inter-Rater Agreement: Actor vs Listeners' in editor.heading(e))
    original=original[:start]+[e for e,_ in dataset]+original[end:]
    survey=next(e for e,m in model if m['kind']=='model-survey')
    arch=[e for e,m in model if m['kind']=='model-architecture']
    old=next(i for i,e in enumerate(original) if editor.heading(e)=='Ten Evaluated Models' or editor.slide(e).find('p:cSld',NS).get('name','').startswith('ser-refresh:model-survey'))
    original[old]=survey
    original=[e for e in original if not editor.slide(e).find('p:cSld',NS).get('name','').startswith('ser-refresh:model-architecture')]
    location=next(i for i,e in enumerate(original) if editor.heading(e)=='Dedicated SER Models: Structure and Input Handling')+1
    original[location:location]=arch
    err_by_title={m['title']:e for e,m in errors}
    for i,e in enumerate(original):
        if editor.heading(e) in err_by_title:original[i]=err_by_title[editor.heading(e)]
    listener_entry,listener_meta=listener[0]
    listener_position=next(i for i,e in enumerate(original) if 'Actor vs Listener Labels (CREMA-D)' in editor.heading(e))
    original[listener_position]=listener_entry
    # One complete table per metric replaces the former split family / dataset pages.
    result_by_key={m['key']:e for e,m in results}
    consolidated=[];used=set()
    for e in original:
        heading=editor.heading(e)
        marker=editor.slide(e).find('p:cSld',NS).get('name','')
        metric=None
        if heading.startswith('Dataset Accuracy:'):metric='accuracy'
        elif re.match(r'A\d+ · UAR:',heading):metric='uar'
        elif re.match(r'A\d+ · Macro-F1:',heading):metric='macro_f1'
        elif marker.startswith('ser-refresh:result-table:'):metric=marker.split(':')[2]
        if metric:
            if metric not in used:consolidated.append(result_by_key[metric]);used.add(metric)
        elif 'Label Mapping: Six Listening-Example Corpora' not in heading:
            consolidated.append(e)
    assert used==set(result_by_key),used
    original=consolidated
    editor.entries=original
    # Keep slide-title based navigation and notes coherent after sections grow.
    for i,e in enumerate(editor.entries):
        root=editor.slide(e)
        for t in root.findall('.//a:t',NS):
            if t.text and 'The earlier five-SER listener-vote case studies follow' in t.text:
                t.text='Retained listener-vote case studies follow, now with predictions from all ten models.'
            if t.text and t.text.startswith('A6 · emotion2vec+'):
                t.text=t.text.replace('A6 ·','A3 ·',1)
            if i==0 and t.text and 'Chen Lequn' in t.text:
                t.text='Chen Lequn  ·  Benchmark results and listening inspections  ·  Updated 17 November 2026'
        editor.parts[editor.part(e)]=xml_bytes(root)
        editor.whitespace(e)
    # Chart data/workbooks remain intact; scale only their explicitly authored typography.
    for name,blob in list(editor.parts.items()):
        if not editor.already_compact and re.fullmatch(r'ppt/charts/[^/]+\.xml',name):
            chart=E.fromstring(blob)
            for rpr in chart.xpath('//*[local-name()="rPr" or local-name()="defRPr" or local-name()="endParaRPr"]'):
                if rpr.get('sz'):rpr.set('sz',str(round(int(rpr.get('sz'))*SCALE)))
            editor.parts[name]=xml_bytes(chart)
    save_refresh(editor)


if __name__=='__main__':main()
