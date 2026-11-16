"""Insert authored inspections while retaining original slide, chart and audio parts.

New slides reuse the source deck's layout and notes master. Existing slide content
is unchanged apart from the scope/date/outline text and sequential page numbers.
"""
import copy
import json
import posixpath
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E

DECK=Path(__file__).resolve().parents[1]
BUILD=DECK/'.inspection-build'
P='http://schemas.openxmlformats.org/presentationml/2006/main'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL='http://schemas.openxmlformats.org/package/2006/relationships'
CT='http://schemas.openxmlformats.org/package/2006/content-types'
NS={'p':P,'a':A,'r':R}


def xml_bytes(root):
    return E.tostring(root,encoding='utf-8',xml_declaration=True)


def parts(file):
    with ZipFile(file) as z:
        return {name:z.read(name) for name in z.namelist() if not name.endswith('/')}


def relfile(name):
    return posixpath.join(posixpath.dirname(name),'_rels',posixpath.basename(name)+'.rels')


def resolve(owner,target):
    return target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join(posixpath.dirname(owner),target))


def main():
    out=parts(DECK/'SER_evaluation.pptx')
    types=E.fromstring(out['[Content_Types].xml'])
    known_types={x.get('PartName') for x in types if x.tag.endswith('Override')}
    known_ext={x.get('Extension') for x in types if x.tag.endswith('Default')}

    def import_slides(file,prefix,targets):
        imported=parts(file)
        imported_types=E.fromstring(imported['[Content_Types].xml'])
        override={x.get('PartName').lstrip('/'):x.get('ContentType') for x in imported_types if x.tag.endswith('Override')}
        for x in imported_types:
            if x.tag.endswith('Default') and x.get('Extension') not in known_ext:
                types.append(copy.deepcopy(x));known_ext.add(x.get('Extension'))
        mapping={f'ppt/slides/slide{i}.xml':dst for i,dst in targets.items()}
        copied=set()

        def cp(src):
            if src.startswith('ppt/slideLayouts/'):
                return 'ppt/slideLayouts/slideLayout1.xml'
            if src.startswith('ppt/notesMasters/'):
                return 'ppt/notesMasters/notesMaster1.xml'
            dst=mapping.setdefault(src,posixpath.join(posixpath.dirname(src),prefix+posixpath.basename(src)))
            if src in copied:
                return dst
            copied.add(src)
            out[dst]=imported[src]
            if src in override and '/'+dst not in known_types:
                E.SubElement(types,f'{{{CT}}}Override',PartName='/'+dst,ContentType=override[src]);known_types.add('/'+dst)
            rp=relfile(src)
            if rp in imported:
                relationships=E.fromstring(imported[rp])
                for rel in relationships:
                    if rel.get('TargetMode')=='External':
                        continue
                    target=resolve(src,rel.get('Target'))
                    rel.set('Target',posixpath.relpath(cp(target),posixpath.dirname(dst)))
                out[relfile(dst)]=xml_bytes(relationships)
            return dst

        return {i:cp(f'ppt/slides/slide{i}.xml') for i in targets}

    case_targets={1:'ppt/slides/slide37.xml',**{i:f'ppt/slides/slide{66+i}.xml' for i in range(2,18)}}
    case_paths=import_slides(BUILD/'inspections-audio.pptx','inspection_',case_targets)
    semantic_paths=import_slides(BUILD/'semantic-candidate.pptx','semantic_',{1:'ppt/slides/slide84.xml'})
    root=E.fromstring(out['ppt/presentation.xml'])
    root_rels=E.fromstring(out['ppt/_rels/presentation.xml.rels'])
    sldlist=root.find('p:sldIdLst',NS)
    originals=list(sldlist)
    next_id=max(int(x.get('id')) for x in originals)+1
    inserted=[]
    for i,slide_path in enumerate([semantic_paths[1],*[case_paths[j] for j in range(2,18)]],1):
        rid=f'rIdInspectionSlide{i}'
        E.SubElement(root_rels,f'{{{REL}}}Relationship',Id=rid,Type=R+'/slide',Target=posixpath.relpath(slide_path,'ppt'))
        inserted.append(E.Element(f'{{{P}}}sldId',{'id':str(next_id),f'{{{R}}}id':rid}))
        next_id+=1
    order=originals[:38]+inserted+originals[38:]
    sldlist[:]=order
    by_rid={x.get('Id'):resolve('ppt/presentation.xml',x.get('Target')) for x in root_rels}

    replacements={
        'Chen Lequn  ·  Completed benchmark  ·  15 November 2026':'Chen Lequn  ·  Completed benchmark and listening inspections  ·  16 November 2026',
        'Audio clips from public research corpora, used under their licences (CREMA-D ODbL, RAVDESS CC BY-NC-SA, TESS and MCAE-SPPS CC BY-NC, MESS CC BY, JL-Corpus CC0).':'Internal research inspection copy. Audio from the 16 benchmark corpora retains its original source terms and credits.',
        'Case studies of disagreement between models, acted labels and listeners':'Ten-model listening inspections on all 16 datasets and same-sentence API examples',
        '5 · HISTORICAL SER LISTENING EXAMPLES':'5 · LISTENING INSPECTIONS',
    }
    footer_source=E.fromstring(out['ppt/slides/slide38.xml'])
    footer=None
    for sp in footer_source.findall('p:cSld/p:spTree/p:sp',NS):
        txt=''.join(sp.itertext())
        off=sp.find('p:spPr/a:xfrm/a:off',NS)
        if off is not None and int(off.get('x'))>10_000_000 and int(off.get('y'))>6_000_000:
            footer=copy.deepcopy(sp);break
    assert footer is not None
    pages=[]
    page=0
    for number,entry in enumerate(order,1):
        name=by_rid[entry.get(f'{{{R}}}id')]
        slide=E.fromstring(out[name])
        for text in slide.findall('.//a:t',NS):
            if text.text in replacements:
                text.text=replacements[text.text]
            # Keep Calibri for Latin and specify a real script font for the
            # Mandarin/Thai transcripts, including Office/PDF fallback readers.
            if 40 <= number <= 55 and any('\u0e00' <= c <= '\u0e7f' or '\u3000' <= c <= '\u9fff' for c in (text.text or '')):
                run=text.getparent()
                props=run.find(f'{{{A}}}rPr')
                if props is None:
                    props=E.Element(f'{{{A}}}rPr');run.insert(0,props)
                for script in ['ea','cs']:
                    font=props.find(f'{{{A}}}{script}')
                    if font is None:
                        font=E.SubElement(props,f'{{{A}}}{script}')
                    font.set('typeface','Arial Unicode MS')
        text_values=[x.text or '' for x in slide.findall('.//a:t',NS)]
        is_cover=number==1
        is_thanks=any('Thank you' in x for x in text_values)
        if not is_cover and not is_thanks:
            page+=1
            current=None
            for sp in slide.findall('p:cSld/p:spTree/p:sp',NS):
                off=sp.find('p:spPr/a:xfrm/a:off',NS)
                ts=sp.findall('.//a:t',NS)
                if off is not None and int(off.get('x'))>10_000_000 and int(off.get('y'))>6_000_000 and len(ts)==1 and (ts[0].text or '').isdigit():
                    current=sp;break
            if current is None:
                current=copy.deepcopy(footer)
                max_id=max(int(x.get('id')) for x in slide.findall('.//p:cNvPr',NS))
                current.find('p:nvSpPr/p:cNvPr',NS).set('id',str(max_id+1))
                slide.find('p:cSld/p:spTree',NS).append(current)
            current.find('.//a:t',NS).text=str(page)
        out[name]=xml_bytes(slide)
        pages.append({'slide':number,'part':name,'title':text_values[1] if len(text_values)>1 else text_values[0]})
    out['ppt/presentation.xml']=xml_bytes(root)
    out['ppt/_rels/presentation.xml.rels']=xml_bytes(root_rels)
    out['[Content_Types].xml']=xml_bytes(types)
    if 'docProps/app.xml' in out:
        app=E.fromstring(out['docProps/app.xml'])
        for x in app:
            if E.QName(x).localname=='Slides':
                x.text=str(len(order))
        out['docProps/app.xml']=xml_bytes(app)
    destination=BUILD/'SER_evaluation_extended-candidate.pptx'
    with ZipFile(destination,'w',ZIP_DEFLATED) as z:
        for name,blob in out.items():
            z.writestr(name,blob)
    (BUILD/'slide-order.json').write_text(json.dumps(pages,indent=2)+'\n')
    print(f'Merged {len(order)} slides: {destination}')


if __name__=='__main__':
    main()
