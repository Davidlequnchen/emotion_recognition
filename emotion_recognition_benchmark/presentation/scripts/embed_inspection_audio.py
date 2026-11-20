"""Attach full-length MP3s to the authored play icons, preserving native objects."""
import ast
import copy
import json
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E

DECK = Path(__file__).resolve().parents[1]
P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
P14 = 'http://schemas.microsoft.com/office/powerpoint/2010/main'
CT = 'http://schemas.openxmlformats.org/package/2006/content-types'


def embed(source, manifest, destination):
    # Reuse the project's existing, standards-compatible click-to-toggle timing.
    tree = ast.parse((DECK/'fix_audio.py').read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'timing')
    scope = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'timing', 'exec'), scope)
    with ZipFile(source) as z:
        parts = {name: z.read(name) for name in z.namelist()}
    records = json.loads(Path(manifest).read_text())
    if isinstance(records, list):
        records = {'1': records}
    for slide_number, audio in records.items():
        slide_path = f'ppt/slides/slide{slide_number}.xml'
        rel_path = f'ppt/slides/_rels/slide{slide_number}.xml.rels'
        slide, rels = E.fromstring(parts[slide_path]), E.fromstring(parts[rel_path])
        pics = slide.findall(f'.//{{{P}}}pic')
        if audio and 'picture_indices' in audio[0]:
            pics = [pics[x['picture_indices']] for x in audio]
        assert len(pics) == len(audio), (slide_number, len(pics), len(audio))
        ids = []
        next_id = max(int(x.get('id')) for x in slide.findall(f'.//{{{P}}}cNvPr')) + 1
        for index, (pic, record) in enumerate(zip(pics, audio), 1):
            # Keep a visible poster below the media object. PDF exporters may omit
            # media thumbnails even when they are visible in PowerPoint.
            media_pic = copy.deepcopy(pic)
            pic.addnext(media_pic)
            pic = media_pic
            nv = pic.find(f'{{{P}}}nvPicPr')
            props = nv.find(f'{{{P}}}cNvPr')
            props.set('id',str(next_id))
            next_id += 1
            props.set('name', 'Play '+record['clip_id'])
            props.set('descr', record['clip_id'])
            ids.append(props.get('id'))
            E.SubElement(props, f'{{{A}}}hlinkClick', action='ppaction://media')
            nonvisual = nv.find(f'{{{P}}}nvPr')
            rid_audio, rid_media = f'rIdAudio{index}', f'rIdMedia{index}'
            E.SubElement(nonvisual, f'{{{A}}}audioFile', {f'{{{R}}}link':rid_audio})
            ext = E.SubElement(E.SubElement(nonvisual, f'{{{P}}}extLst'), f'{{{P}}}ext', uri='{DAA4B4D4-6D71-4841-9C94-3DE7FCFB9230}')
            E.SubElement(ext, f'{{{P14}}}media', {f'{{{R}}}embed':rid_media}, nsmap={'p14':P14})
            media_name = f'inspection-{slide_number}-{index}.mp3'
            parts['ppt/media/'+media_name] = (DECK/record['path']).read_bytes()
            for rid, typ in [(rid_audio, R+'/audio'), (rid_media,'http://schemas.microsoft.com/office/2007/relationships/media')]:
                E.SubElement(rels, f'{{{REL}}}Relationship', Id=rid, Type=typ, Target='../media/'+media_name)
        slide.append(E.fromstring(scope['timing'](ids).replace('<p:timing>', f'<p:timing xmlns:p="{P}">').encode()))
        parts[slide_path] = E.tostring(slide, encoding='utf-8', xml_declaration=True)
        parts[rel_path] = E.tostring(rels, encoding='utf-8', xml_declaration=True)
    types = E.fromstring(parts['[Content_Types].xml'])
    if not any(x.get('Extension') == 'mp3' for x in types):
        E.SubElement(types, f'{{{CT}}}Default', Extension='mp3', ContentType='audio/mpeg')
    parts['[Content_Types].xml'] = E.tostring(types, encoding='utf-8', xml_declaration=True)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, 'w', ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name,data)
    print(f'Embedded {sum(map(len, records.values()))} audio clips: {destination}')


if __name__ == '__main__':
    embed(*sys.argv[1:])
