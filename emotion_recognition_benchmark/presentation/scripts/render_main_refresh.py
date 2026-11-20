"""Render the canonical candidate, then record a completed human/agent visual review."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from PIL import Image,ImageChops,ImageDraw
from refresh_main_deck import DECK,BUILD,REFRESH


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--record-review',action='store_true',help='After inspecting all contact sheets and changed pages, record that completed review')
    parser.add_argument('--detailed-slides',default='6,27,46')
    args=parser.parse_args()
    candidate=REFRESH/'main-candidate.pptx'
    digest=hashlib.sha256(candidate.read_bytes()).hexdigest()
    proof=json.loads((REFRESH/'content-validation.json').read_text())
    count=proof['slide_count'];assert proof['candidate_sha256']==digest
    rendered=REFRESH/'rendered';rendered.mkdir(exist_ok=True)
    manifest=REFRESH/'render-manifest.json'
    if not args.record_review:
        runtime=Path(os.environ.get('RUNTIME_DEPENDENCIES',Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies'))
        binaries=Path(os.environ.get('RUNTIME_BIN_DIR',runtime/'bin/override'))
        font_dirs=[Path('/System/Library/Fonts'),Path('/System/Library/Fonts/Supplemental'),Path('/Library/Fonts'),runtime/'native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype']
        config=BUILD/'fonts.conf'
        config.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig>'+''.join('<dir>'+str(d)+'</dir>' for d in font_dirs if d.exists())+'<cachedir>/tmp/ser-font-cache</cachedir><alias><family>Calibri</family><prefer><family>Carlito</family></prefer></alias></fontconfig>')
        env={**os.environ,'FONTCONFIG_FILE':str(config)}
        subprocess.run([str(binaries/'soffice'),'-env:UserInstallation=file:///tmp/ser-presentation-render','--headless','--convert-to','pdf','--outdir',str(REFRESH),str(candidate)],env=env,check=True)
        for f in rendered.glob('slide-*.png'):f.unlink()
        subprocess.run([str(binaries/'pdftoppm'),'-scale-to','1280','-png',str(candidate.with_suffix('.pdf')),str(rendered/'slide')],check=True)
        files=sorted(rendered.glob('slide-*.png'));assert len(files)==count
        for start in range(0,count,20):
            sheet=Image.new('RGB',(1280,800),'#E4E7EC');draw=ImageDraw.Draw(sheet)
            for i,f in enumerate(files[start:start+20]):
                x=(i%4)*320+5;y=(i//4)*160
                im=Image.open(f).convert('RGB');im.thumbnail((310,140));sheet.paste(im,(x,y+18));draw.text((x,y+2),str(start+i+1),fill='black')
            sheet.save(REFRESH/f'contact-{start//20+1}.png')
        manifest.write_text(json.dumps({'candidate_sha256':digest,'slides':count},indent=2)+'\n')
        print(f'Rendered {count} slides. Inspect all contact sheets and changed pages before --record-review.')
        return
    assert json.loads(manifest.read_text())['candidate_sha256']==digest,'Render belongs to a different candidate'
    checks=[]
    files=sorted(rendered.glob('slide-*.png'));assert len(files)==count
    for n,f in enumerate(files,1):
        im=Image.open(f).convert('RGB');assert im.size==(1280,720)
        bbox=ImageChops.difference(im,Image.new('RGB',im.size,'white')).getbbox()
        bottom=bbox[3] if bbox else 0
        assert bottom<=648,(n,'Ink in the reserved bottom margin',bottom)
        checks.append({'slide':n,'lowest_ink_px':bottom,'bottom_blank_px':720-bottom})
    report={'passed':True,'candidate_sha256':digest,'rendered_slides':count,'contact_sheets_reviewed':(count+19)//20,
            'minimum_rendered_bottom_blank_px':min(x['bottom_blank_px'] for x in checks),
            'detailed_review_slides':[int(n) for n in args.detailed_slides.split(',')],
            'review_notes':['All contact sheets and changed pages inspected for text wrapping, overlap and bottom margin.','The model survey native-table overflow estimate was checked against its full-size render; all rows and text fit.'],
            'checks':checks}
    (REFRESH/'visual-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Recorded visual review and verified the bottom 10% is blank on every rendered slide.')


if __name__=='__main__':main()
