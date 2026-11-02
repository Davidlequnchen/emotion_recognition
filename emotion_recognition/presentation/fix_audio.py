"""Make the embedded MP3s proper PowerPoint audio that plays when its icon is clicked.

pptxgenjs writes audio as <a:videoFile> without timing; this rewrites it as <a:audioFile> and adds, per slide, one
click trigger per audio icon (start/pause on click, as PowerPoint's "Start: when clicked on"), then fixes the MP3
content type. Usage: python fix_audio.py deck.pptx
"""
import re
import sys
import zipfile
import shutil
import tempfile

src = sys.argv[1]
tmp = tempfile.mktemp(suffix=".pptx")
zin = zipfile.ZipFile(src)
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)


def timing(spids):
    """Timing tree: an interactive sequence per audio shape (toggle play/pause on click) + the media nodes."""
    nid = [1]

    def n():
        nid[0] += 1
        return nid[0]

    seqs, media = [], []
    for sp in spids:
        a, b, c, d, e = n(), n(), n(), n(), n()
        seqs.append(
            f'<p:seq concurrent="1" nextAc="seek"><p:cTn id="{a}" restart="whenNotActive" fill="hold" evtFilter="cancelBubble" nodeType="interactiveSeq">'
            f'<p:stCondLst><p:cond evt="onClick" delay="0"><p:tgtEl><p:spTgt spid="{sp}"/></p:tgtEl></p:cond></p:stCondLst>'
            f'<p:endSync evt="end" delay="0"><p:rtn val="all"/></p:endSync><p:childTnLst>'
            f'<p:par><p:cTn id="{b}" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
            f'<p:par><p:cTn id="{c}" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
            f'<p:par><p:cTn id="{d}" presetID="2" presetClass="mediacall" presetSubtype="0" fill="hold" nodeType="clickEffect">'
            f'<p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
            f'<p:cmd type="call" cmd="togglePause"><p:cBhvr><p:cTn id="{e}" dur="1" fill="hold"/><p:tgtEl><p:spTgt spid="{sp}"/></p:tgtEl></p:cBhvr></p:cmd>'
            f'</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>'
            f'</p:childTnLst></p:cTn><p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:spTgt spid="{sp}"/></p:tgtEl></p:cond></p:nextCondLst></p:seq>')
    for sp in spids:
        media.append(f'<p:audio><p:cMediaNode vol="80000"><p:cTn id="{n()}" fill="hold" display="0"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst></p:cTn>'
                     f'<p:tgtEl><p:spTgt spid="{sp}"/></p:tgtEl></p:cMediaNode></p:audio>')
    return ('<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>'
            + "".join(seqs) + "".join(media) + '</p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>')


n_audio = n_slides = 0
names = zin.namelist()
names.sort(key=lambda x: x != "[Content_Types].xml")
for name in names:
    data = zin.read(name)
    if name == "[Content_Types].xml":
        data = data.replace(b'ContentType="audio/mp3"', b'ContentType="audio/mpeg"')
    m = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
    if m:
        rels = zin.read(f"ppt/slides/_rels/slide{m.group(1)}.xml.rels").decode()
        mp3 = set(re.findall(r'Id="(rId\d+)"[^>]*Target="[^"]+\.mp3"', rels))
        xml = data.decode()
        spids = []

        def fix(pic):
            s = pic.group(0)
            link = re.search(r'<a:videoFile r:link="(rId\d+)"/>', s)
            if not link or link.group(1) not in mp3:
                return s
            spids.append(re.search(r'<p:cNvPr id="(\d+)"', s).group(1))
            return s.replace("<a:videoFile ", "<a:audioFile ")

        xml = re.sub(r"<p:pic>.*?</p:pic>", fix, xml, flags=re.S)
        if spids:
            assert "<p:timing" not in xml
            xml = xml.replace("</p:clrMapOvr>", "</p:clrMapOvr>" + timing(spids), 1)
            n_audio += len(spids)
            n_slides += 1
        data = xml.encode()
    zout.writestr(zin.getinfo(name), data)
zout.close()
shutil.move(tmp, src)
print(f"{src}: {n_audio} audio clips on {n_slides} slides set to play on click")
