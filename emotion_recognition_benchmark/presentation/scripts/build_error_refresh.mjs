// Compact ten-model continuations of the retained two-clip error case studies.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';

const DECK=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime=process.env.RUNTIME_NODE_MODULES||'/Users/lequnchen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')));
const data=JSON.parse(await fs.readFile(path.join(DECK,'assets/error_refresh/data.json'),'utf8'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={ink:'#1B2A41',text:'#2F3B4C',blue:'#0072B2',green:'#2E8B57',orange:'#D55E00',red:'#C0392B',line:'#D5DAE1',panel:'#F4F6F9',muted:'#687789'};
const EMO={angry:'#E5484D',disgusted:'#46A758',fearful:'#8E4EC6',happy:'#F5A524',neutral:'#8B8D98',sad:'#3E63DD',surprised:'#12A594'};
const NAMES=['emotion2vec+ large','MERaLiON-SER-v1','Melati','Vox-Profile Whisper','Vox-Profile WavLM','Qwen2.5-Omni 7B','Qwen2-Audio 7B Instruct','Gemma 4 E4B','GPT Realtime','Gemini 3.8 Flash'];
const audioManifest={};

function text(s,name,value,x,y,w,h,size=18.67,color=C.text,bold=false){
 const t=s.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=value;t.text.style={fontSize:size,typeface:'Calibri',color,bold,verticalAlignment:'top',insets:0,autoFit:'none'};return t;
}
function rect(s,name,x,y,w,h,fill,line=C.line,lw=.8){return s.shapes.add({geometry:'rect',name,position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width:lw}});}
function color(c,model){
 const label=c.predictions[model].label;
 if(label===c.reference)return C.green;
 if(c.listener_majority.split(' or ').includes(label))return C.blue;
 return C.red;
}

async function casePanel(s,c,index){
 const y=index===0?167:403;
 rect(s,`case-${index+1}`,57.6,y,1164,222,'#FFFFFF');
 const cover=await fs.readFile(path.join(DECK,'img/play.png'));
 s.images.add({blob:cover.buffer.slice(cover.byteOffset,cover.byteOffset+cover.byteLength),contentType:'image/png',alt:'Play '+c.id,fit:'contain',position:{left:76,top:y+34,width:50,height:50}});
 text(s,`identity-${index}`,`${c.dataset} · ${c.speaker}`,142,y+12,286,27,15.7,C.ink,true);
 const quote=c.text?`“${c.text.replace(/^"|"$/g,'')}”`:'Sentence text not published';
 text(s,`transcript-${index}`,quote,142,y+39,286,44,17.3,c.text?C.text:C.muted);
 text(s,`acted-${index}`,`Acted: ${c.reference}`,142,y+88,286,26,18.67,C.ink,true);
 const voteEntries=Object.entries(c.listener_votes).sort((a,b)=>b[1]-a[1]);
 const n=voteEntries.reduce((sum,[_,v])=>sum+v,0);
 if(n){
  text(s,`listener-label-${index}`,`Listeners (n=${n})`,142,y+119,286,20,15.2,C.ink,true);
  text(s,`listener-votes-${index}`,voteEntries.map(([label,v])=>`${label} ${v}`).join(', '),142,y+141,286,40,15.2);
  let x=142;
  for(const [label,v] of voteEntries){const w=286*v/n;rect(s,`vote-${index}-${label}`,x,y+184,w,11,EMO[label]||C.muted,'#FFFFFF',.6);x+=w;}
 }else{
  text(s,`no-votes-${index}`,'No published per-clip listener votes',142,y+121,286,45,15.2,C.muted);
 }
 text(s,`clip-${index}`,c.id,76,y+202,352,17,12.8,C.muted);

 const values=[['Model','Prediction','Score'],...data.model_order.map((m,i)=>[NAMES[i],c.predictions[m].label,c.predictions[m].ser_score===null?'—':c.predictions[m].ser_score.toFixed(2)])];
 const table=s.tables.add({rows:11,columns:3,left:452,top:y+7,width:460,height:210,columnWidths:[245,145,70],values});
 table.borders.assign({style:'solid',fill:C.line,width:.5});
 table.cells.block({row:0,column:0,rowCount:11,columnCount:3}).assign({textStyle:{fontSize:15.3,typeface:'Calibri',color:C.ink},margins:{left:6,right:4,top:.2,bottom:.2},anchor:'center'});
 for(let row=0;row<11;row++){
  table.rows[row].height=row===0?22:18.8;
  for(let col=0;col<3;col++){
   const cell=table.getCell(row,col);cell.fill=row===0?C.ink:(row%2===0?C.panel:'#FFFFFF');
   cell.text.style={fontSize:row===0?(col===2?14.2:15.3):15.3,typeface:'Calibri',bold:row===0||col===1,color:row===0?'#FFFFFF':(col===1?color(c,data.model_order[row-1]):C.ink),verticalAlignment:'middle'};
  }
 }
 // A single rule separates dedicated SER from audio LLMs without extra row labels.
 rect(s,`family-separator-${index}`,452,y+7+22+5*18.8,460,.8,C.blue,C.blue,.4);
 rect(s,`insight-panel-${index}`,929,y+9,280,204,C.panel,C.panel,0);
 text(s,`insight-title-${index}`,c.insight_heading,943,y+21,253,30,17.3,C.orange,true);
 text(s,`insight-body-${index}`,c.insight,943,y+57,253,149,18);
 return {path:c.audio_asset,clip_id:c.id};
}

for(let i=0;i<data.slides.length;i++){
 const info=data.slides[i],s=p.slides.add();s.background.fill='#FFFFFF';
 text(s,'section','5 · LISTENING INSPECTIONS',57.6,27,1164,29,14.67,C.blue,true);
 text(s,'title',info.title,57.6,53,1164,66,34.67,C.ink,true);
 text(s,'legend','Green: acted label. Blue: listener majority/tie. Red: neither. Scores are shown only for dedicated SER models.',57.6,130,1164,25,15.7);
 audioManifest[String(i+1)]=[];
 for(let j=0;j<2;j++)audioManifest[String(i+1)].push(await casePanel(s,info.cases[j],j));
 s.speakerNotes.text=[
  `Retained selected-error analysis: ${info.title}. All ten current model predictions are shown for the original two audio examples.`,
  data.interpretation,data.audio_processing,data.provenance_note,
  ...info.cases.map(c=>`Clip ${c.id}: original ${c.audio_path}; retained MP3 ${c.audio_asset}; SHA-256 ${c.audio_sha256}. Corpus acted reference ${c.reference}. ${c.listener_info}. Exact current model outputs: ${JSON.stringify(c.predictions)}. Selected interpretation: ${c.insight}`),
  'Source manifest: assets/error_refresh/data.json. Complete CSV/JSON source paths and SHA-256 digests are recorded there. Source files are read only; no new inference was performed.',
  'The green/reference coloring does not silently replace actor intent with listener perception. Blue means a listener modal label, including each tied maximum. A displayed SER score of 1.00 is a rounded model output, not proof of correctness or cross-model calibration. Audio LLM probability-like one-hot CSV fields are deliberately omitted.',
  ...data.source_artifacts.filter(x=>x.path.includes('/'+info.cases[0].dataset_key+'_')||x.path.includes('/'+info.cases[1].dataset_key+'_')).map(x=>`${x.path} SHA-256 ${x.sha256}`)
 ].join('\n');
}

const build=path.join(DECK,'.inspection-build');
await fs.mkdir(path.join(build,'error-refresh-previews'),{recursive:true});
await (await PresentationFile.exportPptx(p)).save(path.join(build,'error-refresh.pptx'));
await fs.writeFile(path.join(build,'error-refresh-audio.json'),JSON.stringify(audioManifest,null,2)+'\n');
await fs.writeFile(path.join(build,'error-refresh-order.json'),JSON.stringify(data.slides.map(({kind,key,title})=>({kind,key,title})),null,2)+'\n');
for(let i=0;i<p.slides.items.length;i++)await fs.writeFile(path.join(build,'error-refresh-previews',`page-${i+1}.png`),new Uint8Array(await (await p.slides.items[i].export({format:'png',scale:1})).arrayBuffer()));
console.log('Authored five compact error-analysis slides, ten models and two audio covers per slide');
