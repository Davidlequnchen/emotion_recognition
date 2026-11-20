// Editable inspection slides; original deck parts are preserved by merge_inspections.py.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const DECK=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime=process.env.RUNTIME_NODE_MODULES||'/Users/lequnchen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')));
const data=JSON.parse(await fs.readFile(path.join(DECK,'assets/inspections/manifest.json'),'utf8'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={ink:'#1B2A41',text:'#2F3B4C',blue:'#0072B2',line:'#D5DAE1',panel:'#F4F6F9',good:'#2E8B57',bad:'#C0392B'};
const NAMES=['emotion2vec+ large','MERaLiON-SER-v1','Melati','Vox-Profile Whisper','Vox-Profile WavLM','Qwen2.5-Omni 7B','Qwen2-Audio 7B','Gemma 4 E4B','GPT Realtime','Gemini 3.8 Flash'];
const audioMap={};
function text(s,name,value,x,y,w,h,size=20,color=C.text,bold=false){
 const t=s.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=value;t.text.style={fontSize:size,typeface:'Calibri',color,bold,verticalAlignment:'top',insets:0,autoFit:'none'};return t;
}
function rect(s,x,y,w,h,fill=C.panel,line=C.line){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width:.8}});}
function base(title){const s=p.slides.add();s.background.fill='#FFFFFF';text(s,'section','5 · LISTENING INSPECTIONS',57.6,27,1164,29,14.67,C.blue,true);text(s,'title',title,57.6,53,1164,66,34.67,C.ink,true);return s;}
async function pic(s,file,x,y,w,h,alt){const b=await fs.readFile(path.join(DECK,file));return s.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:'image/png',alt,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
const overview=base('Listening Inspections: All Ten Models, All Sixteen Datasets');
for(const [i,num,label] of [[0,'16','datasets'],[1,'32','selected recordings'],[2,'10','predictions per recording']]){
 text(overview,`stat-${i}`,num,57.6+i*390,145,350,70,52,C.blue,true);
 text(overview,`caption-${i}`,label,57.6+i*390,218,350,35,23,C.ink);
}
const overviewRows=[
 ['Coverage','Two playable examples from each current dataset cohort. Each example shows five dedicated SER models, three local audio LLMs and both API models.'],
 ['Selection','A maximizes the SER-minus-LLM correct-count gap. B maximizes the opposite gap using a different reference emotion and speaker. These deliberate contrasts are illustrative.'],
 ['Reading the page','Green labels match the dataset reference and red labels differ. Listener ratings appear separately when available. JL-Corpus “excited” accepts happy or surprised.'],
 ['Matched-sentence test','One CREMA-D actor says the same sentence in different emotions. Compare Gemini 3.8 Flash and GPT Realtime with the acted labels and listener votes.']
];
overviewRows.forEach(([label,body],i)=>{const y=292+i*86;rect(overview,57.6,y,5,66,C.blue,C.blue);text(overview,`label-${i}`,label,80,y,240,33,23,C.ink,true);text(overview,`body-${i}`,body,325,y,882,72,21);});
text(overview,'retained','The earlier five-SER listener-vote case studies follow this complete ten-model inspection set.',57.6,650,1164,29,19);
text(overview,'source','All examples use saved current-protocol predictions. No new inference or training. Audio remains subject to each corpus’s terms.',57.6,688,1164,18,12,'#000000');
overview.speakerNotes.text=data.selection_rule+'\nCurrent cohort alignment and accuracy were recomputed and checked for all 160 model-dataset runs before selection. Source snapshot: research_paper/output/cross_dataset_analysis.json, SHA-256 '+data.analysis_sha256+'.\nThe same two examples are scored by every model. The 32 examples deliberately emphasize contrasts and cannot estimate performance. Full-cohort tables earlier in the deck remain the quantitative results. Full source hashes and clip paths: assets/inspections/manifest.json. The earlier listener-vote cases retain their historical five-model scope. '+data.audio_processing;

for(const ds of data.datasets){
 const display=ds.name.replace(' (gold_both)','');
 const s=base('Listening Inspection: '+display);
 const labels={meld:'video/transcript annotation',emotiontalk:'audio-perception gold',dailytalk:'published ECSS category',multidialog:'prescribed performance emotion'};
 text(s,'cohort',`${ds.language} · ${ds.clips.toLocaleString('en-US')} clips in the current cohort · Reference: ${labels[ds.key]||'intended emotion'}`,57.6,130,1164,30,19);
 text(s,'families','Five dedicated SER models, followed by five audio LLMs',650,163,572,24,17);
 const values=[['Model','Clip A','Clip B'],...data.model_order.map((m,i)=>[NAMES[i],ds.cases[0].predictions[m].label,ds.cases[1].predictions[m].label])];
 const t=s.tables.add({rows:11,columns:3,left:650,top:191,width:572,height:410,columnWidths:[294,139,139],values});
 t.borders.assign({style:'solid',fill:C.line,width:.8});
 t.cells.block({row:0,column:0,rowCount:11,columnCount:3}).assign({textStyle:{fontSize:20,typeface:'Calibri',color:C.ink},margins:{left:11,right:6,top:7,bottom:7},anchor:'center'});
 for(let row=0;row<11;row++){
  t.rows[row].height=row===0?40:37;
  for(let col=0;col<3;col++){
   const cell=t.getCell(row,col);cell.fill=row===0?C.ink:(row%2===0?C.panel:'#FFFFFF');
   cell.text.style={fontSize:row===0?20:19.4,typeface:'Calibri',bold:row===0||col>0,color:row===0?'#FFFFFF':C.ink,verticalAlignment:'middle'};
   if(row>0&&col>0){cell.text.color=ds.cases[col-1].predictions[data.model_order[row-1]].correct?C.good:C.bad;}
  }
 }
 // Intentional family separator after the fifth model.
 rect(s,650,191+40+5*37,572,1.6,C.blue,C.blue);
 const thisAudio=[];
 for(const [i,c] of ds.cases.entries()){
  const y=181+i*242;
  rect(s,57.6,y,554,229,'#FFFFFF');
  text(s,`case-${i}`,`${i===0?'A':'B'} · Speaker ${c.speaker} · ${c.duration_s.toFixed(1)} s`,74,y+11,520,25,19,C.ink,true);
  const id=ds.key==='mead'?c.audio_path.split('/').slice(-4).join('/'):c.clip_id;
  text(s,`clip-id-${i}`,id,74,y+39,520,20,id.length>50?12.5:14);
  const clean=c.text.replace(/\[\/?over\/?\]/g,'').trim();
  const size=/[\u3000-\u9fff]/.test(clean)?18:(clean.length>100?18:20);
  text(s,`transcript-${i}`,clean?`“${clean}”`:'Transcript unavailable in the local metadata',74,y+66,520,60,size,clean?C.text:'#586474',!clean?false:false);
  await pic(s,'img/play.png',74,y+130,43,43,'Play '+c.clip_id);
  await pic(s,c.waveform_asset,130,y+130,464,43,'Waveform of '+c.clip_id);
  thisAudio.push({path:c.audio_asset,clip_id:c.clip_id,picture_indices:i*2});
  text(s,`reference-${i}`,`Reference: ${c.reference}${ds.accept[c.reference]?` (${ds.accept[c.reference].join(' or ')})`:''}`,74,y+178,520,25,20,C.ink,true);
  let listener='';
  if(Object.keys(c.listener_votes).length){
   const votes=Object.entries(c.listener_votes).sort((a,b)=>b[1]-a[1]);
   const total=votes.reduce((a,[_,v])=>a+v,0);
   listener=`Listeners (n=${total}): `+votes.map(([e,v])=>`${e} ${v}`).join(', ');
  }else if(c.listener_majority){listener=`Listener majority: ${c.listener_majority}${c.extra?' ('+c.extra+')':''}`;}
  else{listener=c.extra||'Per-clip listener ratings not available';}
  text(s,`listeners-${i}`,listener,74,y+206,520,22,listener.length>68?13.5:15);
 }
 text(s,'count-a',`A: SER ${ds.cases[0].ser_correct}/5 correct, audio LLMs ${ds.cases[0].llm_correct}/5`,650,620,572,28,20,C.ink,true);
 text(s,'count-b',`B: SER ${ds.cases[1].ser_correct}/5 correct, audio LLMs ${ds.cases[1].llm_correct}/5`,650,653,572,28,20,C.ink,true);
 text(s,'source','Selected contrasts, not a performance estimate. Green = matches reference, red = differs. Research inspection copy.',57.6,691,1164,18,12,'#000000');
 audioMap[String(p.slides.items.length)]=thisAudio;
 const paths=data.model_order.map(m=>`outputs/${m}/${ds.key}_predictions.csv`).join('\n');
 s.speakerNotes.text=`Dataset: ${ds.name}. Reference definition: ${ds.ground_truth_description}.\n${data.selection_rule}\n${data.audio_processing}\nSources:\n${paths}\n`+ds.cases.map((c,i)=>`Clip ${i===0?'A':'B'}: ${c.audio_path}; original SHA-256 ${c.audio_sha256}. Selection: ${c.selection}. Exact transcript: ${c.text}. Listener majority: ${c.listener_majority||'not available'}. Listener votes: ${JSON.stringify(c.listener_votes)}. ${c.extra}.`).join('\n')+'\nListener votes are not substituted for the benchmark reference. Correctness uses the stored dataset-specific scoring rule, including composite labels and invalid answers. Model outputs are label predictions, not calibrated confidence scores. Dataset source terms and credits are retained in scripts/common/eval_datasets.py:SOURCES and data/<dataset>/README.md. This copy is for internal research inspection.';
}
const build=path.join(DECK,'.inspection-build');await fs.mkdir(path.join(build,'previews'),{recursive:true});
await (await PresentationFile.exportPptx(p)).save(path.join(build,'inspections-base.pptx'));
await fs.writeFile(path.join(build,'inspections-audio.json'),JSON.stringify(audioMap,null,2));
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.items[i];
 await fs.writeFile(path.join(build,'previews',`inspection-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await (await s.export({format:'png',scale:1})).arrayBuffer()));
}
console.log('Authored 17 inspection slides');
