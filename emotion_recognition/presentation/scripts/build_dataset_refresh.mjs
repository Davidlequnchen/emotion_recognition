// Dataset overview, label mapping and one editable listening page per corpus.
// All source notes remain in speaker notes; visible content ends above 630 px.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';

const DECK=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const BUILD=path.join(DECK,'.inspection-build');
const runtime=process.env.RUNTIME_NODE_MODULES||'/Users/lequnchen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')));
const data=JSON.parse(await fs.readFile(path.join(DECK,'assets/dataset_refresh/data.json'),'utf8'));
const datasets=data.datasets;
if(datasets.length!==16)throw new Error('Expected sixteen dataset records');
const byKey=new Map(datasets.map(x=>[x.key,x]));
const order=data.overview_order||['meld','emotiontalk','dailytalk','multidialog','crema_d','ravdess','savee','jl_corpus','tess','mead','mess','mcae_spps','thai_ser','nemo','emovo','emozionalmente'];
const models=data.model_order||['emotion2vec_plus_large','meralion_ser_v1','melati_emotion','vox_profile_whisper','vox_profile_wavlm','qwen2_5_omni_7b','qwen2_audio_7b','gemma4_e4b','gpt_realtime','gemini-3.8-flash'];
const modelNames={emotion2vec_plus_large:'emotion2vec+ large',meralion_ser_v1:'MERaLiON-SER-v1',melati_emotion:'Melati',vox_profile_whisper:'Vox-Profile Whisper',vox_profile_wavlm:'Vox-Profile WavLM',qwen2_5_omni_7b:'Qwen2.5-Omni 7B',qwen2_audio_7b:'Qwen2-Audio Instruct',gemma4_e4b:'Gemma 4 E4B',gpt_realtime:'GPT Realtime','gemini-3.8-flash':'Gemini 3.8 Flash'};
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={ink:'#1B2A41',text:'#2F3B4C',muted:'#586474',blue:'#0072B2',line:'#D5DAE1',panel:'#F4F6F9',good:'#2E8B57',bad:'#C0392B',orange:'#D55E00'};
const audio={},slideOrder=[];

function text(s,name,value,x,y,w,h,size=20,color=C.text,bold=false){
 const t=s.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=String(value??'');t.text.style={fontSize:size,typeface:'Calibri',color,bold,verticalAlignment:'top',insets:0,autoFit:'none'};
 return t;
}
function line(s,x,y,w,color=C.line,width=.8){s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:width},fill:color,line:{fill:color,width:0}});}
function base(title,kind,dataset){
 const s=p.slides.add();s.background.fill='#FFFFFF';
 text(s,'section','2 · PUBLIC DATASETS',57.6,27,1164,28,14.67,C.blue,true);
 text(s,'title',title,57.6,53,1164,65,34.67,C.ink,true);
 slideOrder.push({slide:p.slides.items.length,kind,title,...(dataset?{dataset}: {})});
 return s;
}
function clean(value){return String(value??'').replace(/\[\/?over\/?\]|【\/?over\/?】/gi,'').replace(/\[interrupted\]/gi,'…').replace(/\s+/g,' ').trim();}
function noteValue(value){return typeof value==='string'?value:JSON.stringify(value,null,2);}
function notes(ds){return `${ds.name}\n${noteValue(ds.source_notes)}\nParallel-text definition: ${ds.parallel_detail||ds.parallel_text}.\nReference labels: ${ds.label_detail||ds.label_summary}.\nSpeaker scope: ${ds.speaker_detail||ds.speakers}.\nAll counts describe the current evaluated cohort unless a different source scope is explicitly stated.`;}
function truth(value){return value===true||['true','yes','1'].includes(String(value).toLowerCase());}
function table(s,values,x,y,width,columnWidths,rowHeights,fontSize=18.5){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height:rowHeights.reduce((a,b)=>a+b,0),columnWidths,values});
 t.borders.assign({style:'solid',fill:C.line,width:.7});
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({textStyle:{fontSize,typeface:'Calibri',color:C.ink},margins:{left:9,right:7,top:2,bottom:2},anchor:'center'});
 for(let r=0;r<values.length;r++){
  t.rows[r].height=rowHeights[r];
  for(let c=0;c<values[0].length;c++){
   const cell=t.getCell(r,c);cell.fill=r===0?C.ink:(r%2===0?C.panel:'#FFFFFF');
   cell.text.style={fontSize:r===0?17.3:fontSize,typeface:'Calibri',color:r===0?'#FFFFFF':C.ink,bold:r===0||c===0,verticalAlignment:'middle'};
  }
 }
 return t;
}
function languageShort(value){return /English/i.test(value)?'English':String(value).replace(/ Chinese/g,'');}
function speakersShort(ds){
 const value=String(ds.speakers??'—');
 if(/role|character/i.test(value)){const m=value.match(/\d+/);return `${m?.[0]||''} role IDs`.trim();}
 if(ds.speakers_evaluated!==undefined)return `${ds.speakers_evaluated} speakers`;
 const m=value.match(/\d+/);return m?`${m[0]} speakers`:value;
}
function parallelSummary(ds){return ds.key==='mead'?'~30/emo + 40 neutral, partial':ds.parallel_text;}

// One complete overview, with separate clip and speaker columns for compact rows.
{
 const s=base('Sixteen Public Benchmark Datasets','overview');
 text(s,'cohort','16 datasets and 67,062 recordings per model, with recording design and label sources',57.6,120,1164,30,20);
 const current=order.map(k=>byKey.get(k));
 if(current.some(x=>!x))throw new Error('Overview ordering references an absent dataset');
 const values=[['Dataset','Language','Recording','Clips','Speakers','Parallel text','Reference labels / raters'],...current.map(ds=>[
  ds.name.replace(/ \(.*?\)$/,''),languageShort(ds.language),ds.type==='dialogue'?'dialogue':'standalone',
  Number(ds.clips).toLocaleString('en-US'),speakersShort(ds).replace(/ speakers$/,''),parallelSummary(ds),ds.label_summary
 ])];
 const t=table(s,values,57.6,158,1164,[165,100,100,78,112,250,359],[42,...Array(16).fill(27)],17.3);
 t.cells.block({row:0,column:0,rowCount:17,columnCount:7}).assign({margins:{left:7,right:5,top:2,bottom:2}});
 for(let r=1;r<17;r++){
  for(const c of [1,2,5,6])t.getCell(r,c).text.style={fontSize:16.6,typeface:'Calibri',color:C.ink,verticalAlignment:'middle'};
  for(const c of [3,4])t.getCell(r,c).text.style={fontSize:17.3,typeface:'Calibri',color:C.ink,verticalAlignment:'middle',alignment:'center'};
 }
 s.speakerNotes.text=`All sixteen adopted datasets cover 67,062 recordings per model under the completed ten-model benchmark. This overview uses evaluated clip/speaker scope. A source-corpus speaker count may therefore be larger. In MELD the count is published role/speaker labels, including grouped roles, rather than an asserted count of distinct people. Parallel text means a controlled shared sentence design across emotions, not merely the existence of transcripts. Dialogue denotes recording source; inference still uses one utterance without history.\n${current.map(notes).join('\n\n')}`;
}

// One complete matrix, with raw-label exceptions explained beside the table.
{
 const s=base('Label Mapping: All Sixteen Benchmark Datasets','label_mapping');
 text(s,'mapping-scope','Seven project categories, with each corpus’s label names and scoring rules retained',57.6,132,1164,31,20);
 const emotions=['angry','disgusted','fearful','happy','neutral','sad','surprised'];
 function mark(ds,e){
  const classes=new Set(ds.classes||[]);
  if(ds.key==='jl_corpus'){
   if(e==='happy')return '●⁴';
   if(e==='surprised')return '⁴';
  }
  if(!classes.has(e))return '—';
  if(ds.key==='ravdess'&&e==='neutral')return '●¹';
  if(ds.key==='mess'&&e==='neutral')return 'calm²';
  if(ds.key==='tess'&&e==='surprised')return '●³';
  return '●';
 }
 const values=[['Dataset',...emotions],...order.map(k=>{const ds=byKey.get(k);return [ds.name.replace(/ \(.*?\)$/,''),...emotions.map(e=>mark(ds,e))];})];
 const t=table(s,values,57.6,176,868,[196,...Array(7).fill(96)],[44,...Array(16).fill(25.5)],17.3);
 t.cells.block({row:0,column:0,rowCount:17,columnCount:8}).assign({margins:{left:7,right:5,top:2,bottom:2}});
 for(let r=0;r<17;r++)for(let c=1;c<8;c++){
  const cell=t.getCell(r,c);cell.text.alignment='center';
  if(r>0){cell.text.color=values[r][c]==='—'?C.muted:C.blue;cell.text.bold=true;}
 }
 text(s,'legend-heading','Mapping rules',954,180,267,30,21,C.ink,true);
 const entries=[
  ['●','Corpus label retained'],
  ['¹ RAVDESS','calm merges into neutral'],
  ['² MESS','calm maps to neutral'],
  ['³ TESS','pleasant surprise maps\nto surprised'],
  ['⁴ JL-Corpus','excited accepts either\nhappy or surprised'],
  ['Excluded categories','THAI-SER: frustrated\nMEAD: contempt']
 ];
 let y=222;
 for(const [head,body] of entries){
  text(s,`legend-${y}`,head,954,y,267,25,17.3,C.blue,true);
  text(s,`rule-${y}`,body,954,y+25,267,47,17.3,C.text);
  y+=head==='●'?54:67;
 }
 s.speakerNotes.text=`The table includes all sixteen evaluated datasets. A dot denotes a corpus category mapped identically into the project’s seven canonical classes. Blank coverage uses an em dash. RAVDESS neutral includes neutral and calm. MESS calm is scored as neutral. TESS pleasant surprise is scored as surprised. JL-Corpus has five primary raw emotions, including excited: excited is accepted as happy or surprised, but does not create an independent surprise gold category. Native model unknown/other classes are excluded from candidate argmax; invalid generated answers remain errors. Dataset-specific exclusions and exact candidate lists are read from the stored prediction run metadata.\n${order.map(k=>notes(byKey.get(k))).join('\n\n')}`;
}

async function playPoster(s,x,y,c){
 const bytes=await fs.readFile(path.join(DECK,'img/play.png'));
 s.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:'image/png',alt:`Play ${c.id}`,fit:'contain',position:{left:x,top:y,width:44,height:44}});
}
function shortenText(value,maxLatin=65,maxCjk=33){
 const v=clean(value);const max=/[\u3000-\u9fff]/.test(v)?maxCjk:maxLatin;
 return v.length>max?v.slice(0,max-1)+'…':v;
}
function trainingShort(value){return String(value??'not established').replaceAll('emotion2vec+ large','emotion2vec+').replaceAll('MERaLiON-SER-v1','MERaLiON').replaceAll('melati_emotion','Melati');}
function emotionValue(ds){
 const exceptions={crema_d:'6 classes (no surprised)',mess:'4: angry, neutral, happy, sad',ravdess:'8 source labels mapped to 7',tess:'7 (pleasant surprise mapped)',jl_corpus:'5 primary labels (incl. excited)',mead:'7 (contempt excluded)',nemo:'6 (no disgusted)',thai_ser:'4 (frustrated excluded)'};
 return exceptions[ds.key]||'All 7 canonical emotions';
}

for(const ds of datasets){
 if(ds.cases.length!==4)throw new Error(`${ds.key}: expected four examples`);
 for(const c of ds.cases)for(const m of models)if(!c.predictions?.[m])throw new Error(`${ds.key}/${c.id}: missing ${m}`);
 const s=base('Listening Example: '+ds.name.replace(/ \(gold_both\)$/,''),'dataset_example',ds.key);
 const sharedText=new Set(ds.cases.map(c=>clean(c.text))).size===1&&clean(ds.cases[0].text);
 const sameSpeaker=new Set(ds.cases.map(c=>String(c.speaker))).size===1;
 const caption=sharedText&&sameSpeaker?`Same speaker and text: “${clean(ds.cases[0].text)}”`:clean(ds.matching_caption||'Four selected recordings from the current evaluated cohort.');
 text(s,'example-caption',caption.length>205?caption.slice(0,202)+'…':caption,57.6,132,1164,43,19).text.autoFit='shrinkText';
 const facts=[
  ['Language',ds.language],['Recording',ds.type==='dialogue'?'Dialogue'+(ds.recording_detail?` (${ds.recording_detail})`:''):'Standalone utterances'],
  ['Speakers',ds.speakers],['Clips',Number(ds.clips).toLocaleString('en-US')],
  ['Parallel text',ds.key==='mead'?'~30 per emotion + 40 neutral; partly shared':ds.parallel_text],['Emotions',emotionValue(ds)],
  ['Labels',String(ds.label_summary).replace(/^\d+\s+classes[;,:· ]*/i,'')],['Licence',ds.licence],['Training',trainingShort(ds.training_involvement)]
 ];
 facts.forEach(([label,value],i)=>{
  const y=185+i*42;
  text(s,`fact-label-${i}`,label,57.6,y+2,120,39,17.3,C.ink,true);
  text(s,`fact-value-${i}`,value,184,y+2,378,40,18.7,C.ink).text.autoFit='shrinkText';
  line(s,57.6,y+41,504);
 });
 const values=[['Model',...ds.cases.map((c,i)=>`${String.fromCharCode(65+i)}\n${c.reference}`)],...models.map(m=>[modelNames[m]||m,...ds.cases.map(c=>c.predictions[m].label==='other answer'?'invalid':c.predictions[m].label)])];
 const t=table(s,values,600,185,622,[210,103,103,103,103],[104,...Array(10).fill(33.8)],17.6);
 t.cells.block({row:0,column:0,rowCount:11,columnCount:5}).assign({margins:{left:8,right:6,top:4,bottom:4}});
 for(let c=1;c<5;c++){
  const cell=t.getCell(0,c);cell.text.style={fontSize:18.7,typeface:'Calibri',bold:true,color:'#FFFFFF',alignment:'center',verticalAlignment:'top'};
  cell.text.insets={left:2,right:2,top:7,bottom:0};
 }
 for(let r=1;r<11;r++)for(let c=1;c<5;c++){
  const cell=t.getCell(r,c);cell.text.style={fontSize:17.6,typeface:'Calibri',bold:true,color:truth(ds.cases[c-1].predictions[models[r-1]].correct)?C.good:C.bad,alignment:'center',verticalAlignment:'middle'};
 }
 line(s,600,185+104+5*33.8,622,C.blue,1.5);
 const thisAudio=[];
 for(let i=0;i<4;i++){
  const c=ds.cases[i];
  await playPoster(s,810+i*103+29.5,239,c);
  thisAudio.push({path:c.audio_asset,clip_id:c.id});
  const original=clean(c.text);
  const votes=Object.entries(c.listener_votes||{}).filter(([_,n])=>Number(n)>0).sort((a,b)=>b[1]-a[1]);
  let heard='';
  if(votes.length){
   const total=votes.reduce((sum,[_,n])=>sum+Number(n),0),top=Number(votes[0][1]);
   const labels=votes.filter(([_,n])=>Number(n)===top).map(([label])=>label);
   heard=labels.length===1?`listeners: ${labels[0]} ${top}/${total}`:`listener tie: ${labels.join('/')} (${top} each/${total})`;
  }
  let detail;
  if(sharedText&&sameSpeaker){detail=heard||`${c.id} (${Number(c.duration_s).toFixed(1)} s)`;}
  else{
   const displayId=ds.key==='mead'&&c.audio_path?c.audio_path.split('/').slice(-4).join('/'):c.id;
   detail=(original?'“'+shortenText(original,heard?42:65,heard?22:33)+'”':displayId)+(heard?' · '+heard:'');
  }
  const label=`${String.fromCharCode(65+i)}  ${detail}`;
  const captionShape=text(s,`clip-text-${i}`,label,57.6,566+i*15.5,504,17,14,C.text);
  captionShape.text.wrap='none';captionShape.text.autoFit='shrinkText';
 }
 audio[String(p.slides.items.length)]=thisAudio;
 const exact=ds.cases.map((c,i)=>`${String.fromCharCode(65+i)}: ${c.id}. Reference: ${c.reference}. Speaker: ${c.speaker}. Text: ${c.text||'not available in local metadata'}. Duration: ${c.duration_s}s. Listener votes: ${JSON.stringify(c.listener_votes||{})}. Listener majority: ${c.listener_majority||'not available'}. Predictions: ${JSON.stringify(c.predictions)}.`).join('\n');
 s.speakerNotes.text=`${notes(ds)}\nSelection: ${ds.matching_caption}. The four recordings are illustrative examples, not a representative estimate of performance. Every prediction uses the same saved clip for each model. Green matches the corpus reference and red differs; listener votes are not silently substituted for the intended/reference label. Displayed invalid means the stored string “other answer”. The first five rows are dedicated SER and the final five are audio LLMs.\n${exact}\nAudio files: ${ds.cases.map(c=>c.audio_asset).join(', ')}. Published examples retain their source terms. Audio provenance and prediction-source hashes: assets/dataset_refresh/data.json.`;
}

if(p.slides.items.length!==18)throw new Error('Expected one overview, one label map and sixteen listening pages');
await fs.mkdir(path.join(BUILD,'dataset-refresh-previews'),{recursive:true});
await (await PresentationFile.exportPptx(p)).save(path.join(BUILD,'dataset-refresh.pptx'));
await fs.writeFile(path.join(BUILD,'dataset-refresh-audio.json'),JSON.stringify(audio,null,2));
await fs.writeFile(path.join(BUILD,'dataset-refresh-order.json'),JSON.stringify(slideOrder,null,2));
for(const [i,s] of p.slides.items.entries()){
 const png=await s.export({format:'png',scale:1});
 await fs.writeFile(path.join(BUILD,'dataset-refresh-previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log('Authored eighteen dataset refresh pages with sixty-four listening examples');
