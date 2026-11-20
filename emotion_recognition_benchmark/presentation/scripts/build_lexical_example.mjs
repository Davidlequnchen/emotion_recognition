// Continue the preceding same-sentence slide with a more affective sentence.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const DECK=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime=process.env.RUNTIME_NODE_MODULES||'/Users/lequnchen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')));
const p=Presentation.create({slideSize:{width:1280,height:720}});
const s=p.slides.add();s.background.fill='#FFFFFF';
const C={ink:'#1B2A41',text:'#2F3B4C',blue:'#0072B2',line:'#D5DAE1',panel:'#F4F6F9',good:'#2E8B57',bad:'#C0392B'};
const data=JSON.parse(await fs.readFile(path.join(DECK,'assets/inspections/lexical/manifest.json'),'utf8'));
function text(name,value,x,y,w,h,size=20,color=C.text,bold=false){
 const t=s.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=value;t.text.style={fontSize:size,typeface:'Calibri',color,bold,verticalAlignment:'top',insets:0,autoFit:'none'};return t;
}
text('section','5 · LISTENING INSPECTIONS',57.6,27,1164,29,14.67,C.blue,true);
text('title','Emotionally Loaded Words: Unchanged API Predictions',57.6,53,1164,62,34.67,C.ink,true);
text('sentence','“Work hard or you lose.”',57.6,133,1164,45,30,C.ink,true);
text('setup','JL-Corpus, speaker male2. Same sentence, session b and repetition 1, acted in five emotions.',57.6,183,1164,35,20);
const values=[['Listen','Acted emotion','Take ID','Gemini 3.8 Flash','GPT Realtime','emotion2vec+']];
for(const c of data.cases)values.push(['',c.reference+(c.reference==='excited'?'*':''),c.clip_id,...['gemini-3.8-flash','gpt_realtime','emotion2vec_plus_large'].map(m=>c.predictions[m].label)]);
const table=s.tables.add({rows:6,columns:6,left:57.6,top:241,width:1164,height:326,columnWidths:[78,159,239,237,214,237],values});
table.borders.assign({style:'solid',fill:C.line,width:.8});
table.cells.block({row:0,column:0,rowCount:6,columnCount:6}).assign({textStyle:{fontSize:22,typeface:'Calibri',color:C.ink},margins:{left:12,right:8,top:7,bottom:7},anchor:'center'});
for(let row=0;row<6;row++){
 table.rows[row].height=row===0?60:53.2;
 for(let col=0;col<6;col++){
  const cell=table.getCell(row,col);cell.fill=row===0?C.ink:(row%2===0?C.panel:'#FFFFFF');
  cell.text.style={fontSize:row===0?20:(col===2?17:22),typeface:'Calibri',bold:row===0||col>=3,color:row===0?'#FFFFFF':C.ink,verticalAlignment:'middle'};
  if(row>0&&col>=3){const model=['gemini-3.8-flash','gpt_realtime','emotion2vec_plus_large'][col-3];cell.text.color=data.cases[row-1].predictions[model].correct?C.good:C.bad;}
 }
}
const audio=[];
for(let i=0;i<data.cases.length;i++){
 const img=await fs.readFile(path.join(DECK,'img/play.png'));
 s.images.add({blob:img.buffer.slice(img.byteOffset,img.byteOffset+img.byteLength),contentType:'image/png',alt:'Play '+data.cases[i].clip_id,fit:'contain',position:{left:76,top:241+60+i*53.2+5.6,width:42,height:42}});
 audio.push({path:data.cases[i].audio_asset,clip_id:data.cases[i].clip_id});
}
text('observation','Both APIs predict angry for every take, including the neutral, happy and sad performances.',57.6,587,1164,33,21,C.ink,true);
text('interpretation','The wording suggests pressure or a threat. This pattern is consistent with lexical bias, whose cause is not isolated here.',57.6,625,1164,28,18.7);
text('selection','*excited accepts happy or surprised. Selected same-speaker series. Acted labels without per-clip listener votes.',57.6,656,1164,23,15);
text('source','Saved current-protocol JL-Corpus predictions. Across all 80 takes of this sentence, both APIs output angry together on 33.',57.6,689,1164,18,12,'#000000');
s.speakerNotes.text=`${data.selection}\n${data.ground_truth}\nSentence: ${data.sentence}. Interpreting the conditional imperative as pressure or a threat is a semantic reading, not an independently scored text-only model condition. The neutral, happy and sad performances are classified as angry by both APIs. Their identical predictions across this series are consistent with lexical bias, but the stored label outputs do not isolate semantic causation, rule out a general anger prior, or establish the emotion that a new listener will hear. A text-only/audio-only intervention and independent perception ratings would be needed for that claim.\nThe full five-emotion male2/12b/repetition-1 group is shown, including excited, which accepts happy or surprised under the existing protocol. emotion2vec+ gets the four primary emotions right and mislabels excited as angry. JL-Corpus is documented in emotion2vec+'s upstream training, so its success is not independent evidence of generalization.\nCurrent protocol and source hashes: assets/inspections/lexical/manifest.json. ${data.audio_processing}\nSources: ${data.sources.map(x=>x.path+' SHA256 '+x.sha256).join('\n')}\nExact clips: ${data.cases.map(x=>x.audio_path+' SHA256 '+x.source_audio_sha256).join('\n')}\nDataset credit: Jesin James, Li Tian and Catherine Watson, An Open Source Emotional Speech Corpus for Human Robot Interaction Applications, Interspeech 2018. Local author-released raw JL-Corpus recordings and transcripts, CC0 1.0. Source: data/jl_corpus/README.md.\nAcross the full 80 stored recordings of this sentence, both APIs output angry together in 33 cases. This page intentionally selects a constant-output group and does not estimate the overall failure rate. No new model inference.`;
const build=path.join(DECK,'.inspection-build');await fs.mkdir(build,{recursive:true});
await (await PresentationFile.exportPptx(p)).save(path.join(build,'lexical-base.pptx'));
await fs.writeFile(path.join(build,'lexical-audio.json'),JSON.stringify(audio,null,2));
await fs.writeFile(path.join(build,'lexical.png'),new Uint8Array(await (await s.export({format:'png',scale:1.2})).arrayBuffer()));
console.log('Authored lexical-base.pptx');
