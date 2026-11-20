// Author the priority slide with editable PowerPoint text and a native table.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
const DECK = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runtime = process.env.RUNTIME_NODE_MODULES || '/Users/lequnchen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const { Presentation, PresentationFile } = await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')));
const p = Presentation.create({slideSize:{width:1280,height:720}});
const s = p.slides.add();
s.background.fill = '#FFFFFF';
const C={ink:'#1B2A41',text:'#2F3B4C',blue:'#0072B2',line:'#D5DAE1',panel:'#F4F6F9',good:'#2E8B57',bad:'#C0392B'};
function text(name,value,x,y,w,h,size=20,color=C.text,bold=false){
 const t=s.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=value;t.text.style={fontSize:size,typeface:'Calibri',color,bold,verticalAlignment:'top',insets:0,autoFit:'none'};return t;
}
text('section','5 · LISTENING INSPECTIONS',57.6,27,1150,29,14.67,C.blue,true);
text('title','Same Words, Different Emotions: API Predictions',57.6,53,1164,62,34.67,C.ink,true);
text('sentence','“Don’t forget a jacket.”',57.6,133,1164,45,30,C.ink,true);
text('setup','CREMA-D, actor 1002. Four emotional takes of one sentence, with voice-only listener ratings.',57.6,183,1164,35,20);
const data=JSON.parse(await fs.readFile(path.join(DECK,'assets/inspections/semantic.json'),'utf8'));
const cases=data.same_sentence;
const values=[['Listen','Acted emotion','Listeners choosing\nacted emotion','Gemini 3.8 Flash','GPT Realtime','emotion2vec+']];
for(const c of cases){const n=Object.values(c.listener_votes).reduce((a,b)=>a+b,0),k=c.listener_votes[c.reference];values.push(['',c.reference,`${k}/${n} (${Math.round(100*k/n)}%)`,c.predictions['gemini-3.8-flash'].label,c.predictions.gpt_realtime.label,c.predictions.emotion2vec_plus_large.label]);}
const table=s.tables.add({rows:5,columns:6,left:57.6,top:241,width:1164,height:326,columnWidths:[78,159,239,237,214,237],values});
table.borders.assign({style:'solid',fill:C.line,width:.8});
table.cells.block({row:0,column:0,rowCount:5,columnCount:6}).assign({textStyle:{fontSize:22,typeface:'Calibri',color:C.ink},margins:{left:12,right:8,top:10,bottom:8},anchor:'center'});
for(let r=0;r<5;r++){
 table.rows[r].height=r===0?62:66;
 for(let col=0;col<6;col++){
  const cell=table.getCell(r,col);cell.fill=r===0?C.ink:(r%2===0?C.panel:'#FFFFFF');
  cell.text.style={fontSize:r===0?20:23,typeface:'Calibri',bold:r===0||col>=3,color:r===0?'#FFFFFF':C.ink,verticalAlignment:'middle'};
  if(r>0&&col>=3){const c=cases[r-1];const m=['gemini-3.8-flash','gpt_realtime','emotion2vec_plus_large'][col-3];cell.text.color=c.predictions[m].label===c.reference?C.good:C.bad;}
 }
}
const audio=[];
for(let i=0;i<4;i++){
 const y=241+62+i*66+10;
 const img=await fs.readFile(path.join(DECK,'img/play.png'));
 s.images.add({name:`audio-cover-${i}`,blob:img.buffer.slice(img.byteOffset,img.byteOffset+img.byteLength),contentType:'image/png',alt:`Play ${cases[i].clip_id}`,fit:'contain',position:{left:73,top:y,width:46,height:46}});
 audio.push({shapeName:`audio-cover-${i}`,path:cases[i].audio_asset,clip_id:cases[i].clip_id});
}
text('observation','Gemini gives the same label to all four takes. GPT does too. The SER model follows all four acted emotions.',57.6,587,1164,33,20,C.ink,true);
text('interpretation','This suggests limited use of vocal emotion cues. Reliance on sentence meaning remains a hypothesis.',57.6,625,1164,28,19);
text('selection','Selected emotional takes. Gemini changes its answer on this actor’s neutral and sad takes of the same sentence.',57.6,656,1164,23,15);
text('source','Saved current-protocol CREMA-D predictions and voice-only listener votes. Green = matches acted emotion, red = differs.',57.6,689,1164,18,12,'#000000');
s.speakerNotes.text='Source: outputs/{gemini-3.8-flash,gpt_realtime,emotion2vec_plus_large}/crema_d_predictions.csv and matching .json; data/crema_d/processedResults/tabulatedVotes.csv, response IDs beginning with 1 (voice only). Exact clips: '+cases.map(c=>c.clip_id).join(', ')+'. The same actor and sentence control speaker identity and lexical content, but these are different performances, not controlled acoustic interventions. Four selected emotional takes have listener support of 6/8, 7/10, 8/9 and 7/10. Gemini predicts happy on all four and GPT Realtime neutral on all four. emotion2vec+ predicts all four acted labels correctly; CREMA-D is documented in its upstream training. Additional takes 1002_DFA_NEU_XX and 1002_DFA_SAD_XX receive neutral and sad from Gemini, while GPT remains neutral. Therefore the selected pattern does not show universal invariance, nor prove semantic reliance or rule out a label prior. No new model inference. Audio is full-length mono 16 kHz, without trimming or loudness normalization.';
const build=path.join(DECK,'.inspection-build');await fs.mkdir(build,{recursive:true});
await (await PresentationFile.exportPptx(p)).save(path.join(build,'semantic-base.pptx'));
await fs.writeFile(path.join(build,'semantic-audio.json'),JSON.stringify(audio,null,2));
await fs.writeFile(path.join(build,'semantic.png'),new Uint8Array(await (await s.export({format:'png',scale:1.2})).arrayBuffer()));
console.log('Authored semantic-base.pptx');
