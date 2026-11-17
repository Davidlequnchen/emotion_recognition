// Native editable model inventory and audio-LLM architecture continuations.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const DECK=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime=process.env.RUNTIME_NODE_MODULES||'/Users/lequnchen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')));
const facts=JSON.parse(await fs.readFile(path.join(DECK,'assets/model_refresh/facts.json'),'utf8'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={ink:'#1B2A41',text:'#2F3B4C',blue:'#0072B2',green:'#2E8B57',orange:'#D55E00',line:'#D5DAE1',panel:'#F4F6F9',enc:'#EEF4FA',bridge:'#E2F2EA',llm:'#FCE9D9',muted:'#687789'};
function text(s,name,value,x,y,w,h,size=18.67,color=C.text,bold=false,align='left'){
 const t=s.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=value;t.text.style={fontSize:size,typeface:'Calibri',color,bold,alignment:align,verticalAlignment:'top',insets:0,autoFit:'none'};return t;
}
function base(title,subtitle){
 const s=p.slides.add();s.background.fill='#FFFFFF';
 text(s,'section','3 · BENCHMARKING MODELS',57.6,27,1164,29,14.67,C.blue,true);
 text(s,'title',title,57.6,53,1164,66,34.67,C.ink,true);
 if(subtitle)text(s,'subtitle',subtitle,57.6,134,1164,34,20,C.ink);
 return s;
}
function box(s,name,label,x,y,w,h,kind='io',size=16){
 const color=kind==='enc'?C.blue:kind==='bridge'?C.green:kind==='llm'?C.orange:C.line;
 const fill=kind==='enc'?C.enc:kind==='bridge'?C.bridge:kind==='llm'?C.llm:C.panel;
 const b=s.shapes.add({geometry:'roundRect',name,position:{left:x,top:y,width:w,height:h},fill,line:{fill:color,width:1.67},borderRadius:7});
 b.text=label;b.text.style={fontSize:size,typeface:'Calibri',color:kind==='muted'?C.muted:C.ink,alignment:'center',verticalAlignment:'middle',insets:{left:6,right:6,top:5,bottom:5},autoFit:'none'};return b;
}
function connect(s,a,b,from='right',to='left'){
 return s.shapes.connect(a,b,{kind:'straight',fromSide:from,toSide:to,line:{style:'solid',fill:C.ink,width:1.67},tail:{type:'triangle',width:'sm',length:'sm'}});
}
function notes(s,key,extra){
 const f=facts.models[key];s.speakerNotes.text=extra+'\nOfficial sources:\n'+facts.official_sources[key].join('\n')+'\nLocal checkpoint evidence: models/'+key+'/config.json and safetensors headers, recorded without loading tensors in assets/model_refresh/facts.json. Sources and SHA-256:\n'+f.sources.map(x=>x.path+' '+x.sha256).join('\n')+'\nCheckpoint parameter elements: '+f.parameter_elements_in_checkpoint+'.\nNo new inference or training was performed to prepare this slide.';
}
function detail(s,i,label,body){const y=474+i*49;text(s,'detail-label-'+i,label,57.6,y,225,43,18.67,C.ink,true);text(s,'detail-body-'+i,body,292,y,928,45,18.67);}
function pipeline(s,labels){
 const xs=[57.6,177.6,321.6,589.6,777.6,1079.6],widths=[102,126,250,170,284,142],kinds=['io','io','enc','bridge','llm','io'];
 const bs=labels.map((label,i)=>box(s,'pipeline-'+i,label,xs[i],218,widths[i],112,kinds[i]));
 for(let i=0;i<bs.length-1;i++)connect(s,bs[i],bs[i+1]);
 const prompt=box(s,'prompt','System + task prompt\nDataset emotion candidates',799.6,365,240,64,'io',16);
 connect(s,prompt,bs[4],'top','bottom');return bs;
}

// 1: Complete benchmark inventory. The following three pages expand the open audio LLMs.
{
 const s=base('Model Survey: All Ten Evaluated Systems');
 text(s,'subtitle','Five dedicated SER models, three open audio LLMs and two API models. All completed 16 datasets.',57.6,124,1164,28,18.67,C.ink);
 const rows=facts.survey_rows.map(r=>[r.model,r.release,r.parameters,r.emotion_training_data,r.output]);
 const t=s.tables.add({rows:11,columns:5,left:57.6,top:161,width:1164,height:467,columnWidths:[237,99,122,426,280],values:[facts.survey_columns,...rows]});
 t.borders.assign({style:'solid',fill:C.line,width:.8});
 t.cells.block({row:0,column:0,rowCount:11,columnCount:5}).assign({textStyle:{fontSize:17.33,typeface:'Calibri',color:C.ink},margins:{left:9,right:7,top:0,bottom:0},anchor:'center'});
 for(let r=0;r<11;r++){
  t.rows[r].height=r===0?37:43;
  for(let c=0;c<5;c++){
   const cell=t.getCell(r,c);cell.fill=r===0?C.ink:r%2===0?C.panel:'#FFFFFF';
   cell.text.style={fontSize:r===0?18.67:17.33,typeface:'Calibri',bold:r===0||c===0,color:r===0?'#FFFFFF':C.ink,verticalAlignment:'middle'};
  }
 }
 s.speakerNotes.text=facts.survey_protocol_notes+'\n\n'+facts.survey_rows.map(r=>r.model+' — '+r.training_detail+'\nSources: '+r.sources.join('; ')).join('\n\n')+'\n\nAll ten systems completed the same 67,062 clips per model on sixteen dataset cohorts under the current saved benchmark protocol. Parameter counts for the five SER models follow the existing measured-model documentation in presentation/content.js and facts.md. Open audio LLM counts are measured from the released local checkpoint tensors, not inferred from the 7B/E4B names. Qwen2.5-Omni 8.93B is the loaded Thinker, including an unused vision tower. Qwen2-Audio is 8.397B in the retained Instruct checkpoint; its paper rounds a related model count to 8.2B. Gemma 4 E4B is 8.00B stored with per-layer embedding tables, 4.5B effective according to the official model card, and dense. API internals and parameter counts are not published. Architecture details are expanded in the following three native schematic slides. No new inference or training was performed.';
}

// 2: Qwen2.5-Omni: only the Thinker audio-to-text route runs in this benchmark.
{
 const s=base('Qwen2.5-Omni 7B: Architecture','8.93B loaded Thinker. The benchmark uses its audio and text path.');
 text(s,'encoder-caption','Audio tower · 639.6M',321.6,182,250,26,16,C.blue,true,'center');
 text(s,'bridge-caption','Audio-to-text bridge',589.6,182,170,26,16,C.green,true,'center');
 text(s,'decoder-caption','Qwen2.5 Thinker decoder',777.6,182,284,26,16,C.orange,true,'center');
 pipeline(s,['Waveform\n16 kHz\nmono','Log-mel\n128 bins\n25 ms / 10 ms','Conv stem + 32 Transformer blocks\nd 1280 · 20 heads\nWhisper-initialized','Temporal pooling\nLinear 1280 → 3584\n≈40 ms / audio token','Thinker: 28 decoder blocks\nd 3584 · FFN 18944\n28 query / 4 KV heads','Generate text\n“happy”\nParse one label']);
 box(s,'unused-vision','Vision tower · 676.6M\nLoaded, unused on these inputs',57.6,364,274,66,'muted',16);
 box(s,'unused-speech','Talker + token2wav\nSpeech generation not loaded',350,364,310,66,'muted',16);
 detail(s,0,'Audio representation','The encoder converts acoustic frames to embeddings in the language decoder’s 3,584-dimensional space.');
 detail(s,1,'Joint decoding','Audio embeddings and prompt tokens enter the Thinker decoder, which generates the emotion label as text.');
 detail(s,2,'Input and output','One utterance, without conversation history. Chunked audio input is supported; the benchmark produces no speech.');
 notes(s,'qwen2_5_omni_7b','This schematic shows the executed audio-to-text route rather than every released modality. The local 32-layer audio encoder is 1280-dimensional with 20 attention heads and an output projection weight of shape [3584,1280]. Audio representations correspond to about 40 ms of the source signal. Thinker has 28 text decoder layers, hidden width 3584, FFN width 18944, 28 query heads and 4 KV heads. The retained Thinker has 8,931,813,888 parameter elements. Its 639,647,232-parameter audio tower includes the shown projection, so that projection must not be added to the tower total a second time. The 676,550,144-parameter vision tower is loaded by the class but does not receive image/video inputs. The local loader explicitly selects Qwen2_5OmniThinkerForConditionalGeneration and excludes Talker and token2wav weights. The 300-second feature-extractor setting is not presented as an independently measured unlimited context guarantee; the local loader has no explicit MAX_SECONDS cap. The neutral benchmark prompt provides dataset emotion candidates and a saved parser scores the generated answer.');
}

// 3: Qwen2-Audio: retained checkpoint resolves the sparse language configuration.
{
 const s=base('Qwen2-Audio 7B Instruct: Architecture','8.40B checkpoint. Audio analysis generates text from the recording and the task prompt.');
 text(s,'encoder-caption','Audio encoder · 637.0M',321.6,182,250,26,16,C.blue,true,'center');
 text(s,'bridge-caption','Projection · 5.25M',589.6,182,170,26,16,C.green,true,'center');
 text(s,'decoder-caption','Qwen-7B language model',777.6,182,284,26,16,C.orange,true,'center');
 pipeline(s,['Waveform\n16 kHz mono\nFirst 30 s','Log-mel\n128 bins\n25 ms / 10 ms','Conv stem + 32 Transformer blocks\nd 1280 · 20 heads\nWhisper-large-v3 initialization','Mean pool, stride 2\nLinear 1280 → 4096\n≈40 ms / audio token','Qwen decoder: 32 blocks\nd 4096 · FFN 11008\nNext-token prediction','Generate text\n“happy”\nParse one label']);
 box(s,'training-recipe','Author training: pre-training, instruction tuning and DPO',57.6,365,640,64,'io',18.67);
 detail(s,0,'Audio encoder','The Whisper-style encoder retains a sequence of acoustic representations rather than a single pooled emotion vector.');
 detail(s,1,'Learned projection','A 1,280-to-4,096 linear layer aligns those representations with the language model’s token-embedding width.');
 detail(s,2,'Benchmark protocol','The local processor uses a 30-second window. Audio comes before the text instruction in the current-utterance prompt.');
 notes(s,'qwen2_audio_7b','The official Qwen2-Audio technical report describes Whisper-large-v3 initialization, 128-channel mel input at 16 kHz with 25 ms windows and 10 ms hops, and stride-two pooling, giving approximately 40 ms per audio representation. The sparse retained config omits language hidden size and layer count; this slide resolves them from the actual retained weight headers instead of substituting Qwen2-7B defaults. There are language_model.model.layers.0 through .31, token embeddings [156032,4096], query and key projections [4096,4096], FFN gate projections [11008,4096], and multimodal projection [4096,1280]. The checkpoint has 8,397,094,912 parameter elements: audio tower 636,968,960; projector 5,246,976; text model including output head 7,754,878,976. The paper’s rounded 8.2B is not the exact retained Instruct checkpoint count. The author’s pre-training/SFT/DPO sequence is architecture context, not new training performed by this project. The local Qwen2AudioForConditionalGeneration loader returns text, not speech, and clips audio at 30 seconds before scoring a generated label.');
}

// 4: Gemma: dense decoder, audio tower and per-layer embedding lookups.
{
 const s=base('Gemma 4 E4B: Architecture','8.00B stored parameters, 4.5B effective. A dense decoder with per-layer token embeddings.');
 text(s,'encoder-caption','Audio tower · 304.8M',321.6,182,250,26,16,C.blue,true,'center');
 text(s,'bridge-caption','Projection · 3.93M',589.6,182,170,26,16,C.green,true,'center');
 text(s,'decoder-caption','Gemma text decoder + PLE',777.6,182,284,26,16,C.orange,true,'center');
 pipeline(s,['Waveform\n16 kHz mono\nFirst 30 s','Log-mel\n128 bins\n20 ms / 10 ms','2 convolution subsampling layers\n12 audio blocks · d 1024\n8 heads · output width 1536','Linear projection\n1536 → 2560\nAudio token embeddings','42 dense decoder blocks\nd 2560 · 8 query / 2 KV heads\nLocal + full attention, with PLE','Generate text\nThinking off\nParse one label']);
 box(s,'unused-vision','Vision tower · 167.4M\nLoaded, unused on these inputs',57.6,365,294,66,'muted',16);
 box(s,'ple','Per-layer embeddings (PLE)\nToken lookups supply 256 values per decoder layer',370,365,372,66,'bridge',16);
 detail(s,0,'Audio token stream','The processor uses 750 audio tokens at 40 ms each. The local loader therefore limits the utterance to 30 seconds.');
 detail(s,1,'Hybrid attention','The text decoder uses 512-token local windows and full attention every sixth layer. Its configuration enables KV sharing for 18 layers.');
 detail(s,2,'Per-layer embeddings','Large token-lookup tables supplement the 42 dense layers. This accounts for the gap between stored and effective parameters.');
 notes(s,'gemma4_e4b','The retained Gemma 4 E4B checkpoint uses a dedicated 304,825,088-element audio tower, in agreement with the official model card’s approximately 300M encoder. Local config: two 2-D subsampling convolutions with channels [128,32], 12 audio layers of hidden width 1024 with 8 heads, convolution kernel 5, and output width 1536. The embedding_projection weight has shape [2560,1536] and 3,932,160 elements. The text decoder has 42 layers, hidden width 2560, FFN width 10240, 8 query heads and 2 KV heads, local sliding window 512 and a full-attention layer every sixth layer. Local attention head_dim is 256, with global_head_dim 512. The config declares 18 KV-shared layers. Per-layer embeddings supply a 256-dimensional token-specific input at each of the 42 decoder layers; the large embedding table has shape [262144,10752]. This is a dense architecture, not the distinct 26B A4B MoE model. The full checkpoint contains 7,996,157,418 tensor elements including PLE, audio, vision and small state tensors; the official model card describes 8B with embeddings and 4.5B effective. The 167,365,056-element vision tower is loaded but receives no image/video in this benchmark. The local processor defines 750 audio tokens ×40 ms =30 s and places user text before audio, with enable_thinking=False.');
}

const build=path.join(DECK,'.inspection-build');
await fs.mkdir(path.join(build,'model-refresh-previews'),{recursive:true});
await (await PresentationFile.exportPptx(p)).save(path.join(build,'model-refresh.pptx'));
const order=[{kind:'model-survey',key:'survey',title:'Model Survey: All Ten Evaluated Systems'},
 {kind:'model-architecture',key:'omni',model:'qwen2_5_omni_7b',title:'Qwen2.5-Omni 7B: Architecture'},
 {kind:'model-architecture',key:'audio',model:'qwen2_audio_7b',title:'Qwen2-Audio 7B Instruct: Architecture'},
 {kind:'model-architecture',key:'gemma',model:'gemma4_e4b',title:'Gemma 4 E4B: Architecture'}];
await fs.writeFile(path.join(build,'model-refresh-order.json'),JSON.stringify(order,null,2)+'\n');
for(let i=0;i<p.slides.items.length;i++)await fs.writeFile(path.join(build,'model-refresh-previews',`page-${i+1}.png`),new Uint8Array(await (await p.slides.items[i].export({format:'png',scale:1})).arrayBuffer()));
console.log('Authored four model-refresh slides with editable tables, shapes and connectors');
