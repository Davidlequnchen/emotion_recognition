// Three native tables covering the same complete 10-model / 16-dataset cohort.
// Values remain exact in the companion data file; the slide displays percentages.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';

const DECK=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const ROOT=path.resolve(DECK,'../..');
const BUILD=path.join(DECK,'.inspection-build');
const ASSETS=path.join(DECK,'assets/result_refresh');
const runtime=process.env.RUNTIME_NODE_MODULES||'/Users/lequnchen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')));
const sourcePaths=[
 'research_paper/output/cross_dataset_analysis.json',
 'emotion_recognition/reports/summary.json',
 'emotion_recognition/presentation/content.js',
 'emotion_recognition/presentation/s_results.js',
 'emotion_recognition/presentation/s_appendix.js',
 'emotion_recognition/docs/training_provenance.md'
];
const sourceBuffers=await Promise.all(sourcePaths.map(f=>fs.readFile(path.join(ROOT,f))));
const sources=sourcePaths.map((file,i)=>({path:file,sha256:crypto.createHash('sha256').update(sourceBuffers[i]).digest('hex')}));
const analysis=JSON.parse(sourceBuffers[0]);
const summary=JSON.parse(sourceBuffers[1]);
const models=analysis.manifest.model_order;
const order=['meld','emotiontalk','dailytalk','multidialog','crema_d','ravdess','savee','jl_corpus','tess','mead','mess','mcae_spps','thai_ser','nemo','emovo','emozionalmente'];
const names=Object.fromEntries(summary.models.map(m=>[m.model,m.name]));
const dsNames=Object.fromEntries(analysis.dataset_rows.map(d=>[d.dataset,d.name]));
const headers=['e2v+\nlarge','MERaLiON\nSER-v1','Melati','Vox\nWhisper','Vox\nWavLM','Qwen2.5\nOmni','Qwen2\nAudio','Gemma 4\nE4B','GPT\nRealtime','Gemini\n3.8 Flash'];
const metricSpecs=[['accuracy','accuracy','Dataset Accuracy: All Ten Models'],['uar','unweighted_accuracy','UAR: All Ten Models'],['macro_f1','macro_f1','Macro-F1: All Ten Models']];
const row=(model,dataset)=>analysis.model_rows.find(r=>r.model===model&&r.dataset===dataset);
const involved=status=>['confirmed_exposure','probable_exposure'].includes(status);
if(models.length!==10||analysis.manifest.selected_run_count!==160||analysis.manifest.full_models.length!==10)throw new Error('Requires the complete 160-run analysis');
for(const model of models)for(const dataset of order){
 const r=row(model,dataset),s=summary.results[model]?.[dataset];
 if(!r||!s||s.sample||s.clips!==s.total_clips||r.n!==s.clips)throw new Error(`Missing/incomplete cohort: ${model}/${dataset}`);
 for(const [metric,summaryMetric] of metricSpecs)if(Math.abs(r[metric]-s[summaryMetric])>.00000051)throw new Error(`Source mismatch: ${model}/${dataset}/${metric}`);
 if(r.training_status!==s.training_status)throw new Error(`Provenance mismatch: ${model}/${dataset}`);
}
const C={ink:'#1B2A41',text:'#2F3B4C',muted:'#586474',blue:'#0072B2',purple:'#7A5195',orange:'#D55E00',line:'#D5DAE1',panel:'#F4F6F9',mean:'#E7EDF4',good:'#2E8B57',bad:'#C0392B'};
const family=i=>i<5?'SER':i<8?'open_audio_llm':'API';
const familyColor=i=>i<5?C.blue:i<8?C.purple:C.orange;
const p=Presentation.create({slideSize:{width:1280,height:720}});
const slideOrder=[];
const data={
 schema_version:1,
 authority:'research_paper/output/cross_dataset_analysis.json:model_rows; cross-checked against reports/summary.json (six-decimal source rounding)',
 generated_from:analysis.manifest.generated_at_singapore,
 sources,
 model_order:models,dataset_order:order,
 models:models.map((key,i)=>({key,name:names[key],header:headers[i],family:family(i)})),
 datasets:order.map(key=>({key,name:dsNames[key],n:row(models[0],key).n})),
 table_structure:{columns:11,rows:18,header_rows:1,dataset_rows:16,mean_rows:1,mean_definition:'Equal weight per dataset, not pooled clips'},
 training_marker:'†',
 training_marker_definition:'Confirmed or probable documented training-process involvement; includes upstream teacher lineage and development use. This does not establish exact test-clip overlap. No marker means involvement is not established, not certified unseen.',
 training_status:order.map(d=>models.map(m=>row(m,d).training_status)),
 metrics:{},metric_details:{}
};
function text(s,name,value,x,y,w,h,size=20,color=C.text,bold=false){
 const t=s.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=value;t.text.style={fontSize:size,typeface:'Calibri',color,bold,verticalAlignment:'top',insets:0,autoFit:'none'};
 return t;
}
function base(title,key){
 const s=p.slides.add();s.background.fill='#FFFFFF';
 text(s,'section',key==='accuracy'?'4 · EVALUATION RESULTS':'APPENDIX · COMPLETE METRICS',57.6,27,1164,28,14.67,C.blue,true);
 text(s,'title',title,57.6,53,1164,65,34.67,C.ink,true);
 text(s,'legend-ser','SER',57.6,126,38,24,16.5,C.blue,true);
 text(s,'legend-open','Open audio LLM',107,126,150,24,16.5,C.purple,true);
 text(s,'legend-api','API',273,126,42,24,16.5,C.orange,true);
 text(s,'legend-extrema','Green / red: best / worst within each dataset',340,126,395,24,16.5,C.text);
 text(s,'legend-training','† documented training involvement',757,126,465,24,16.5,C.text);
 slideOrder.push({slide:p.slides.items.length,kind:'result-table',key,title,rows:18,columns:11});
 return s;
}
for(const [metric,,title] of metricSpecs){
 const raw=order.map(d=>models.map(m=>row(m,d)[metric]));
 const values=raw.map(v=>v.map(x=>100*x));
 const display=values.map((v,r)=>v.map((x,c)=>`${x.toFixed(1)}${involved(data.training_status[r][c])?'†':''}`));
 const mean=models.map((m,c)=>100*raw.reduce((sum,r)=>sum+r[c],0)/order.length);
 const means=mean.map(x=>x.toFixed(1));
 const tableValues=[['Dataset',...headers],...order.map((d,r)=>[dsNames[d],...display[r]]),['Dataset mean',...means]];
 data.metrics[metric]=values;
 data.metric_details[metric]={unit:'percent',raw_fractions:raw,display,mean,mean_display:means,table_values:tableValues};
 const s=base(title,metric);
 const heights=[50,...Array(17).fill(24.5)];
 const t=s.tables.add({rows:18,columns:11,left:57.6,top:158,width:1164,height:heights.reduce((a,b)=>a+b,0),columnWidths:[184,...Array(10).fill(98)],values:tableValues});
 t.borders.assign({style:'solid',fill:C.line,width:.7});
 t.cells.block({row:0,column:0,rowCount:18,columnCount:11}).assign({textStyle:{fontSize:17.3,typeface:'Calibri',color:C.ink},margins:{left:7,right:5,top:2,bottom:2},anchor:'center'});
 for(let r=0;r<18;r++){
  t.rows[r].height=heights[r];
  for(let c=0;c<11;c++){
   const cell=t.getCell(r,c);
   cell.fill=r===0?(c===0?C.ink:familyColor(c-1)):r===17?C.mean:r%2===0?C.panel:'#FFFFFF';
   let color=r===0?'#FFFFFF':C.ink,bold=r===0||c===0||r===17;
   if(r>0&&r<17&&c>0){
    const v=raw[r-1][c-1],best=Math.max(...raw[r-1]),worst=Math.min(...raw[r-1]);
    if(Math.abs(v-best)<1e-12){color=C.good;bold=true;}
    else if(Math.abs(v-worst)<1e-12){color=C.bad;bold=true;}
   }
   if(r===0)cell.margins={left:3,right:3,top:2,bottom:2};
   cell.text.style={fontSize:r===0?16.3:17.3,typeface:'Calibri',color,bold,verticalAlignment:'middle',alignment:c===0?'left':'center'};
  }
 }
 const metricDefinition=metric==='accuracy'?'Accuracy: fraction of utterances whose prediction is accepted for the corpus reference.':metric==='uar'?'UAR: unweighted mean recall over the original gold classes.':'Macro-F1: unweighted mean class F1 under the original benchmark composite-label convention.';
 const perDataset=order.map((d,r)=>`${dsNames[d]} (n=${row(models[0],d).n}): ${models.map((m,c)=>`${names[m]} = ${values[r][c]}% [display ${display[r][c]}; training_status=${data.training_status[r][c]}]`).join('; ')}`).join('\n');
 const provenance=order.flatMap(d=>models.filter(m=>involved(row(m,d).training_status)).map(m=>{
  const r=summary.results[m][d];return `${names[m]} / ${dsNames[d]}: ${r.training_status}; scope=${r.training_scope}; ${r.training_reason||r.training_source}. Evidence: ${(r.training_evidence||[]).join(', ')}`;
 })).join('\n');
 s.speakerNotes.text=`${title}. Native editable table: 11 columns, 18 rows (one header, sixteen corpora, one mean). All values are percentages, rounded to one decimal for display.\n${metricDefinition}\nAll ten models cover the same 67,062 saved clips across sixteen full cohorts; no sampled pilot runs. Invalid generated answers remain errors. Candidate label sets and original accepted-label scoring are preserved; JL-Corpus excited accepts happy or surprised. Values originate in the full-precision model_rows analysis, as in content.js and the appendix; they agree with the six-decimal reports/summary.json used by s_results.js to within its rounding precision.\nGreen and red text identify the largest and smallest full-precision values within each dataset among all ten models, including ties. They are relative extrema, not fixed performance bands, confidence intervals or significance claims. The mean row gives every dataset equal weight and has no extrema coloring.\n${data.training_marker_definition}\nHeader groups: first five blue = dedicated SER; next three purple = open audio LLMs; final two orange = API audio LLMs. Abbreviated labels correspond to: ${models.map((m,i)=>`${headers[i].replace(/\n/g,' ')} = ${names[m]}`).join('; ')}.\nAuthoritative generation: ${analysis.manifest.generated_at_singapore}.\nSources and SHA256:\n${sources.map(x=>`${x.path}: ${x.sha256}`).join('\n')}\nFull precision values and displayed strings:\n${perDataset}\nEqual-dataset means: ${models.map((m,i)=>`${names[m]} = ${mean[i]}% [display ${means[i]}]`).join('; ')}\nTraining involvement details:\n${provenance}`;
}
await fs.mkdir(BUILD,{recursive:true});
await fs.mkdir(ASSETS,{recursive:true});
await fs.mkdir(path.join(BUILD,'result-refresh-previews'),{recursive:true});
await fs.writeFile(path.join(ASSETS,'data.json'),JSON.stringify(data,null,2)+'\n');
await fs.writeFile(path.join(BUILD,'result-refresh-order.json'),JSON.stringify(slideOrder,null,2)+'\n');
await(await PresentationFile.exportPptx(p)).save(path.join(BUILD,'result-refresh.pptx'));
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.slides.items[i].export({format:'png',scale:1});
 await fs.writeFile(path.join(BUILD,'result-refresh-previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log(JSON.stringify({pptx:path.join(BUILD,'result-refresh.pptx'),order:slideOrder,data:path.join(ASSETS,'data.json'),maxVisibleBottom:624.5}));
