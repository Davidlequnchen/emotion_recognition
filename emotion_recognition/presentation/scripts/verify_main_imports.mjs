// Verify every final slide in bounded exact-content packages.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
const DECK=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime=process.env.RUNTIME_NODE_MODULES||'/Users/lequnchen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {FileBlob,PresentationFile}=await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')));
const dir=path.join(DECK,'.inspection-build/refresh');
const batches=JSON.parse(await fs.readFile(path.join(dir,'import-batches.json'),'utf8'));
const checks=[];
for(const batch of batches){
 const p=await PresentationFile.importPptx(await FileBlob.load(batch.path));
 if(p.slides.items.length!==batch.slides)throw new Error('Slide count mismatch: '+batch.path);
 checks.push({path:batch.path,slides:batch.slides,passed:true});
 console.log(path.basename(batch.path)+': '+batch.slides+' slides imported');
}
const count=checks.reduce((s,x)=>s+x.slides,0);if(count!==93)throw new Error('Expected 93 slides');
await fs.writeFile(path.join(dir,'import-validation.json'),JSON.stringify({passed:true,slides:count,candidateSha256:createHash('sha256').update(await fs.readFile(path.join(dir,'main-candidate.pptx'))).digest('hex'),checks},null,2));
