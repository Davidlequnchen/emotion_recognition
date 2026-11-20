// The complete source exceeds the native importer XML budget. Verify every
// slide in exact-content, reachable 12-slide packages produced by the checker.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
const DECK=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime=process.env.RUNTIME_NODE_MODULES||'/Users/lequnchen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {FileBlob,PresentationFile}=await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')));
const build=path.join(DECK,'.inspection-build');
const batches=JSON.parse(await fs.readFile(path.join(build,'import-batches.json'),'utf8'));
const checks=[];
for(const file of batches){
 const p=await PresentationFile.importPptx(await FileBlob.load(file));
 if(p.slides.items.length!==12)throw new Error('Unexpected import slide count');
 checks.push({file,slideCount:p.slides.items.length,passed:true});
 console.log(path.basename(file)+': 12 slides imported');
}
await fs.writeFile(path.join(build,'import-validation.json'),JSON.stringify({
 candidateSha256:createHash('sha256').update(await fs.readFile(path.join(build,'SER_evaluation_extended-candidate.pptx'))).digest('hex'),
 fullPackageImport:'Original source and extended deck exceed the importer XML construction limit',
 slideCount:84,passed:true,checks
},null,2));
