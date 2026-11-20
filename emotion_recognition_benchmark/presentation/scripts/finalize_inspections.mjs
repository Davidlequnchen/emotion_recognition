import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
const DECK=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
process.env.RUNTIME_NODE_MODULES ||= '/Users/lequnchen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const SKILL=process.env.PRESENTATIONS_SKILL_DIR || '/Users/lequnchen/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11011/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')));
const source=process.env.PRESENTATION_REFERENCE_PATH || path.join(DECK,'SER_evaluation.pptx');
const candidate=path.resolve(process.argv[2]);
const finalPath=path.resolve(process.argv[3]);
const count=Number(process.argv[4]||1);
const tables=JSON.parse(process.argv[5]||'[1]');
// The packaged finalizer accepts at most 96 layout arguments. Validate every
// table owner through bounded finalization passes over the identical candidate.
// Private staging copies are byte-identical; only the last is published.
const chunks=[];
for(let i=0;i<tables.length;i+=40)chunks.push(tables.slice(i,i+40));
if(!chunks.length)chunks.push([]);
const checks=[];
for(let i=0;i<chunks.length;i++){
 const owners=chunks[i],last=i===chunks.length-1;
 const checkedPath=last?finalPath:path.join(path.dirname(finalPath),'table-check-'+(i+1),path.basename(finalPath));
 await fs.mkdir(path.dirname(checkedPath),{recursive:true});
 const result=await finalizePresentation({
 workspaceDir:DECK,candidatePath:candidate,finalPath:checkedPath,
 pythonExecutable:process.env.RUNTIME_PYTHON || '/Users/lequnchen/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...owners.flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:count,requiredNativeTableOwnerSlides:owners,
 fontPolicy:{basis:'reference',families:count<=12?['Calibri']:['Calibri','Arial'],...(count>12?{scriptFonts:{ea:'Arial Unicode MS',cs:'Arial Unicode MS'}}:{}),referencePath:source,referenceSha256:createHash('sha256').update(await fs.readFile(source)).digest('hex')},
 // The complete source exceeds the first-party XML construction budget.
 // check_main_refresh.py makes reachable eight-slide packages for exact-content imports.
 verifyArtifactToolImport:count<=12,
 receiptPath:path.join(DECK,'.inspection-build',path.basename(path.dirname(finalPath))+'__'+path.basename(finalPath)+(last?'':'.tables-'+(i+1))+'.validation.json')
});
 checks.push({owners,finalSha256:result.finalSha256,receiptPath:result.receiptPath});
 console.log(JSON.stringify({pass:i+1,tableOwners:owners,finalPath:result.finalPath,finalSha256:result.finalSha256}));
}
if(new Set(checks.map(c=>c.finalSha256)).size!==1)throw new Error('Validation passes changed candidate bytes');
await fs.writeFile(finalPath+'.table-validation.json',JSON.stringify({passed:true,tableOwners:tables,checks},null,2)+'\n');
