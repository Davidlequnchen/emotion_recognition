import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
const DECK=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
process.env.RUNTIME_NODE_MODULES ||= '/Users/lequnchen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const SKILL=process.env.PRESENTATIONS_SKILL_DIR || '/Users/lequnchen/.codex/plugins/cache/openai-primary-runtime/presentations/26.1105.12137/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')));
const source=path.join(DECK,'SER_evaluation.pptx');
const candidate=path.resolve(process.argv[2]);
const finalPath=path.resolve(process.argv[3]);
const count=Number(process.argv[4]||1);
const tables=JSON.parse(process.argv[5]||'[1]');
const result=await finalizePresentation({
 workspaceDir:DECK,candidatePath:candidate,finalPath,
 pythonExecutable:process.env.RUNTIME_PYTHON || '/Users/lequnchen/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tables.flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:count,requiredNativeTableOwnerSlides:tables,
 fontPolicy:{basis:'reference',families:count===1?['Calibri']:['Calibri','Arial'],...(count>1?{scriptFonts:{ea:'Arial Unicode MS',cs:'Arial Unicode MS'}}:{}),referencePath:source,referenceSha256:createHash('sha256').update(await fs.readFile(source)).digest('hex')},
 // The 67-slide source and 84-slide extension both exceed the first-party
 // XML construction budget. check_inspections.py makes reachable 12-slide
 // packages so the exact slide content can be imported in bounded batches.
 verifyArtifactToolImport:count===1,
 receiptPath:path.join(DECK,'.inspection-build',path.basename(finalPath)+'.validation.json')
});
console.log(JSON.stringify(result));
