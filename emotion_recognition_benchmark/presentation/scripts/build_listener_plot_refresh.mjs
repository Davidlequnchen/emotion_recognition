// Preserve the retained native chart's exact style while extending its data.
import fs from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {fileURLToPath,pathToFileURL} from 'node:url';
const DECK=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const python=process.env.RUNTIME_PYTHON||'/Users/lequnchen/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const runtime=process.env.RUNTIME_NODE_MODULES||'/Users/lequnchen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
execFileSync(python,[path.join(DECK,'scripts/prepare_listener_plot_refresh.py')],{stdio:'inherit'});
const {FileBlob,PresentationFile}=await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')));
const file=path.join(DECK,'.inspection-build/listener-plot-refresh.pptx');
const p=await PresentationFile.importPptx(await FileBlob.load(file));
if(p.slides.items.length!==1)throw new Error('Expected one edited chart slide');
const preview=await p.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(DECK,'.inspection-build/listener-plot-preview.png'),new Uint8Array(await preview.arrayBuffer()));
console.log('One-slide ten-model native chart imported and previewed.');
