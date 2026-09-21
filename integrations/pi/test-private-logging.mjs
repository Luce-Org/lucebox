import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import {stripTypeScriptTypes} from 'node:module';
const dir=fs.mkdtempSync(path.join(os.tmpdir(),'pcg-logging-'));
try {
 for(const enabled of [false,true]) {
  process.env.PCG_LOG='1';process.env.PCG_LOG_TEXT=enabled?'1':'0';
  process.env.PCG_LOG_FILE=path.join(dir,String(enabled));
  let source=fs.readFileSync(new URL('./prefix-cache-guard.ts',import.meta.url),'utf8');
  source=source.replace('import { convertToLlm } from "@earendil-works/pi-coding-agent";','const convertToLlm = x => x;');
  const mod=await import('data:text/javascript;base64,'+Buffer.from(stripTypeScriptTypes(source)+`\n// ${enabled}`).toString('base64'));
  const hooks=new Map();mod.default({on:(k,v)=>hooks.set(k,v),registerCommand:()=>{}});
  for(const content of ['PRIVATE_OLD_CONTENT','PRIVATE_NEW_CONTENT']) hooks.get('context')({messages:[{role:'user',content}]},{ui:{notify:()=>{}}});
  const log=fs.readFileSync(process.env.PCG_LOG_FILE,'utf8');
  assert(log.includes('BREAK'));assert.equal(log.includes('PRIVATE_OLD_CONTENT'),enabled);
 }
 console.log('PASS: default logs omit prompt text; explicit opt-in includes it.');
} finally {fs.rmSync(dir,{recursive:true,force:true});}
