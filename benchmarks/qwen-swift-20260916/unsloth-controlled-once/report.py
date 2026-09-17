import json,pathlib,hashlib
r=pathlib.Path(__file__).resolve().parent
base=r.parent/'swift-controlled-v1'; app=r.parent.parent/'outputs/swift-controlled-report'
a=json.loads((r/'results.json').read_text()); old=json.loads((base/'results.json').read_text()); assert len(a)==16
rows=[]
for x in a:
 b=next(y for y in old if y['case']==x['case'] and y['backend']=='dflash')
 assert x['request']==b['request'] and x['prompt_tokens']==b['prompt_tokens']
 assert x['score']['completed'] and x['reasoning_text_tokens']>0
for group in ['quality','speed']:
 b=[x for x in a if x['suite']==group]; decode=sum(x['decode_ms'] for x in b)/1000
 rows.append(dict(suite=group,backend='Unsloth + DFlash2 (one run)',prefill=round(sum(x['prefill_ms'] for x in b)/1000,2),decode=round(decode,2),wall=round(sum(x['wall_seconds'] for x in b),2),tokens=sum(x['total_tokens'] for x in b),thinking=sum(x['reasoning_text_tokens'] for x in b),tps=round(sum(x['total_tokens'] for x in b)/decode,2)))
passed=sum(x['score']['pass'] for x in a if x['suite']=='quality')
q=rows[0];s=rows[1]
text=f'''## Regular Unsloth model: one additional run

The regular **Qwen3.8-27B-UD-IQ4_XS** model with DFlash2 finished the ten article prompts in **{s['wall']:.2f}s**, producing **{s['tokens']:,} tokens**, including **{s['thinking']:,} thinking-text tokens**, at **{s['tps']:.2f} tokens/s**. It passed **{passed}/6** fixed quality checks.

This is **one run**, compared with the three-run Swift medians above. The same 16 request bodies, scoring rules, xhigh settings, frozen Swift template, drafter and server binary were used. Input token counts match for every task. Prefix caching and compression remain disabled. The Unsloth model's native template differs; we intentionally retained the frozen template to hold the rendered instructions constant. The quantization files are both IQ4_XS-family but have different per-layer quantization layouts, so this is not a pure fine-tuning ablation.

Thinking text is counted with the same tokenizer as before; the models' vocabulary and merge hashes match. No response hit the 64,000-token ceiling. This single run does not establish timing variability. Article code remains unscored for correctness.
'''
d=json.loads((app/'src/data.json').read_text());d['unslothText']=text
d['queries']['unsloth']={'rows':rows,'source':{'name':'Unsloth controlled single run','description':'work/unsloth-controlled-once/results.json; exact suite and request bodies match swift-controlled-v1. One repetition, not a median.'}}
d['queries']['unsloth_detail']={'rows':[dict(prompt=x['case'],suite=x['suite'],wall=round(x['wall_seconds'],2),tokens=x['total_tokens'],thinking=x['reasoning_text_tokens'],quality=str(x['score']['pass']) if x['suite']=='quality' else 'Not scored') for x in a],'source':d['queries']['unsloth']['source']}
(app/'src/data.json').write_text(json.dumps(d,indent=2))
p=app/'src/content/report/ReportContent.jsx';c=p.read_text();marker=' <ReportSection id="methods"'
if 'id="unsloth-findings"' not in c:
 c=c.replace(marker,''' {snapshot.queries.unsloth&&<>
 <ReportSection id="unsloth-findings" title="Unsloth follow-up" queryId="unsloth" showHeading={false}><RichNarrative id="unsloth-text" value={snapshot.unslothText}/></ReportSection>
 {table('unsloth','Unsloth + DFlash2 — one run',[['suite','Suite'],['prefill','Prefill (s)'],['decode','Generation (s)'],['wall','Total (s)'],['tokens','Generated tokens'],['thinking','Thinking text tokens'],['tps','Decode tok/s']])}
 {table('unsloth_detail','Unsloth per-prompt results',[['prompt','Prompt'],['suite','Suite'],['wall','Total (s)'],['tokens','Generated tokens'],['thinking','Thinking text tokens'],['quality','Quality pass']])}
 </>}
'''+marker)
 p.write_text(c)
md=text+'\n| Suite | Prefill s | Generation s | Total s | Generated tokens | Thinking tokens | Tok/s |\n|---|---:|---:|---:|---:|---:|---:|\n'
for x in rows:md+='| '+' | '.join(str(x[k]) for k in ['suite','prefill','decode','wall','tokens','thinking','tps'])+' |\n'
(r.parent.parent/'outputs/unsloth-controlled-once.md').write_text(md)
p=r.parent.parent/'outputs/swift-controlled-report.md';v=p.read_text();v=v.split('\n## Regular Unsloth model: one additional run')[0];p.write_text(v+'\n'+md)
(r/'summary.json').write_text(json.dumps({'rows':rows,'quality_passes':passed},indent=2));print(json.dumps({'rows':rows,'quality_passes':passed},indent=2))
