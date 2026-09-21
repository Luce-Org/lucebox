import json,pathlib,statistics,hashlib
r=pathlib.Path(__file__).resolve().parent;app=r.parent.parent/'outputs/swift-controlled-report';a=json.loads((r/'results.json').read_text());analysis=json.loads((r/'analysis.json').read_text());s=json.loads((r/'suite.json').read_text());assert len(a)==96
for c in s['cases']:
 rows=[x for x in a if x['case']==c['id']];assert len(rows)==6
 assert len({json.dumps(x['request'],sort_keys=True) for x in rows})==1
 assert len({x['prompt_tokens'] for x in rows})==1
assert all(x['score']['completed'] for x in a)
assert all(x['reasoning_text'] and 'reasoning_text_tokens' in x for x in a)
assert all(
 (x['backend']=='dflash' and x['response']['usage']['timings']['cached_prefix_tokens']==0)
 or (x['backend']=='mtp' and x['response']['timings']['cache_n']==0)
 for x in a
)
assert hashlib.sha256((r/'suite.json').read_bytes()).hexdigest()==(r/'suite.sha256').read_text().split()[0]
summary=analysis['summary'];pick=lambda b,g:next(x for x in summary if x['backend']==b and x['suite']==g)
df=pick('Lucebox + DFlash2','speed');mtp=pick('llama.cpp + MTP','speed');saving=100*(1-df['median_wall']/mtp['median_wall']);passes={b:sum(x['score']['pass'] for x in a if x['suite']=='quality' and x['backend']==b) for b in ['dflash','mtp']}
finding=f"## DFlash2 finished the article suite {saving:.1f}% sooner\n\nWith the frozen inputs and xhigh thinking, median total request time was **{df['median_wall']:.2f}s for Lucebox + DFlash2** and **{mtp['median_wall']:.2f}s for llama.cpp + MTP**. Median generation throughput was **{df['median_tps']:.2f} versus {mtp['median_tps']:.2f} tokens/s**, including thinking.\n\nBoth passed **18/18 quality checks**: six distinct tasks repeated three times, with exact answers and strict JSON/tool-call formatting. No output reached the ceiling. This is a small correctness screen, not proof of equal quality on broader coding work.\n\nThe backends generated different numbers of tokens despite identical inputs. Total latency therefore reflects both output length and generation speed. This compares complete serving configurations, not the speculation algorithms in isolation."
methods="""## Reading the measurements

**Two suites, never pooled.** Quality has six fixed, known-answer tasks. Speed has the ten canonical Lucebox article prompts; generated code is not executed or scored for correctness. Each row sums one full suite, then takes the median across three repetitions. Per-prompt rows take medians separately, so their sums need not equal the suite median.

**Tokens.** Input and total generated counts come from each server. Total generated includes thinking, final answers, and any generated protocol tokens. Thinking text is retokenized with the same Swift tokenizer for both backends, without added special tokens. This excludes wrapper markers and may differ slightly from native reasoning-token counts; Lucebox's native counts are retained in the raw data. Do not subtract thinking text tokens from total generated and label the remainder exactly as visible-answer tokens.

**Time.** Prefill and generation are the backend's own reported phase times. Total time is externally measured HTTP wall time; it also includes parsing, transport and other request overhead. Model loading, a single unmeasured warmup, and post-run token counting are excluded. Decode tok/s divides all generated tokens by generation time. Three repeats quantify limited timing variation; fixed-seed repeats do not create new independent quality tasks.

## Frozen settings and controls

Same Swift IQ4_XS file and embedded Jinja template; xhigh thinking, temperature 0, seed 42, non-streaming requests, single stream, 65,536-token context, 64,000-token output ceiling, GPU weights and Q8 K/V. Forced thinking closure and prefix caching are disabled. The finite context prevents truly unlimited output; no response approached the ceiling. DFlash2 uses its existing Q8 drafter and block 16; MTP uses the embedded head and maximum three draft tokens. No prompt compression.

Run order: DFlash2, MTP, MTP, DFlash2, DFlash2, MTP. Fresh process and the same short warmup per repetition; GPU memory is checked between loads. Request bodies and prompt-token counts match across all six runs. All 96 responses contain thinking text, complete normally, and report zero cached input tokens.

## Scope and exclusions

The six quality prompts and rubric were frozen before measured runs. The JSON-array requirement is explicit in this version and identical for both backends. No answers were repaired or coerced. The exact text, expected objects and tool schema appear below.

Earlier exploratory results are excluded. A discarded setup run exposed a silent 32,768-token server clamp; startup was corrected and all measured repetitions restarted. The partial data remain in a separately named rejected-preflight folder and are not included here. Earlier startup recovery briefly encountered delayed GPU-memory release; measured runs used a release check between loads.

These results describe this machine, model, quantization and the two complete backends. Different numerical kernels and generated traces remain part of that comparison. The speed suite does not establish code correctness, and the quality suite is too small to establish broad model equivalence.
"""
p=app/'src/data.json';d=json.loads(p.read_text());d['finding']=finding;d['methodsText']=methods;d['status']='reviewed';d['buildStatus']='complete';p.write_text(json.dumps(d,indent=2))
md=['# Swift IQ4_XS: controlled DFlash2 versus MTP comparison','',finding,'','## Suite medians (three repetitions)','', '| Suite | Backend | Input tokens | Generated tokens | Thinking text tokens | Prefill s | Generation s | Total s | Decode tok/s |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for x in summary:md.append(f"| {x['suite']} | {x['backend']} | {x['median_prefill_tokens']:,.0f} | {x['median_tokens']:,.0f} | {x['median_thinking']:,.0f} | {x['median_prefill']:.2f} | {x['median_decode']:.2f} | {x['median_wall']:.2f} | {x['median_tps']:.2f} |")
md+=['','## Variation across the three runs','','| Suite | Backend | Total-time range (s) | Decode-rate range (tok/s) |','|---|---|---:|---:|']
for x in summary:md.append(f"| {x['suite']} | {x['backend']} | {x['min_wall']:.2f}–{x['max_wall']:.2f} | {x['min_tps']:.2f}–{x['max_tps']:.2f} |")
md+=['',methods,'','## Exact quality inputs and frozen criteria']
for c in s['cases']:
 md+=['',f"### {c['suite']}: {c['id']}",'','```text',c['messages'][0]['content'],'```','',c['criterion']]
 if 'expected' in c:md+=['','Expected:','```json',json.dumps(c['expected'],indent=2),'```']
 if 'tools' in c:md+=['','Tool schema:','```json',json.dumps(c['tools'],indent=2),'```']
md+=['','## Evidence','',f"Protocol SHA-256: `{(r/'suite.sha256').read_text().split()[0]}`.",'','Raw responses, requests, native timings, launch flags, executable/model/template identities, per-prompt CSVs, and reproduction scripts are supplied in the evidence bundle. Production was restored after measurement; see the final health record.']
(r.parent.parent/'outputs/swift-controlled-report.md').write_text('\n'.join(md)+'\n')
print(json.dumps({'saving_percent':saving,'summary':summary,'quality_passes':passes},indent=2))
