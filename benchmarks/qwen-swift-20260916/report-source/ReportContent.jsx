import React from 'react';
import { useDataApp, RichNarrative, ReportSection, DataComponent, DataTable, EvidenceChart } from '../../data-app-public.jsx';
export function ReportContent(){
 const {snapshot,appTitle}=useDataApp();
 const rows=id=>snapshot.queries[id]?.rows??[];
 const table=(id,title,columns)=> <DataComponent id={id} title={title} queryId={id} sourceRows={rows(id)} displayRows={rows(id)} kind="table"><DataTable rows={rows(id)} columns={columns.map(([key,label])=>({key,label}))} searchable={rows(id).length>8} caption={title}/></DataComponent>;
 return <article className="report-content">
 <header className="report-hero"><h1>{appTitle}</h1><RichNarrative id="intro" value="One frozen protocol. Thinking enabled at xhigh. Same Swift IQ4_XS weights and embedded template. Three repetitions per configuration." className="report-deck"/></header>
 <ReportSection id="conclusion" title="Findings" queryId={snapshot.queries.summary?'summary':'protocol'} showHeading={false}><RichNarrative id="finding" value={snapshot.finding??'## Measurements in progress\n\nThe comparison is running. No winner or quality conclusion is available yet.'}/></ReportSection>
 {snapshot.queries.summary&&<>
 {table('summary','Time and generation speed',[['suite','Suite'],['backend','Configuration'],['median_prefill','Prefill (s)'],['median_decode','Generation (s)'],['median_wall','Total (s)'],['median_tps','Decode tok/s']])}
 {table('token_summary','Tokens processed and produced',[['suite','Suite'],['backend','Configuration'],['median_prefill_tokens','Input tokens'],['median_tokens','Total generated'],['median_thinking','Thinking text tokens']])}
 <EvidenceChart id="time-chart" queryId="summary" title="Time to finish the full prompt suite" rows={rows('summary')} sourceRows={rows('summary')} spec={{type:'bar',x:'suite',y:'median_wall',series:'backend',stackable:false}} height={300}/>
 <EvidenceChart id="thinking-chart" queryId="summary" title="Thinking text tokens across the full suite" rows={rows('summary')} sourceRows={rows('summary')} spec={{type:'bar',x:'suite',y:'median_thinking',series:'backend',stackable:false}} height={300}/>
 {table('variation','Variation across the three repetitions',[['suite','Suite'],['backend','Configuration'],['min_wall','Fastest total (s)'],['max_wall','Slowest total (s)'],['min_tps','Lowest tok/s'],['max_tps','Highest tok/s']])}
 {table('quality','Quality: fixed answers and strict formatting',[['prompt','Prompt'],['backend','Configuration'],['answers','Correct answers / 3'],['format','Correct format / 3'],['passed','Both passed / 3']])}
 {table('detail','Per-prompt medians across three repetitions',[['suite','Suite'],['prompt','Prompt'],['backend','Configuration'],['wall','Total time (s)'],['prefill','Prefill (s)'],['decode','Generation (s)'],['tokens','Output tokens'],['thinking','Thinking text tokens']])}
 </>}
 {snapshot.queries.unsloth&&<>
 <ReportSection id="unsloth-findings" title="Unsloth follow-up" queryId="unsloth" showHeading={false}><RichNarrative id="unsloth-text" value={snapshot.unslothText}/></ReportSection>
 {table('unsloth','Unsloth + DFlash2 — one run',[['suite','Suite'],['prefill','Prefill (s)'],['decode','Generation (s)'],['wall','Total (s)'],['tokens','Generated tokens'],['thinking','Thinking text tokens'],['tps','Decode tok/s']])}
 {table('unsloth_detail','Unsloth per-prompt results',[['prompt','Prompt'],['suite','Suite'],['wall','Total (s)'],['tokens','Generated tokens'],['thinking','Thinking text tokens'],['quality','Quality pass']])}
 </>}
 <ReportSection id="methods" title="Method" queryId="protocol" showHeading={false}><RichNarrative id="method-text" value={snapshot.methodsText??'## What is held constant\n\nSix quality prompts and ten canonical article prompts; suites are reported separately. Identical request bodies, xhigh thinking, temperature 0, seed 42, 64,000-token output ceiling and 65,536-token context. Prefix caching and forced reasoning closure are disabled. Warmup and model loading are excluded from request timings.\n\nThe finite context prevents truly unlimited output. A length-limited answer is incomplete, never a quality pass. Exact JSON field types and tool arguments are frozen before measurement.'}/></ReportSection>
 <DataComponent id="protocol" queryId="protocol" title="Exact frozen inputs and scoring rules" sourceRows={rows('protocol')} kind="table"><div data-reviewed-rows>{rows('protocol').map(c=><details key={c.prompt} style={{padding:'14px 0',borderBottom:'1px solid #ddd'}}><summary>{c.suite} · {c.prompt}</summary><pre style={{whiteSpace:'pre-wrap',fontSize:'13px'}}>{c.input}</pre><p>{c.criterion}</p><pre style={{whiteSpace:'pre-wrap',fontSize:'13px'}}>{c.expected}</pre>{c.tools!=='[]'&&<pre style={{whiteSpace:'pre-wrap',fontSize:'13px'}}>{c.tools}</pre>}</details>)}</div></DataComponent>
 </article>;
}
