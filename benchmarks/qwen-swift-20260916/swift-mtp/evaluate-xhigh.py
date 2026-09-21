import json,time,urllib.request,pathlib,sys
from evaluation_helpers import score_message
base,model,label,outdir=sys.argv[1:];out=pathlib.Path(outdir);out.mkdir(parents=True,exist_ok=True)
cases=[
('arithmetic','A machine makes 17 parts every 6 minutes. It runs for 3 hours 24 minutes, but is stopped for 18 minutes during that interval. It starts a fresh cycle when restarted. Both active intervals are multiples of 6 minutes. How many parts? Return JSON only: {"parts": integer}.',{'parts':527}),
('constraints','Schedule tasks A,B,C,D, each one hour, on one machine starting at hour 0. A must precede C, B must precede D, and D must precede A. Return JSON only with the unique order: {"order":[...]}.',{'order':['B','D','A','C']}),
('code_aliasing','What does Python print?\na = [[0]] * 3\na[0].append(1)\na[1] = [2]\na[2][0] = 9\nprint(a)\nReturn JSON only with result as an array of arrays of integers, not a string: {"result": [[...], ...]}.',{'result':[[9,1],[2],[9,1]]}),
('code_intervals','A function merges CLOSED integer intervals when they overlap, but NOT merely when adjacent. Given [[8,10],[1,3],[3,6],[7,7],[12,12],[10,11]], return the sorted merged list as JSON only: {"intervals":[...]}.',{'intervals':[[1,6],[7,7],[8,11],[12,12]]}),
('extraction','Data: [{"id":"a","active":true,"score":7},{"id":"b","active":false,"score":99},{"id":"c","active":true,"score":7},{"id":"d","active":true,"score":4}]. Select active items, sort by score descending then id descending, take first two. Return JSON only: {"ids":[...]}.',{'ids':['c','a']}),
('tool_call','Use lookup_weather to check Toronto in Celsius.',{'city':'Toronto','unit':'celsius'})]
rows=[]
for name,prompt,expected in cases:
 body={'model':model,'messages':[{'role':'user','content':prompt}],'max_tokens':32768,'temperature':0,'reasoning_effort':'xhigh','seed':42}
 if label=='current':body['extra_body']={'lucebox_cache':{'mode':'exact'}}
 if name=='tool_call':body['tools']=[{'type':'function','function':{'name':'lookup_weather','description':'Look up weather for a city','parameters':{'type':'object','properties':{'city':{'type':'string'},'unit':{'type':'string','enum':['celsius','fahrenheit']}},'required':['city','unit']}}}]
 t=time.monotonic();req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=240) as r:response=json.load(r)
 elapsed=time.monotonic()-t;choice=response['choices'][0];msg=choice['message']
 row={'name':name,'wall_seconds':elapsed,'passed':score_message(name,msg,choice['finish_reason'],expected),'expected':expected,'response':response,'request':body};rows.append(row)
 (out/(label+'.json')).write_text(json.dumps(rows,indent=2));print(json.dumps({'name':name,'seconds':elapsed,'passed':row['passed'],'usage':response.get('usage'),'finish':choice['finish_reason']}),flush=True)
print(json.dumps({'label':label,'passed':sum(x['passed'] for x in rows),'total':len(rows),'seconds':sum(x['wall_seconds'] for x in rows)}),flush=True)
