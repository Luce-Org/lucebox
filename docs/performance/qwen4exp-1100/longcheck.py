import json,sys,time,requests
base='http://127.0.0.1:'+sys.argv[1]
filler='Lucebox is a GPU inference engine for local language models. It runs on a single integrated GPU with unified memory. '
plant='OPERATIONS NOTE 47-B: the calibration key for the Strix Halo test rig is QUINCE-AMBER-7731, and the rig thermal ceiling is recorded as 54 degrees Celsius. '
reps=(8875-40)//13
records=[]
target_count=None
for position in [0,reps//2]:
 prompt=filler*position+plant+filler*(reps-position)+'\nQuestion: What is the calibration key? Answer verbatim.'
 payload={'model':'dflash','messages':[{'role':'user','content':prompt}],'max_tokens':64,'temperature':0,'stream':False}
 count=requests.post(base+'/v1/messages/count_tokens',json=payload,timeout=60)
 count.raise_for_status(); n=count.json()['input_tokens']
 if target_count is None: target_count=n
 # Match the original benchmark prompt's count; verify actual OpenAI usage below.
 if n != target_count:
  template=filler*position+plant+filler*(reps-position-1)+'\nQuestion: What is the calibration key? Answer verbatim.'
  padding=0
  for attempt in range(5):
   prompt=template.replace('\nQuestion:', ' x'*padding+'\nQuestion:')
   payload['messages'][0]['content']=prompt
   count=requests.post(base+'/v1/messages/count_tokens',json=payload,timeout=60)
   count.raise_for_status();n=count.json()['input_tokens']
   if n==target_count: break
   padding+=target_count-n
   assert padding>=0
  assert n==target_count
 print('LONG CHECK START',position,'matched count',n,flush=True)
 t=time.perf_counter()
 r=requests.post(base+'/v1/chat/completions',json=payload,timeout=1800)
 r.raise_for_status();o=r.json();answer=o['choices'][0]['message']['content'];pt=o['usage']['prompt_tokens']
 row={'plant_position':position,'prompt_tokens':pt,'text':answer,'seconds':time.perf_counter()-t,'ok':pt==16366 and 'QUINCE-AMBER-7731' in answer}
 print(json.dumps(row),flush=True);records.append(row)
open(sys.argv[2],'w').write(json.dumps(records,indent=2))
assert all(r['ok'] for r in records)
