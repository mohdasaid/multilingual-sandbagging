import zipfile,json,math,hashlib,csv,io,collections,re
from pathlib import Path
z=zipfile.ZipFile('/home/muhammad/Downloads/aya101-corrected-v5-aya-n20-g16-a128-r192.zip')
report=[]
def log(x):report.append(str(x));print(x)
inv=json.loads(z.read('sha256_inventory.json'));assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in inv.items());log(f'{len(inv)} inventory hashes match')
D={}
for s,n in [('main',320),('rotation',1280),('generation_validation',320),('threeway',360)]:
 rows=[json.loads(x) for x in z.read(s+'_responses.jsonl').decode().splitlines()]
 log((s,'statuses',dict(collections.Counter(r['status'] for r in rows))))
 rows=[r for r in rows if r['status']=='completed'];assert len(rows)==n==len({r['request_id'] for r in rows});D[s]=rows
 for r in rows:
  if 'option_probabilities' not in r:continue
  p=r['option_probabilities'];ll=r['candidate_log_likelihoods'];assert all(math.isfinite(v) for v in [*p.values(),*ll.values()])
  mx=max(ll.values());den=sum(math.exp(v-mx) for v in ll.values())
  assert all(math.isclose(p[k],math.exp(ll[k]-mx)/den,abs_tol=1e-6) for k in p)
  assert math.isclose(sum(p.values()),1,abs_tol=1e-5)
  assert r['parsed_answer']==max(p,key=p.get)
  assert r['correct']==(r['parsed_answer']==r['gold'])
  if 'gold_probability' in r:assert r['gold_probability']==p[r['gold']]
log('All saved scoring vectors finite, normalized, consistent with log likelihoods, argmax and correctness')
key=lambda r:(r.get('base_sample_id',r['sample_id']),r['question_language'],r['cue_language'],r['condition'])
main={key(r):r for r in D['main']}
for r in D['rotation']:
 base=main[key(r)];rot=r['rotation'];letters='ABCD'
 assert r['gold']==letters[(letters.index(base['gold'])-rot)%4]
 assert r['selected_base_content']==(letters.index(r['parsed_answer'])+rot)%4
 if rot==0:assert r['option_probabilities']==base['option_probabilities']
log('All rotated gold/selected-content mappings agree; rotation-0 probabilities match main exactly')
for r in D['generation_validation']:
 assert r['prompt']==main[key(r)]['prompt']
 assert r['extracted_answer']==main[key(r)]['parsed_answer']
 assert r['correct']==(r['extracted_answer']==r['gold'])
log('320/320 extractable generated answers match main forced choices under identical saved prompts')
# Recompute pooled point estimates directly from responses.
for s,fn in [('main','main_primary_pooled_effects.csv'),('rotation','rotation_pooled_accuracy_effects.csv')]:
 groups=collections.defaultdict(list)
 for r in D[s]:groups[(key(r)[0],r['question_language'],r['condition'])].append(float(r['correct']))
 avg={k:sum(v)/len(v) for k,v in groups.items()}
 ids=sorted({k[0] for k in avg})
 for lang in ['en','ha']:
  dif=[avg[i,lang,'direct_sandbag']-avg[i,lang,'honest'] for i in ids]
  log((s,lang,'direct-minus-honest accuracy',sum(dif)/len(dif)))
rows=list(csv.DictReader(io.StringIO(z.read('answer-analysis-v5/threeway_answer_audit.csv').decode())))
reason=[r for r in rows if r['method']=='reason_first'];log(('reason-first order eligible',sum(r['reason_first_order_eligible']=='True' for r in reason),'of',len(reason)))
raw=[r for r in D['threeway'] if r['method']=='reason_first']
log(('reason-first example outputs',[r['generated_text'] for r in raw[:5]]))
Path(__file__).with_name('checks.txt').write_text('\n'.join(report)+'\n')
