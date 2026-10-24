import zipfile,json,hashlib,re,io
from pathlib import Path
import pandas as pd
import numpy as np
Z=zipfile.ZipFile('/home/muhammad/Downloads/natlas-corrected-v4-n20-g16-a128-r192.zip')
OUT=Path(__file__).parent
load=lambda name:pd.DataFrame([json.loads(l) for l in Z.read(name).splitlines()])
checks={}
checks['hashes_match']=all(hashlib.sha256(Z.read(k)).hexdigest()==v for k,v in json.loads(Z.read('sha256_inventory.json')).items())
frames={k:load(v) for k,v in {'main':'main_responses.jsonl','generation':'generation_validation_responses.jsonl','rotation':'rotation_responses.jsonl','threeway':'threeway_responses.jsonl'}.items()}
for k,expected in [('main',320),('generation',320),('rotation',1280),('threeway',360)]:
 f=frames[k];assert len(f)==expected and f.request_id.is_unique and f.status.eq('completed').all()
 checks[k+'_count']=len(f)
 assert ((f.parsed_answer==f.gold)&f.parsed_answer.isin(list('ABCD'))).equals(f.correct)
 if k in ['main','rotation']:
  assert np.allclose(f.option_probabilities.map(lambda d:sum(d.values())),1)
  assert np.allclose(f.apply(lambda r:r.option_probabilities[r.gold],axis=1),f.gold_probability)
  assert f.apply(lambda r:max(r.option_probabilities,key=r.option_probabilities.get),axis=1).equals(f.parsed_answer)
items=load('balanced_items.jsonl');assert items.sample_id.is_unique and len(items)==20
checks['gold_positions']=items.answer_en.value_counts().sort_index().to_dict()
for file in ['generation_items.jsonl','rotation_base_items.jsonl','threeway_items.jsonl']:
 assert set(load(file).sample_id)==set(items.sample_id)
# Rotation zero must reproduce the same base prompt/decision distribution.
r=frames['rotation'];m=frames['main'];keys=['question_language','cue_language','condition']
j=r[r.rotation.eq(0)].merge(m,left_on=['base_sample_id']+keys,right_on=['sample_id']+keys,suffixes=('_rot','_main'))
assert len(j)==320 and (j.parsed_answer_rot==j.parsed_answer_main).all()
assert np.allclose(j.gold_probability_rot,j.gold_probability_main)
checks['rotation_zero_matches_main']=True
summary=pd.read_csv(io.BytesIO(Z.read('main_summary.csv')))
actual=m.groupby(keys).agg(responses=('correct','size'),accuracy=('correct','mean'),mean_gold_probability=('gold_probability','mean')).reset_index()
j=summary.merge(actual,on=keys,suffixes=('_saved','_audit'))
for c in ['responses','accuracy','mean_gold_probability']: assert np.allclose(j[c+'_saved'],j[c+'_audit'])
checks['main_summary_recomputed']=True
leading=re.compile(r'\A[ \t]*([ABCD])(?:[.)](?:[ \t]+|$)|[ \t]*(?:\n|$))')
markers=re.compile(r'^[ \t]*(?:Answer|Final answer|Selected answer|Amsa|Amsar da na zaɓa)[ \t]*:[ \t]*([ABCD])\b',re.I|re.M)
phrase=re.compile(r'\b(?:the correct answer is|the answer is|the correct option is|the final answer is)[ \t]*([ABCD])\b',re.I)
g=frames['generation'].copy()
g['initial_answer']=g.generated_text.map(lambda t:leading.match(t).group(1) if leading.match(t) else '')
g['initial_answer_valid']=g.initial_answer.isin(list('ABCD'))
g['initial_agreement']=g.initial_answer.eq(g.forced_answer)
g['later_conflicting_explicit_answer']=g.apply(lambda r:any(x.upper()!=r.initial_answer for x in markers.findall(r.generated_text)),axis=1)
g['initial_correct']=g.initial_answer.eq(g.gold)
g['unambiguous_explicit_answer']=g.initial_answer_valid & ~g.later_conflicting_explicit_answer
g.to_csv(OUT/'generation_extraction_audit.csv',index=False)
gs=g.groupby(keys).agg(responses=('correct','size'),strict_coverage=('valid','mean'),initial_answer_coverage=('initial_answer_valid','mean'),initial_agreement=('initial_agreement','mean'),initial_accuracy=('initial_correct','mean'),conflicting_responses=('later_conflicting_explicit_answer','sum')).reset_index()
gs.to_csv(OUT/'generation_summary_audit.csv',index=False)
checks['generation_initial_answers']=int(g.initial_answer_valid.sum());checks['generation_initial_agreements']=int(g.initial_agreement.sum());checks['generation_conflicts']=int(g.later_conflicting_explicit_answer.sum());checks['generation_strict_valid']=int(g.valid.sum())
# Exploratory deterministic re-extraction; no inference from rationale, no gold-based extraction.
t=frames['threeway'];t=t[t.method.eq('reason_first')].copy()
def extract(text):
 lead=leading.match(text)
 candidates=set(x.upper() for x in markers.findall(text)+phrase.findall(text))
 if lead:candidates.add(lead.group(1))
 return (next(iter(candidates)),'explicit') if len(candidates)==1 else ('','conflicting' if candidates else 'unresolved')
t[['audit_answer','audit_status']]=pd.DataFrame(t.generated_text.map(extract).tolist(),index=t.index)
t['audit_valid']=t.audit_answer.isin(list('ABCD'));t['audit_correct']=t.audit_valid & t.audit_answer.eq(t.gold)
t.to_csv(OUT/'reason_first_extraction_audit.csv',index=False)
ts=t.groupby(['question_language','condition']).agg(responses=('valid','size'),strict_valid=('valid','sum'),audit_valid=('audit_valid','sum'),audit_correct=('audit_correct','sum'),token_limit=('hit_token_limit','sum')).reset_index()
ts.to_csv(OUT/'reason_first_summary_audit.csv',index=False)
checks['reason_first_audit']=ts.to_dict('records')
(OUT/'verification.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
