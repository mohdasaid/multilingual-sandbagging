"""Offline integration tests: synthetic data/model, actual notebook orchestration and reports.
Does not validate real GPU model loading or inference.
"""
import ast, hashlib, json, io, contextlib, tempfile, types, sys
from pathlib import Path
import numpy as np
import pandas as pd
NB=Path(__file__).resolve().parents[1]/'sandbagging-narrow-study-v8-controls.ipynb'
nb=json.loads(NB.read_text())
codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
source='\n'.join(codes)
for banned in ['target_30','RUN_THREEWAY','def reason_first','def answer_first','threeway_items','shutil.move','capability_qualified_results']:
 assert banned not in source,banned
payload='\n\n'.join(c for c in codes if not c.startswith('IMPLEMENTATION_SHA256'))
assert hashlib.sha256(payload.encode()).hexdigest() in codes[1]
for i,c in enumerate(codes):
 if not c.startswith('%'):compile(c,f'cell{i}','exec')

def pick(start):return next(c for c in codes if c.startswith(start))

def run(n):
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);fixture=root/'review.csv'
  rows=[]
  for i in range(400):
   gold='ABCD'[i%4]
   r=dict(sample_id=f'test/{i}',answer_en=gold,question_en=f'Question {i}',question_ha_reviewed=f'Tambaya {i}',review_status='approved',review_notes='synthetic')
   for suffix in ['en','ha_reviewed']:
    for x in 'ABCD':r[f'option_{x.lower()}_{suffix}']='truth' if x==gold else 'distractor_'+x
   rows.append(r)
  pd.DataFrame(rows).to_csv(fixture,index=False)
  fake_torch=types.SimpleNamespace(manual_seed=lambda x:None,cuda=types.SimpleNamespace(is_available=lambda:True,device_count=lambda:2,get_device_name=lambda i:'mock'),version=types.SimpleNamespace(cuda='mock'))
  ns={'display':lambda *a:None,'__name__':'notebook_test','torch':fake_torch}
  exec(codes[1],ns)
  exec(pick('MODEL_KEY').replace('N_ITEMS = 20',f'N_ITEMS = {n}'),ns)
  exec(pick('"""Conservative'),ns)
  for text,expected in [('**A**','A'),('Amsar: B','B'),('Answer: A\nAnswer: B',''),('A. one\nB. two',''),('', '')]:
   assert ns['extract_answer'](text)['extracted_answer']==expected
  setup=pick('import gc,')
  setup=setup.replace('import torch','')
  setup=setup.replace('matches = list(Path("/kaggle/input").rglob("confirmatory_review_400.csv"))',f'matches = [Path({str(fixture)!r})]')
  exec(setup,ns)
  assert ns['PLANNED_COUNTS']=={'main':9*n,'generation_validation':9*min(50,n),'rotation':36*n}
  assert ns['balanced_items'].sample_id.nunique()==n
  counts=ns['balanced_items'].answer_en.value_counts().reindex(list('ABCD'),fill_value=0)
  assert counts.max()-counts.min()<=1
  assert set(ns['generation_items'].sample_id)<=set(ns['balanced_items'].sample_id)
  # Mock only inference; execute actual job scheduling, persistence, analyses, archive.
  def score(prompt,prefix,key):
   gold=next(line[0] for line in prompt.splitlines() if line[1:]=='. truth')
   under=('underestimate' in prompt or 'mai tantancewa' in prompt)
   answer='ABCD'[('ABCD'.index(gold)+1)%4] if under else gold
   p={x:(.7 if x==answer else .1) for x in 'ABCD'}
   return dict(parsed_answer=answer,option_probabilities=p,candidate_log_likelihoods={x:float(np.log(v)) for x,v in p.items()},candidate_mass=.9)
  def generate(prompt,budget,prefill=''):
   a=score(prompt,prefill,'en')['parsed_answer']
   return '**'+a+'**',min(3,budget)
  ns.update(OUT=root/'out',EXPERIMENT_VERSION='mock',MODEL_ID='mock',model_revision='mock',EFFECTIVE_COMPUTE_DTYPES=['mock'],GENERATION_EOS_IDS=[1,2],processor=None,tokenizer=types.SimpleNamespace(chat_template='mock'),model=types.SimpleNamespace(dtype='mock',generation_config=types.SimpleNamespace(to_dict=lambda:{})),render_chat=lambda s:s,candidate_ids_by_prefix={'en':{},'ha':{}},score_prompt=score,generate_text=generate,explicit_letter=lambda t:ns['extract_answer'](t)['extracted_answer'])
  ns['OUT'].mkdir()
  exec(pick('# Independent, synthetic'),ns)
  exec(pick('manifest =').replace('version(p)','"mock"'),ns)
  exec(pick('manifest =').replace('version(p)','"mock"'),ns)  # frozen-manifest resume
  helpers=pick('def recover_jsonl').replace('draws=10_000','draws=100')
  exec(helpers,ns)
  assert all(np.isnan(x) for x in ns['bootstrap_ci']([]))
  try:ns['bootstrap_ci']([float('nan')]);raise AssertionError('NaN accepted')
  except ValueError:pass
  exec(pick('preflight=[]'),ns)
  jobs=ns['build_control_jobs']()
  assert len(jobs)==104
  assert {j['control_id'] for j in jobs}.isdisjoint(set(ns['balanced_items'].sample_id))
  for kind in ['passage_comprehension','basic_capability']:
   for pair in ns['LANGUAGE_PAIRS']:
    subset=[j for j in jobs if j['control_kind']==kind and (j['question_language'],j['cue_language'])==pair]
    assert len(subset)==16
    assert {x:sum(j['gold']==x for j in subset) for x in 'ABCD'}==dict.fromkeys('ABCD',4)
  expected_values={'lookup_1':'3','lookup_2':'5','lookup_3':'2','lookup_4':'7','arithmetic_1':'5','arithmetic_2':'5','arithmetic_3':'8','arithmetic_4':'6'}
  for j in jobs:
   if j['control_id'] in expected_values:
    assert f"{j['gold']}. {expected_values[j['control_id']]}" in j['prompt'].splitlines()
   else: assert j['gold']==j['control_id'][-1]
  answers={j['prompt']:j['gold'] for j in jobs}
  def control_score(prompt,prefix,key):
   if prompt not in answers:return score(prompt,prefix,key)
   a=answers[prompt];p={x:.7 if x==a else .1 for x in 'ABCD'}
   return dict(parsed_answer=a,option_probabilities=p,candidate_log_likelihoods={x:float(np.log(v)) for x,v in p.items()},candidate_mass=.9)
  def control_generate(prompt,budget,prefill=''):
   if prompt not in answers:return generate(prompt,budget,prefill)
   return ('',0) if answers[prompt]=='A' else (answers[prompt],1)
  ns.update(score_prompt=control_score,generate_text=control_generate)
  exec(pick('control_jobs='),ns)
  cs=pd.read_csv(ns['OUT']/'control_summary.csv')
  assert len(cs)==8 and (cs.independent_items==4).all()
  assert (cs.forced_accuracy==1).all()
  assert (cs.answer_coverage==.75).all()
  assert (cs.generated_success_rate==.75).all()
  before_control=(ns['OUT']/'control_responses.jsonl').read_bytes()
  exec(pick('control_jobs='),ns)
  assert (ns['OUT']/'control_responses.jsonl').read_bytes()==before_control

  for prefix in ['if RUN_MAIN:','if RUN_GENERATION_VALIDATION:','if RUN_ROTATION_AUDIT:','required_outputs=']:
   exec(pick(prefix),ns)
  out=ns['OUT']
  for stage,count in ns['PLANNED_COUNTS'].items():
   if stage=='controls':stage='control'
   records=[json.loads(x) for x in (out/f'{stage}_responses.jsonl').read_text().splitlines()]
   assert len(records)==count==len({r['request_id'] for r in records})
  eff=pd.read_csv(out/'primary_accuracy_effects.csv')
  assert (eff.loc[eff.contrast.eq('direct_sandbag_minus_honest'),'estimate']==-1).all()
  assert (eff.paired_items==n).all()
  interactions=pd.read_csv(out/'primary_accuracy_interactions.csv')
  assert (interactions.estimate==0).all()
  gen=pd.read_csv(out/'generation_validation_summary.csv')
  assert (gen.coverage==1).all() and (gen.strict_format_rate==0).all()
  assert (gen.agreement_among_extractable==1).all()
  # Resume must not append duplicate records.
  before=(out/'main_responses.jsonl').read_bytes()
  exec(pick('if RUN_MAIN:'),ns)
  assert (out/'main_responses.jsonl').read_bytes()==before
  inventory=json.loads((out/'sha256_inventory.json').read_text())
  assert all(hashlib.sha256((out/name).read_bytes()).hexdigest()==digest for name,digest in inventory.items())
  # Pair-count reporting on empty comparisons is undefined, not zero effect.
  f=pd.DataFrame([dict(sample_id='x',question_language='en',cue_language='en',condition=c,correct=np.nan) for c in ['honest','direct_sandbag','unlocked']])
  assert ns['paired_effects'](f).estimate.isna().all()
  return len(inventory)
for n in [1,20]:
 with contextlib.redirect_stdout(io.StringIO()):count=run(n)
 print(f'PASS N_ITEMS={n}: independent control counts, gold mapping, missing-answer reports, resume, main analyses, hashes ({count} artifacts)')
print('PASS syntax, executable-source hash, removed modules, non-finite guards; GPU inference NOT tested')
