import ast,json,hashlib,random,re,math,os,time,tempfile,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
nb=json.load(open('sec-llm-sandbagging-four-model-corrected-v4.ipynb'))
sources=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
def get(x):return next(s for s in sources if x in s)
for n in [1,10,20,400]:
 ns=dict(json=json,hashlib=hashlib,random=random,re=re,math=math,os=os,time=time,Path=Path,datetime=datetime,timezone=timezone,np=np,pd=pd,display=lambda *a:None,shutil=shutil)
 exec(get('MODEL_KEY ='),ns); ns['N_ITEMS']=n
 shared=get('import gc,')
 shared=shared.replace('import torch','').replace('torch.manual_seed(SEED)','')
 a=shared.index('assert torch.cuda.is_available()');b=shared.index('matches =',a)
 shared=shared[:a]+shared[b:]
 shared=shared.replace('Path("/kaggle/input")','Path("multilingual-evaluation-pilot/confirmatory-data")')
 exec(shared,ns)
 assert len(ns['balanced_items'])==n
 exec(get('METHOD_INSTRUCTIONS='),ns)
 exec(get('def run_jobs'),ns)
 # quick numerical validation, full notebook keeps 10k bootstrap draws
 orig=ns['bootstrap_ci'];ns['bootstrap_ci']=lambda x:orig(x,draws=20)
 ns['canonical_hash']=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
 def score(*args):
  return dict(parsed_answer='A',option_probabilities=dict(A=.4,B=.3,C=.2,D=.1),candidate_mass=.8,top_two_margin=.1,input_tokens=10)
 ns['score_prompt']=score
 ns['generate_text']=lambda *args,**kwargs:('A',1)
 # extract parser only
 tree=ast.parse(get('def explicit_letter'))
 f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='explicit_letter')
 exec(compile(ast.Module(body=[f],type_ignores=[]),'parser','exec'),ns)
 assert ns['explicit_letter']('Vitamin A; no final answer')==''
 assert ns['explicit_letter']('Answer: B')=='B'
 ns['answer_first']=lambda prompt:dict(generated_text='Answer: A',parsed_answer='A',valid=True,malformed=False,output_tokens=2,hit_token_limit=False,pre_generation_argmax='A',argmax_generation_agree=True)
 ns['reason_first']=lambda prompt:dict(generated_text='Answer: A',parsed_answer='A',valid=True,malformed=False,output_tokens=2,hit_token_limit=False)
 ns['EXPERIMENT_VERSION']='test';ns['CFG']={'architecture':'causal'}
 with tempfile.TemporaryDirectory() as t:
  ns['OUT']=Path(t)
  # silence progress only
  ns['print']=lambda *a,**k:None
  for key in ['MAIN_PATH =','GEN_PATH=','rotation_records=[]','THREE_PATH=']:
   exec(get(key),ns)
  print('PASS count',n,'rows',len(ns['main_results']),len(ns['three']))
  # resume identical jobs must not execute a worker
  ns['run_jobs'](ns['main_jobs'],ns['MAIN_PATH'],lambda job: (_ for _ in ()).throw(Exception('unexpected worker')))
print('all synthetic pipelines passed')
