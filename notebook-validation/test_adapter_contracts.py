"""Exercise actual notebook adapter functions using a NumPy-backed tensor double.
This checks adapter control flow/token boundaries, not checkpoint compatibility.
"""
import ast,json,math,re
from types import SimpleNamespace
from contextlib import nullcontext
import numpy as np
import hashlib
from scipy.special import softmax,log_softmax
nb=json.load(open('sec-llm-sandbagging-four-model-corrected-v4.ipynb'))
class Tensor(np.ndarray):
 def __new__(cls,v,dtype=None):return np.asarray(v,dtype=dtype).view(cls)
 def cpu(self):return self
 def float(self):return Tensor(self,np.float32)
 @property
 def device(self):return 'cpu'
class Enc(dict):
 def __getattr__(self,k):return self[k]
 def to(self,device):return self
class Tokenizer:
 pad_token_id=0;eos_token_id=1;model_max_length=10000
 def encode(self,text,**kwargs):return [ord(c) for c in text]
 def decode(self,ids,**kwargs):return ''.join(chr(int(x)) for x in ids if int(x)>1)
 def __call__(self,text,**kwargs):
  ids=Tensor([self.encode(text)],np.int64)
  return Enc(input_ids=ids,attention_mask=Tensor(np.ones_like(ids)))
class Model:
 config=SimpleNamespace(decoder_start_token_id=0,max_position_embeddings=10000)
 def __init__(self):self.calls=[]
 def __call__(self,input_ids=None,decoder_input_ids=None,**kwargs):
  ids=decoder_input_ids if decoder_input_ids is not None else input_ids
  logits=np.zeros((1,ids.shape[1],256));logits[:,:,65]=3;logits[:,:,66]=2
  return SimpleNamespace(logits=Tensor(logits))
 def generate(self,input_ids=None,decoder_input_ids=None,max_new_tokens=None,**kwargs):
  ids=decoder_input_ids if decoder_input_ids is not None else input_ids
  self.calls.append((max_new_tokens,kwargs,ids.copy()))
  return Tensor(np.concatenate([ids,np.full((1,max_new_tokens),ord('x'))],axis=1))
torch=SimpleNamespace(tensor=lambda v,dtype=None,device=None:Tensor(v,dtype),long=np.int64,
 inference_mode=nullcontext,cat=lambda xs,dim:Tensor(np.concatenate(xs,axis=dim)),
 ones=lambda shape,dtype=None,device=None:Tensor(np.ones(shape,dtype=dtype)),ones_like=lambda x:Tensor(np.ones_like(x)),
 stack=lambda xs:Tensor(np.stack(xs)),softmax=lambda x,dim:Tensor(softmax(x,axis=dim)),
 log_softmax=lambda x,dim:Tensor(log_softmax(x,axis=dim)),exp=lambda x:Tensor(np.exp(x)),argmax=np.argmax)
ns=dict(torch=torch,tokenizer=Tokenizer(),input_device='cpu',LETTERS='ABCD',LETTER_SET=set('ABCD'),math=math,re=re,hashlib=hashlib,
 render_chat=lambda p:p+'\n',ANSWER_FIRST_MAX_NEW_TOKENS=7,REASON_FIRST_MAX_NEW_TOKENS=11)
for c in nb['cells']:
 if c['cell_type']!='code':continue
 s=''.join(c['source'])
 if s.startswith('%'):continue
 nodes=[x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name in {'encode_checked','score_prompt','generate_text','generate_after_committed_answer','answer_first','reason_first','explicit_letter'}]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'adapter','exec'),ns)
ns['REASON_FINAL_RE']=re.compile(r'Answer\s*:\s*([ABCD])\b',re.I)
for key,arch in [('atlas','causal'),('aya101','seq2seq'),('gemma3','causal'),('afrique','causal')]:
 ns['CFG']={'architecture':arch};ns['model']=Model()
 for multi in [False,True]:
  candidates={l:([32,ord(l)] if multi else [ord(l)]) for l in 'ABCD'}
  ns['candidate_ids_by_prefix']={'en':candidates,'threeway':candidates}
  result=ns['score_prompt']('question','Answer:','threeway')
  assert result['parsed_answer']=='A'
  assert abs(sum(result['option_probabilities'].values())-1)<1e-6
  assert result['scoring_mode']==('sequence_log_likelihood' if multi else 'single_token')
  r=ns['answer_first']('question')
  assert r['argmax_generation_agree'] and r['output_tokens']==7+len(candidates['A'])
  assert ns['model'].calls[-1][0]==7
  text,n=ns['generate_text']('question',3,'Answer:')
  assert text=='xxx' and n==3
  r=ns['reason_first']('question');assert not r['valid'] and r['output_tokens']==11
 print('PASS adapter contract',key,'single/multi-token, budgets, extraction')
# conservative parser regressions
for text in ['','Vitamin A is mentioned.','I considered A but cannot decide.','Answer: A\nAnswer: B']:
 assert ns['explicit_letter'](text)==''
for text in ['A','Answer: B','Final answer: C']:
 assert ns['explicit_letter'](text) in 'ABC'
print('PASS parser contracts')
