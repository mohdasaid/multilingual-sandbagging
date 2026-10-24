import json,zipfile,collections
from pathlib import Path
n=json.loads(Path('sandbagging-narrow-study-v10-four-models.ipynb').read_text());ns={};exec(''.join(n['cells'][6]['source']),ns)
cases=[('B. Old answer\n\nTherefore, the correct answer is:\nB. Old answer\n\nSo the correct answer should be:\nC. Revised answer','conflicting'),('Answer:\nA','correct'),('The correct answer is:\nA or B','unresolved'),('A. one\nB. two\nC. three\nD. four','unresolved'),('A cikin wani abu','unresolved'),('**A, B, C, D**','unresolved'),('The correct answer should be:\nA','correct'),('A. one\nB. two\nAnswer: A','correct')]
for text,expected in cases:assert ns['generation_outcome'](ns['extract_answer'](text),'A')==expected,(text,ns['extract_answer'](text))
root=Path('output/conflict-audit-v10');root.mkdir(parents=True,exist_ok=True)
changes=[]
for stem in ['qwen3-4b-instruct-2507-nf4-narrow-v9-reviewed-01-n50-g64','qwen3-4b-instruct-2507-nf4-narrow-v9-reviewed-01-n50-g200','gemma3-4b-it-nf4-narrow-v9-reviewed-01-n50-g64']:
 z=zipfile.ZipFile('/home/muhammad/Downloads/'+stem+'.zip')
 for filename in ['generation_validation_responses.jsonl','control_responses.jsonl']:
  for r in map(json.loads,z.read(filename).decode().splitlines()):
   p=ns['extract_answer'](r['generated_text']);outcome=ns['generation_outcome'](p,r['gold'])
   if outcome!=r['generation_outcome']:
    changes.append(dict(archive=stem,file=filename,id=r.get('sample_id',r.get('control_id')),question_language=r['question_language'],cue_language=r['cue_language'],condition=r.get('condition'),old=r['generation_outcome'],new=outcome,generated_text=r['generated_text'],parsed=p))
assert len(changes)==2,changes
assert all(r['old']=='unresolved' and r['new']=='conflicting' and r['question_language']=='en' and r['archive'].endswith('g200') for r in changes)
(root/'changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
print('PASS 8 parser regressions; re-parsed 1,662 saved responses: only the 2 expected English g200 conflicts changed; archives untouched.')
# Verify all adapters and Granite's exact native template option forwarding.
config={};exec(''.join(n['cells'][4]['source']),config)
assert len(config['MODEL_CONFIGS'])==4
import ast
loader=''.join(n['cells'][10]['source']);func=next(x for x in ast.parse(loader).body if isinstance(x,ast.FunctionDef) and x.name=='render_chat')
class Tokenizer:
 chat_template='test'
 def apply_chat_template(self,messages,**kwargs):return kwargs
for key,cfg in config['MODEL_CONFIGS'].items():
 env=dict(processor=None,tokenizer=Tokenizer(),CFG=cfg)
 exec(compile(ast.Module(body=[func],type_ignores=[]),'adapter','exec'),env)
 kw=env['render_chat']('hello');assert kw['add_generation_prompt'] and not kw['tokenize']
 if key=='granite2':assert kw['thinking'] is False
 else:assert 'thinking' not in kw
print('PASS four adapter configs and explicit Granite thinking=False forwarding (no GPU loading tested).')
