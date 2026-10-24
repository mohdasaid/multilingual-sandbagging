from pathlib import Path
import json,zipfile,pandas as pd
ns={};exec(Path('output/parser-v11/answer_parser.py').read_text(),ns)
cases=[('D.\n\nExplanation','D',False),('**A.**\nExplanation','A',False),('A cikin wani abu','',False),('A, B, C, D','',False),('A. first\nB. second','',False),('Answer: A, B, C, D','',False),('Jibu ni: B','B',False),('Jibu ni C.','C',False),('Kwa hivyo, jibu ni: B.','B',False),('Question, kwa hivyo jibu ni C. 6.','C',False),('Jibu ni A au B','',False),('Jibu ni: A, B','',False),('Someone said "Jibu ni: B"','',False),('Nawa ya rage ni C.','',False),('A, B, C, D\n\nJibu ni: B','B',False),('Answer: A\nJibu ni B','',True),('The correct answer is:\nB\nSo the correct answer should be:\nC','',True),('D.\nExplanation\nAnswer: B','',True)]
for text,answer,conflict in cases:
 p=ns['extract_answer'](text);assert (p['extracted_answer'],p['answer_conflict'])==(answer,conflict),(text,p)
print('PASS',len(cases),'positive/negative parsing regressions')
files=['llama3.2-3b-instruct-nf4-narrow-v10-four-models-01-n20-g64','qwen3-4b-instruct-2507-nf4-narrow-v9-reviewed-01-n50-g64','qwen3-4b-instruct-2507-nf4-narrow-v9-reviewed-01-n50-g200','gemma3-4b-it-nf4-narrow-v9-reviewed-01-n50-g64']
changes=[];summary=[];total=0
for name in files:
 z=zipfile.ZipFile('/home/muhammad/Downloads/'+name+'.zip')
 for stage in ['generation_validation','control']:
  records=[]
  for r in map(json.loads,z.read(stage+'_responses.jsonl').decode().splitlines()):
   total+=1;p=ns['extract_answer'](r['generated_text']);outcome=ns['generation_outcome'](p,r['gold'])
   item=dict(archive=name,stage=stage,id=r.get('sample_id',r.get('control_id')),question_language=r['question_language'],cue_language=r['cue_language'],condition=r.get('condition',r.get('control_kind')),old=r['generation_outcome'],new=outcome,old_extractable=r['answer_extractable'],new_extractable=p['answer_extractable'],generated_text=r['generated_text'],**p)
   records.append(item)
   if item['old']!=item['new']:changes.append(item)
  df=pd.DataFrame(records)
  for keys,g in df.groupby(['question_language','cue_language','condition']):
   summary.append(dict(archive=name,stage=stage,question_language=keys[0],cue_language=keys[1],condition=keys[2],responses=len(g),old_extractable=int(g.old_extractable.sum()),new_extractable=int(g.new_extractable.sum()),new_correct=int(g.new.eq('correct').sum()),new_conflicting=int(g.new.eq('conflicting').sum())))
root=Path('output/parser-v11');pd.DataFrame(changes).to_csv(root/'saved_response_changes.csv',index=False);pd.DataFrame(summary).to_csv(root/'coverage_audit.csv',index=False)
print('Reparsed',total,'responses; changed',len(changes))
print(pd.DataFrame(changes).groupby(['archive','stage','old','new']).size().to_string())
assert sum(c['archive'].startswith('llama') and c['stage']=='generation_validation' for c in changes)==8
assert not any(c['archive'].startswith('qwen') and c['cue_language']=='ha' for c in changes)
