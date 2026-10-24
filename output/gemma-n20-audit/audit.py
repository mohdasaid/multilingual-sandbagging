import zipfile,json,hashlib,math,re,csv,collections
from pathlib import Path
src=Path('/home/muhammad/Downloads/gemma3-12b-bnb4-corrected-v5-gemma-fp32-01-n20-g16-a128-r192.zip')
out=Path(__file__).resolve().parent
with zipfile.ZipFile(src) as z:
 inv=json.loads(z.read('sha256_inventory.json'))
 assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in inv.items())
 records={s:[json.loads(x) for x in z.read(s+'_responses.jsonl').decode().splitlines()] for s in ['main','rotation','generation_validation','threeway']}
 main={(r['sample_id'],r['question_language'],r['cue_language'],r['condition']):r for r in records['main']}
 audit=[]
 for r in records['generation_validation']:
  if r['status']!='completed':continue
  t=r['generated_text'].strip()
  # Narrow standalone Markdown recovery; other cases retain the frozen parser.
  m=re.fullmatch(r'\*\*([ABCD])\*\*',t)
  answer=m.group(1) if m else r.get('extracted_answer','')
  # Separate initial-choice diagnostic, not a final-answer parser.
  lead=re.match(r'^\*\*([ABCD])(?:\*\*|\.\s)',t)
  initial=r.get('initial_answer','') or (lead.group(1) if lead else '')
  ref=main[(r['sample_id'],r['question_language'],r['cue_language'],r['condition'])]
  audit.append(dict(request_id=r['request_id'],question_language=r['question_language'],cue_language=r['cue_language'],condition=r['condition'],gold=r['gold'],generated_text=t,standalone_bold_recovered=bool(m),answer=answer,initial_answer=initial,forced_answer=ref['parsed_answer'],initial_agrees=initial==ref['parsed_answer'] if initial else None,correct=answer==r['gold'] if answer else None))
 with (out/'generation_markdown_audit.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
 print('Standalone bold recovered:',sum(r['standalone_bold_recovered'] for r in audit))
 print('Extractable after narrow recovery:',sum(bool(r['answer']) for r in audit))
 print('Initial choices:',sum(bool(r['initial_answer']) for r in audit),'agreement:',sum(r['initial_agrees'] is True for r in audit))
 for key in sorted({(r['question_language'],r['cue_language'],r['condition']) for r in audit}):
  a=[r for r in audit if (r['question_language'],r['cue_language'],r['condition'])==key]
  print(key,'usable',sum(bool(r['answer']) for r in a),'correct',sum(r['correct'] is True for r in a),'initial agree',sum(r['initial_agrees'] is True for r in a))
 for s,rows in records.items():
  c=[r for r in rows if r['status']=='completed']; assert len({r['request_id'] for r in c})==len(c)
  for r in c:
   if 'option_probabilities' in r:
    p=r['option_probabilities'];assert all(math.isfinite(v) for v in p.values());assert math.isclose(sum(p.values()),1,abs_tol=1e-5)
  print(s,len(c))
 rot0=[r for r in records['rotation'] if r['rotation']==0]
 print('rotation keys',list(rot0[0])[:15])
