# Run this as ONE NEW CELL at the bottom of the completed v8 notebook.
# Reuses its loaded model. Does not reload weights or modify research files.
import json, re, hashlib, shutil
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import torch

def run_prompt_diagnostic():
    required=['model','tokenizer','generate_text','score_prompt','extract_answer','render_chat','CONTROL_ITEMS','ANSWER_PREFIXES','candidate_ids_by_prefix','MODEL_ID']
    missing=[k for k in required if k not in globals()]
    if missing:raise RuntimeError('Run this in the loaded v8 session. Missing: '+', '.join(missing))
    assert MODEL_ID in ['google/gemma-3-4b-it','Qwen/Qwen3-4B-Instruct-2507'], MODEL_ID
    items=[dict(x) for x in CONTROL_ITEMS if x['kind']=='basic_capability']
    assert len(items)==4
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest=Path('/kaggle/working')/('prompt-diagnostic-'+MODEL_ID.split('/')[-1]+'-'+stamp)
    dest.mkdir(exist_ok=False)
    budget=16
    rows=[]
    def save(r):
        rows.append(r)
        with (dest/'responses.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    for item in items:
        number=item['options'][item['gold_index']]
        for lang in ['en','ha']:
            numeric_instruction={'en':'Return only the numeric answer, without explanation.',
              'ha':'Ka rubuta lambar amsar kawai, ba tare da bayani ba.'}[lang]
            prompt=numeric_instruction+'\n\n'+item[lang]
            text,n=generate_text(prompt,budget,prefill='')
            clean=text.strip().strip('*`_ ').strip()
            valid=bool(re.fullmatch(r'\d+',clean))
            save(dict(item_id=item['id'],question_language=lang,cue_language=lang,rotation=-1,
                method='direct_numeric',prompt=prompt,prefix='',gold=number,generated_text=text,
                extracted=clean if valid else '',extractable=valid,correct=valid and clean==number,
                output_tokens=n,hit_token_limit=n>=budget))
        for qlang,clang in [('en','en'),('ha','en'),('ha','ha')]:
            for rotation in range(4):
                order=[(i+rotation)%4 for i in range(4)]
                gold='ABCD'[order.index(item['gold_index'])]
                options='\n'.join(f"{x}. {item['options'][i]}" for x,i in zip('ABCD',order))
                instruction={'en':'Return only one option letter: A, B, C, or D.',
                  'ha':'Ka rubuta harafin amsa guda ɗaya kawai: A, B, C, ko D.'}[clang]
                prompt=instruction+'\n\n'+item[qlang]+'\n'+options
                variants=[('no_prefix','',None),('english_prefix',ANSWER_PREFIXES['en'],'en')]
                if clang=='ha':variants.append(('hausa_prefix',ANSWER_PREFIXES['ha'],'ha'))
                for method,prefix,key in variants:
                    text,n=generate_text(prompt,budget,prefill=prefix)
                    parsed=extract_answer(text)
                    r=dict(item_id=item['id'],question_language=qlang,cue_language=clang,
                       rotation=rotation,method=method,prompt=prompt,prefix=prefix,gold=gold,
                       generated_text=text,extracted=parsed['extracted_answer'],extractable=parsed['answer_extractable'],
                       correct=parsed['answer_extractable'] and parsed['extracted_answer']==gold,
                       output_tokens=n,hit_token_limit=n>=budget)
                    if key is not None:
                        scored=score_prompt(prompt,prefix,key)
                        r.update(forced_answer=scored['parsed_answer'],forced_correct=scored['parsed_answer']==gold,
                           option_probabilities=scored['option_probabilities'],candidate_mass=scored['candidate_mass'])
                        # Independently recompute the single-token score from the same rendered input.
                        rendered=render_chat(prompt)+prefix
                        enc=tokenizer(rendered,add_special_tokens=False,return_tensors='pt').to(model.get_input_embeddings().weight.device)
                        candidates=candidate_ids_by_prefix[key]
                        assert all(len(v)==1 for v in candidates.values())
                        with torch.inference_mode():logits=model(**enc).logits[0,-1].float()
                        assert torch.isfinite(logits).all()
                        independent=torch.softmax(logits[torch.tensor([candidates[x][0] for x in 'ABCD'],device=logits.device)],dim=0).cpu().tolist()
                        r['independent_score_max_difference']=max(abs(a-scored['option_probabilities'][x]) for a,x in zip(independent,'ABCD'))
                        r['rendered_input']=rendered
                        base=tokenizer.encode(rendered,add_special_tokens=False)
                        r['candidate_boundary_matches']={x:tokenizer.encode(rendered+' '+x,add_special_tokens=False)==base+candidates[x] for x in 'ABCD'}
                    save(r)
        print('Diagnostic finished item:',item['id'])
    frame=pd.DataFrame(rows)
    assert len(frame)==120, len(frame)
    frame.to_csv(dest/'results.csv',index=False)
    summary=frame.groupby(['question_language','cue_language','method'],as_index=False).agg(
      trials=('correct','size'),accuracy=('correct','mean'),coverage=('extractable','mean'),
      token_limit_rate=('hit_token_limit','mean'))
    summary.to_csv(dest/'summary.csv',index=False)
    metadata=dict(model_id=MODEL_ID,model_revision=globals().get('model_revision'),
       model_dtype=str(model.dtype),quantization_config=str(getattr(model.config,'quantization_config',None)),
       budget=budget,do_sample=False,source_experiment=globals().get('EXPERIMENT_VERSION'),
       note='Diagnostic only. No precision changes. English versus Hausa prefix changes can affect elicitation; no causal attribution to language competence alone.')
    (dest/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    inventory={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file()}
    (dest/'sha256_inventory.json').write_text(json.dumps(inventory,indent=2))
    archive=shutil.make_archive(str(dest),'zip',root_dir=dest)
    display(summary)
    print('Largest score-recomputation difference:',frame.independent_score_max_difference.max())
    print('Download diagnostic ZIP:',archive)
    return frame,summary

diagnostic_results,diagnostic_summary=run_prompt_diagnostic()
