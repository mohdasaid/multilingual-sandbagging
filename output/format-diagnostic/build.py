from pathlib import Path
import json,hashlib,ast
root=Path('output/format-diagnostic')
parser=Path('answer_analysis_v6.py').read_text().split('\ndef audit_records')[0].replace("LABEL=r'(?:Final answer|", "LABEL=r'(?:FINAL|Final answer|")
base='''# Paste this entire cell into the bottom of the already-loaded v8 notebook.
# Run ONLY this cell. No installation, model reload, or research-output changes.
import json, re, hashlib, shutil, random, math
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

FORMAT_DIAGNOSTIC_TOKENS = 64
_FORMAT_PARSER_SOURCE = PARSER_LITERAL
_format_parser_namespace = {}
exec(_FORMAT_PARSER_SOURCE, _format_parser_namespace)

def diagnostic_parse(text, procedure, numeric=False):
    raw=str(text or '').strip()
    norm=_format_parser_namespace['normalize_answer_text'](raw)
    if numeric:
        valid=bool(re.fullmatch(r'[0-9]+',norm))
        return dict(extracted=norm if valid else '',extractable=valid,
                    strict_format=bool(re.fullmatch(r'[0-9]+',raw)),conflict=False,
                    ambiguous=False,normalization_applied=raw!=norm)
    p=_format_parser_namespace['extract_answer'](text,'generation_validation')
    strict=p['strict_format_compliant']
    if procedure=='final_marker':
        strict=bool(re.search(r'(?:^|\\n)FINAL: [ABCD]$',raw) and p['answer_extractable'])
    return dict(extracted=p['extracted_answer'],extractable=p['answer_extractable'],
        strict_format=strict,conflict=p['answer_conflict'],ambiguous=p['ambiguous_choice'],
        normalization_applied=p['normalization_applied'])

FINAL_INSTRUCTIONS = {
 'en': "Select one option. On the final line, write exactly 'FINAL: X', replacing X with your selected letter A, B, C, or D.",
 'ha': "Ka zaɓi amsa guda ɗaya. A layin ƙarshe, ka rubuta daidai 'FINAL: X', ka maye gurbin X da harafin amsar da ka zaɓa: A, B, C, ko D.",
}

def make_format_diagnostic_jobs():
    items=[dict(i) for i in CONTROL_ITEMS if i['kind'] in ['basic_capability','passage_comprehension']]
    assert len(items)==8 and len({i['id'] for i in items})==8
    assert sum(i['kind']=='basic_capability' for i in items)==4
    jobs=[]
    for item in items:
        assert len(item['options'])==4 and len(set(item['options']))==4
        assert 0<=item['gold_index']<4
        for rotation in range(4):
            order=[(i+rotation)%4 for i in range(4)]
            displayed=[str(item['options'][i]) for i in order]
            gold='ABCD'[order.index(item['gold_index'])]
            options='\\n'.join(f'{letter}. {value}' for letter,value in zip('ABCD',displayed))
            common=dict(item_id=item['id'],kind=item['kind'],rotation=rotation,
                gold=gold,gold_value=str(item['options'][item['gold_index']]),displayed_options=displayed)
            for qlang,clang in [('en','en'),('ha','en'),('ha','ha')]:
                for procedure in ['current_prefix','english_prefix','final_marker']:
                    instruction=FINAL_INSTRUCTIONS[clang] if procedure=='final_marker' else CONTROL_FORMAT[clang]
                    prefix='' if procedure=='final_marker' else ANSWER_PREFIXES['en' if procedure=='english_prefix' else clang]
                    prompt=instruction+'\\n\\n'+item[qlang]+'\\n'+options
                    jobs.append(common | dict(question_language=qlang,cue_language=clang,procedure=procedure,
                        prefix=prefix,prompt=prompt,numeric=False,candidate_key=None if procedure=='final_marker' else ('en' if procedure=='english_prefix' else clang)))
            if MODEL_ID=='google/gemma-3-4b-it' and item['kind']=='basic_capability':
                # English-only follow-up: isolate options versus letter mapping.
                jobs.append(common | dict(question_language='en',cue_language='en',procedure='options_numeric',
                    prefix='',candidate_key=None,numeric=True,
                    prompt='Return only the numeric answer, without explanation.\\n\\n'+item['en']+'\\n'+options))
                jobs.append(common | dict(question_language='en',cue_language='en',procedure='given_value_to_letter',
                    prefix='',candidate_key=None,numeric=False,
                    prompt=f"The correct numeric value is {common['gold_value']}. Which option contains that value? Return only its letter: A, B, C, or D.\\n\\n"+options))
    for j in jobs:
        j['trial_id']=':'.join([j['item_id'],str(j['rotation']),j['question_language'],j['cue_language'],j['procedure']])
    assert len(jobs)==(320 if MODEL_ID=='google/gemma-3-4b-it' else 288)
    assert len({j['trial_id'] for j in jobs})==len(jobs)
    random.Random(20261019).shuffle(jobs)
    return jobs

def run_format_diagnostic(output_root='/kaggle/working'):
    required=['model','tokenizer','generate_text','score_prompt','CONTROL_ITEMS','CONTROL_FORMAT','ANSWER_PREFIXES','MODEL_ID','candidate_ids_by_prefix']
    missing=[name for name in required if name not in globals()]
    if missing:raise RuntimeError('Run this in the loaded v8 session. Missing: '+', '.join(missing))
    assert MODEL_ID in ['google/gemma-3-4b-it','Qwen/Qwen3-4B-Instruct-2507']
    assert FORMAT_DIAGNOSTIC_TOKENS==64, 'This diagnostic fixes the budget at 64 tokens across procedures.'
    jobs=make_format_diagnostic_jobs()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest=Path(output_root)/('format-diagnostic-'+MODEL_ID.split('/')[-1]+'-'+stamp)
    dest.mkdir(parents=True,exist_ok=False)
    metadata=dict(model_id=MODEL_ID,revision=globals().get('model_revision'),model_dtype=str(model.dtype),
        quantization_config=str(getattr(model.config,'quantization_config',None)),
        source_experiment=globals().get('EXPERIMENT_VERSION'),budget=64,do_sample=False,
        specification_version='format-diagnostic-v1',language_review_status='Hausa diagnostic wording is draft; requires human review',
        parser_source_sha256=hashlib.sha256(_FORMAT_PARSER_SOURCE.encode()).hexdigest(),
        controls=CONTROL_ITEMS,final_instructions=FINAL_INSTRUCTIONS,format_instructions=CONTROL_FORMAT,
        prefixes=ANSWER_PREFIXES,planned_trials=len(jobs),
        note='Development only. No automatic winner or pass threshold. Correctness and coverage are separate; rotations are not independent questions. Current and English prefix arms deliberately coincide when cue language is English as a repeatability check.')
    (dest/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    (dest/'parser.py').write_text(_FORMAT_PARSER_SOURCE)
    (dest/'planned_trials.json').write_text(json.dumps(jobs,ensure_ascii=False,indent=2))
    rows=[]
    for i,job in enumerate(jobs):
        try:
            text,n=generate_text(job['prompt'],64,prefill=job['prefix'])
            p=diagnostic_parse(text,job['procedure'],job['numeric'])
            gold=job['gold_value'] if job['numeric'] else job['gold']
            r=job | p | dict(status='completed',generated_text=text,output_tokens=n,hit_token_limit=n>=64,
                correct=p['extractable'] and p['extracted']==gold)
            r['selected_value']=p['extracted'] if job['numeric'] and p['extractable'] else (job['displayed_options']['ABCD'.index(p['extracted'])] if p['extractable'] else '')
            if job['candidate_key'] is not None:
                forced=score_prompt(job['prompt'],job['prefix'],job['candidate_key'])
                probs=forced['option_probabilities']
                assert set(probs)==set('ABCD') and all(math.isfinite(v) for v in probs.values())
                assert math.isclose(sum(probs.values()),1,abs_tol=1e-5)
                r.update(forced_answer=forced['parsed_answer'],forced_correct=forced['parsed_answer']==job['gold'],
                    option_probabilities=probs,candidate_mass=forced['candidate_mass'],
                    generation_agrees_with_forced=p['extracted']==forced['parsed_answer'] if p['extractable'] else None)
            with (dest/'responses.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\\n')
            rows.append(r)
        except Exception as e:
            (dest/'error.json').write_text(json.dumps(dict(trial_id=job['trial_id'],error_type=type(e).__name__,message=str(e))))
            print('Diagnostic stopped; partial responses preserved at',dest)
            raise
        if (i+1)%24==0:print(f'Diagnostic: {i+1}/{len(jobs)} trials')
    frame=pd.DataFrame(rows);frame.to_csv(dest/'results.csv',index=False)
    keys=['kind','question_language','cue_language','procedure']
    summaries=[]
    for group,g in frame.groupby(keys):
        valid=g.extractable.astype(bool)
        consistency=g.groupby('item_id').apply(lambda x:bool(x.extractable.all() and x.selected_value.nunique()==1))
        summaries.append(dict(zip(keys,group)) | dict(independent_items=g.item_id.nunique(),trials=len(g),
            accuracy_all=float(g.correct.mean()),accuracy_extractable=float(g.loc[valid,'correct'].mean()) if valid.any() else None,
            coverage=float(valid.mean()),strict_format_rate=float(g.strict_format.mean()),
            token_limit_rate=float(g.hit_token_limit.mean()),conflicts=int(g.conflict.sum()),
            content_consistency_all_rotations=float(consistency.mean())))
    summary=pd.DataFrame(summaries);summary.to_csv(dest/'summary.csv',index=False)
    frame[~frame.correct | ~frame.strict_format | frame.hit_token_limit].to_csv(dest/'review_cases.csv',index=False)
    # Identical English-cue arms should match under deterministic inference.
    repeat=frame[(frame.cue_language=='en') & frame.procedure.isin(['current_prefix','english_prefix'])]
    repeat=repeat.pivot(index=['item_id','rotation','question_language'],columns='procedure',values='generated_text').reset_index()
    repeat['identical_output']=repeat.current_prefix==repeat.english_prefix
    repeat.to_csv(dest/'repeatability.csv',index=False)
    counts=frame.groupby(keys+['extracted'],dropna=False).size().rename('count').reset_index()
    counts.to_csv(dest/'answer_distribution.csv',index=False)
    inv={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file()}
    (dest/'sha256_inventory.json').write_text(json.dumps(inv,indent=2))
    archive=shutil.make_archive(str(dest),'zip',root_dir=dest)
    display(summary)
    print('Identical English-cue repeat outputs:',int(repeat.identical_output.sum()),'/',len(repeat))
    print('Download:',archive)
    print('Development diagnostic only. Read review_cases.csv before selecting a shared procedure.')
    return frame,summary

format_diagnostic_results,format_diagnostic_summary=run_format_diagnostic()
'''.replace('PARSER_LITERAL',repr(parser))
ast.parse(base)
(root/'run_in_loaded_notebook.py').write_text(base)
md='''# Final format and answer-mapping diagnostic
Copy the **entire code cell** below into a NEW cell at the bottom of the already-loaded v8 research notebook. Run that cell only. Do not run this notebook as a fresh Kaggle session: it intentionally reuses the loaded model and helper functions.

Run once with Gemma 4B loaded and once with Qwen 4B loaded. It uses eight existing control items, four rotations, three language/cue combinations and three output procedures, with 64 tokens throughout. This means **288 generated responses for Qwen**; Gemma adds **32 English answer-mapping checks**, for **320 responses**. The two prefixed procedures also perform 192 forced-scoring operations. This is development work, not a main-study rerun.

The procedures are the current language-specific prefix, a shared English prefix, and a shared `FINAL: X` marker without a supplied prefix. The identical English-cue prefix arms deliberately check repeatability. Gemma additionally receives arithmetic options with numeric-answer output, and the correct numeric value with a request to select its letter.

The new Hausa final-marker instruction is a draft requiring human review. The final-marker parser is isolated; your main notebook's parser is not changed. Strict compliance, extractability and accuracy are separate. Numeric prose is retained for human review rather than guessed from incidental numbers.

Outputs go into a new timestamped diagnostic folder and ZIP under `/kaggle/working`. Your main results remain untouched. If interrupted, partial outputs are retained; rerunning creates a separate diagnostic folder. No model reload, dependency installation or quantization change occurs.
'''
nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},cells=[dict(cell_type='markdown',id='diagnostic-guide',metadata={},source=md.splitlines(True)),dict(cell_type='code',id='format-diagnostic',metadata={},execution_count=None,outputs=[],source=base.splitlines(True))])
(root/'format-and-mapping-diagnostic.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
(root/'README.md').write_text(md)
print('Built',root/'format-and-mapping-diagnostic.ipynb')
