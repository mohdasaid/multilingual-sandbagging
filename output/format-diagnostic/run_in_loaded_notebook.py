# Paste this entire cell into the bottom of the already-loaded v8 notebook.
# Run ONLY this cell. No installation, model reload, or research-output changes.
import json, re, hashlib, shutil, random, math
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

FORMAT_DIAGNOSTIC_TOKENS = 64
_FORMAT_PARSER_SOURCE = '"""Conservative answer extraction shared by fresh inference and offline reanalysis.\nNever infer an answer from subject matter or consult the gold label during parsing.\n"""\nimport re, json, hashlib, unicodedata\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nPARSER_VERSION=\'explicit-answer-v6\'\nLETTERS_SET=set(\'ABCD\')\nLABEL=r\'(?:FINAL|Final answer|Selected answer|Answer|Amsar da na zaɓa|Amsa ta ƙarshe|Amsar ƙarshe|Amsar|Amsa)\'\nMARKER=re.compile(r\'(?im)^[ \\t]*\'+LABEL+r\'[ \\t]*:[ \\t]*([ABCD])(?=[ \\t.,;:!?)\\-]|$)\')\nPHRASE=re.compile(r\'(?im)^[ \\t]*(?:(?:Therefore|Thus|Hence)[, \\t]+)?(?:the (?:correct |final )?answer is|the correct option is|amsar da ta dace ita ce)[ \\t:]*([ABCD])(?=[ \\t.,;:!?)\\-]|$)\')\nLEAD=re.compile(r\'\\A[ \\t]*([ABCD])(?:[.)](?=[ \\t]|$)|(?=[ \\t]*(?:\\n|$)))\')\nOPTION_LINE=re.compile(r\'(?m)^[ \\t]*([ABCD])[.)](?:[ \\t]+|$)\')\n\n\ndef normalize_answer_text(text):\n    # Normalize typography for extraction only; strict compliance uses original text.\n    text=unicodedata.normalize(\'NFC\', text).replace(\'\\r\\n\',\'\\n\').replace(\'\\r\',\'\\n\').replace(\'\\u00a0\',\' \')\n    # Remove balanced inline Markdown wrappers, never arbitrary letters or words.\n    # Restrict wrappers to one line. Repetition handles nested emphasis.\n    for _ in range(3):\n        previous=text\n        for pattern in [r\'\\*\\*([^*\\n]+)\\*\\*\', r\'__([^_\\n]+)__\',\n                        r\'(?<!`)`([^`\\n]+)`(?!`)\', r\'(?<!\\*)\\*([^*\\n]+)\\*(?!\\*)\']:\n            text=re.sub(pattern, lambda m:m.group(1), text)\n        if text==previous:break\n    return text.strip()\n\n\ndef extract_answer(text, method=\'generation_validation\'):\n    raw_text=str(text or \'\').strip()\n    text=normalize_answer_text(raw_text)\n    # An option list is not a commitment. A final explicit marker after it can be.\n    options=list(OPTION_LINE.finditer(text))\n    listed=len({m.group(1) for m in options})>=2\n    lead=LEAD.match(text)\n    events=[]\n    if lead and not listed: events.append((lead.start(),lead.group(1),\'leading_option\',lead.end()))\n    events += [(m.start(),m.group(1).upper(),\'answer_marker\',m.end()) for m in MARKER.finditer(text)]\n    events += [(m.start(),m.group(1).upper(),\'final_phrase\',m.end()) for m in PHRASE.finditer(text)]\n    # Reject explicit alternatives/uncertainty in a choice\'s own line.\n    accepted=[]; ambiguous=False\n    for start,letter,kind,end in events:\n        line=text[start:text.find(\'\\n\',start) if \'\\n\' in text[start:] else len(text)]\n        if re.search(r\'\\b(?i:or|ko)\\s+[ABCD]\\b\',line) or re.search(r\'\\b(?:maybe|perhaps|not sure|cannot decide)\\b\',line,re.I):\n            ambiguous=True\n        else: accepted.append((start,letter,kind,end))\n    accepted.sort()\n    choices={e[1] for e in accepted}\n    conflict=len(choices)>1\n    initial=accepted[0][1] if accepted and accepted[0][0]==0 else \'\'\n    explicit_finals=[e for e in accepted if e[2]!=\'leading_option\']\n    final=explicit_finals[-1][1] if explicit_finals else (initial if initial else \'\')\n    usable=len(choices)==1 and not ambiguous\n    answer=next(iter(choices)) if usable else \'\'\n    if method==\'generation_validation\':\n        strict=bool(re.fullmatch(r\'[ABCD]\',raw_text))\n    elif method==\'reason_first\':\n        matches=list(re.finditer(r\'(?m)^Answer: ([ABCD])$\',raw_text))\n        strict=bool(matches and matches[-1].end()==len(raw_text) and raw_text[:matches[-1].start()].strip() and usable)\n    else:\n        strict=None  # Forced/injected choices have no free output-format compliance test.\n    # Positional evidence, not a judgment of the explanation\'s semantic quality.\n    if method==\'reason_first\':\n        if not accepted: order=\'unresolved\'\n        elif not text[:accepted[0][0]].strip():\n            order=\'answer_before_reason\' if text[accepted[0][3]:].strip(\' .):\\n\\t\') else \'answer_only\'\n        else: order=\'text_before_answer\'  # Text preceding a choice is not proof of valid reasoning.\n    else: order=\'not_applicable\'\n    return {\'parser_version\':PARSER_VERSION,\'strict_format_compliant\':strict,\n       \'normalization_applied\':text!=raw_text,\'initial_answer\':initial,\'explicit_final_answer\':final,\'answer_conflict\':conflict,\n       \'ambiguous_choice\':ambiguous,\'option_list_detected\':listed,\'answer_extractable\':usable,\n       \'extracted_answer\':answer,\'parse_method\':\'+\'.join(sorted({e[2] for e in accepted})) if accepted else \'none\',\n       \'answer_order\':order,\'reason_first_order_eligible\':method==\'reason_first\' and usable and order==\'text_before_answer\'}\n\n'
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
        strict=bool(re.search(r'(?:^|\n)FINAL: [ABCD]$',raw) and p['answer_extractable'])
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
            options='\n'.join(f'{letter}. {value}' for letter,value in zip('ABCD',displayed))
            common=dict(item_id=item['id'],kind=item['kind'],rotation=rotation,
                gold=gold,gold_value=str(item['options'][item['gold_index']]),displayed_options=displayed)
            for qlang,clang in [('en','en'),('ha','en'),('ha','ha')]:
                for procedure in ['current_prefix','english_prefix','final_marker']:
                    instruction=FINAL_INSTRUCTIONS[clang] if procedure=='final_marker' else CONTROL_FORMAT[clang]
                    prefix='' if procedure=='final_marker' else ANSWER_PREFIXES['en' if procedure=='english_prefix' else clang]
                    prompt=instruction+'\n\n'+item[qlang]+'\n'+options
                    jobs.append(common | dict(question_language=qlang,cue_language=clang,procedure=procedure,
                        prefix=prefix,prompt=prompt,numeric=False,candidate_key=None if procedure=='final_marker' else ('en' if procedure=='english_prefix' else clang)))
            if MODEL_ID=='google/gemma-3-4b-it' and item['kind']=='basic_capability':
                # English-only follow-up: isolate options versus letter mapping.
                jobs.append(common | dict(question_language='en',cue_language='en',procedure='options_numeric',
                    prefix='',candidate_key=None,numeric=True,
                    prompt='Return only the numeric answer, without explanation.\n\n'+item['en']+'\n'+options))
                jobs.append(common | dict(question_language='en',cue_language='en',procedure='given_value_to_letter',
                    prefix='',candidate_key=None,numeric=False,
                    prompt=f"The correct numeric value is {common['gold_value']}. Which option contains that value? Return only its letter: A, B, C, or D.\n\n"+options))
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
            with (dest/'responses.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
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
