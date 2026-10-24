"""Conservative answer extraction shared by fresh inference and offline reanalysis.
Never infer an answer from subject matter or consult the gold label during parsing.
"""
import re, json, hashlib, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
PARSER_VERSION='explicit-answer-v6'
LETTERS_SET=set('ABCD')
LABEL=r'(?:Final answer|Selected answer|Answer|Amsar da na zaɓa|Amsa ta ƙarshe|Amsar ƙarshe|Amsar|Amsa)'
MARKER=re.compile(r'(?im)^[ \t]*'+LABEL+r'[ \t]*:[ \t]*([ABCD])(?=[ \t.,;:!?)\-]|$)')
PHRASE=re.compile(r'(?im)^[ \t]*(?:(?:Therefore|Thus|Hence)[, \t]+)?(?:the (?:correct |final )?answer is|the correct option is|amsar da ta dace ita ce)[ \t:]*([ABCD])(?=[ \t.,;:!?)\-]|$)')
LEAD=re.compile(r'\A[ \t]*([ABCD])(?:[.)](?=[ \t]|$)|(?=[ \t]*(?:\n|$)))')
OPTION_LINE=re.compile(r'(?m)^[ \t]*([ABCD])[.)](?:[ \t]+|$)')


def normalize_answer_text(text):
    # Normalize typography for extraction only; strict compliance uses original text.
    text=unicodedata.normalize('NFC', text).replace('\r\n','\n').replace('\r','\n').replace('\u00a0',' ')
    # Remove balanced inline Markdown wrappers, never arbitrary letters or words.
    # Restrict wrappers to one line. Repetition handles nested emphasis.
    for _ in range(3):
        previous=text
        for pattern in [r'\*\*([^*\n]+)\*\*', r'__([^_\n]+)__',
                        r'(?<!`)`([^`\n]+)`(?!`)', r'(?<!\*)\*([^*\n]+)\*(?!\*)']:
            text=re.sub(pattern, lambda m:m.group(1), text)
        if text==previous:break
    return text.strip()


def extract_answer(text, method='generation_validation'):
    raw_text=str(text or '').strip()
    text=normalize_answer_text(raw_text)
    # An option list is not a commitment. A final explicit marker after it can be.
    options=list(OPTION_LINE.finditer(text))
    listed=len({m.group(1) for m in options})>=2
    lead=LEAD.match(text)
    events=[]
    if lead and not listed: events.append((lead.start(),lead.group(1),'leading_option',lead.end()))
    events += [(m.start(),m.group(1).upper(),'answer_marker',m.end()) for m in MARKER.finditer(text)]
    events += [(m.start(),m.group(1).upper(),'final_phrase',m.end()) for m in PHRASE.finditer(text)]
    # Reject explicit alternatives/uncertainty in a choice's own line.
    accepted=[]; ambiguous=False
    for start,letter,kind,end in events:
        line=text[start:text.find('\n',start) if '\n' in text[start:] else len(text)]
        if re.search(r'\b(?i:or|ko)\s+[ABCD]\b',line) or re.search(r'\b(?:maybe|perhaps|not sure|cannot decide)\b',line,re.I):
            ambiguous=True
        else: accepted.append((start,letter,kind,end))
    accepted.sort()
    choices={e[1] for e in accepted}
    conflict=len(choices)>1
    initial=accepted[0][1] if accepted and accepted[0][0]==0 else ''
    explicit_finals=[e for e in accepted if e[2]!='leading_option']
    final=explicit_finals[-1][1] if explicit_finals else (initial if initial else '')
    usable=len(choices)==1 and not ambiguous
    answer=next(iter(choices)) if usable else ''
    if method=='generation_validation':
        strict=bool(re.fullmatch(r'[ABCD]',raw_text))
    elif method=='reason_first':
        matches=list(re.finditer(r'(?m)^Answer: ([ABCD])$',raw_text))
        strict=bool(matches and matches[-1].end()==len(raw_text) and raw_text[:matches[-1].start()].strip() and usable)
    else:
        strict=None  # Forced/injected choices have no free output-format compliance test.
    # Positional evidence, not a judgment of the explanation's semantic quality.
    if method=='reason_first':
        if not accepted: order='unresolved'
        elif not text[:accepted[0][0]].strip():
            order='answer_before_reason' if text[accepted[0][3]:].strip(' .):\n\t') else 'answer_only'
        else: order='text_before_answer'  # Text preceding a choice is not proof of valid reasoning.
    else: order='not_applicable'
    return {'parser_version':PARSER_VERSION,'strict_format_compliant':strict,
       'normalization_applied':text!=raw_text,'initial_answer':initial,'explicit_final_answer':final,'answer_conflict':conflict,
       'ambiguous_choice':ambiguous,'option_list_detected':listed,'answer_extractable':usable,
       'extracted_answer':answer,'parse_method':'+'.join(sorted({e[2] for e in accepted})) if accepted else 'none',
       'answer_order':order,'reason_first_order_eligible':method=='reason_first' and usable and order=='text_before_answer'}

