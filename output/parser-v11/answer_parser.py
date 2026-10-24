"""Conservative answer extraction shared by fresh inference and offline reanalysis.
Never infer an answer from subject matter or consult the gold label during parsing.
"""
import re, json, hashlib, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
PARSER_VERSION='explicit-answer-v11-reviewed-markers'
LETTERS_SET=set('ABCD')
LABEL=r'(?:Final answer|Selected answer|Answer|Amsar da na zaɓa|Amsa ta ƙarshe|Amsar ƙarshe|Amsar|Amsa)'
# Permit one newline between an explicit answer cue and its letter.
ANSWER_GAP=r'[ \t]*(?:\n[ \t]*)?'
MARKER=re.compile(r'(?im)^[ \t]*'+LABEL+r'[ \t]*:'+ANSWER_GAP+r'([ABCD])(?=[ \t.,;:!?)\-]|$)')
PHRASE=re.compile(r'(?im)^[ \t]*(?:(?:Therefore|Thus|Hence|So)[, \t]+)?(?:the (?:correct |final )?answer (?:is|should be)|the correct option is|amsar da ta dace ita ce)[ \t]*:?'+ANSWER_GAP+r'([ABCD])(?=[ \t.,;:!?)\-]|$)')
LEAD=re.compile(r'\A[ \t]*([ABCD])(?:[.)](?=[ \t\n]|$)|(?=[ \t]*(?:\n|$)))')
OPTION_LINE=re.compile(r'(?m)^[ \t]*([ABCD])[.)](?:[ \t]+|$)')

# Observed code-switched Swahili answer marker; apply identically to every model.
# Only line-start assertions or explicit 'kwa hivyo' clauses, not arbitrary 'ni A'.
SWAHILI_ANSWER=re.compile(r'(?im)(?:^[ \t]*(?:kwa hivyo[, \t]+)?|(?<=[,;])[ \t]*kwa hivyo[, \t]+)jibu[ \t]+ni[ \t]*:?'+ANSWER_GAP+r'([ABCD])(?=[ \t.,;:!?)\-]|$)')


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
    events += [(m.start(),m.group(1).upper(),'swahili_answer_marker',m.end()) for m in SWAHILI_ANSWER.finditer(text)]
    # Reject explicit alternatives/uncertainty in a choice's own line.
    accepted=[]; ambiguous=False
    for start,letter,kind,end in events:
        line_end=text.find('\n',end)
        line=text[start:line_end if line_end>=0 else len(text)]
        tail=text[end:line_end if line_end>=0 else len(text)]
        if re.match(r'[ \t]*(?:[,/]|(?:or|ko|au)\b)[ \t]*[ABCD]\b',tail,re.I) or re.search(r'\b(?i:or|ko|au)\s+[ABCD]\b',line) or re.search(r'\b(?:maybe|perhaps|not sure|cannot decide)\b',line,re.I):
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
    strict=bool(re.fullmatch(r'[ABCD]',raw_text))
    return {'parser_version':PARSER_VERSION,'strict_format_compliant':strict,
       'normalization_applied':text!=raw_text,'initial_answer':initial,'explicit_final_answer':final,'answer_conflict':conflict,
       'ambiguous_choice':ambiguous,'option_list_detected':listed,'answer_extractable':usable,
       'extracted_answer':answer,'parse_method':'+'.join(sorted({e[2] for e in accepted})) if accepted else 'none'}



def generation_outcome(parsed, gold):
    if parsed['answer_conflict']: return 'conflicting'
    if not parsed['answer_extractable']: return 'unresolved'
    return 'correct' if parsed['extracted_answer']==gold else 'incorrect'

OUTCOME_CATEGORIES = ['correct','incorrect','unresolved','conflicting']

def outcome_counts(frame, keys):
    counts=frame.groupby(keys+['generation_outcome']).size().unstack(fill_value=0)
    counts=counts.reindex(columns=OUTCOME_CATEGORIES,fill_value=0)
    counts['responses']=counts.sum(axis=1)
    return counts.reset_index()




ANALYSIS_CODE_SHA256 = '1641bb258d1cd882ade34b5ba848060b000dcd2ee1aec6b5de13ed55e14be7e6'
