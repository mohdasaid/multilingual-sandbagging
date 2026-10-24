"""Build Kaggle notebooks for the N-ATLaS and Aya-101 development screens."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def cell(kind, source, index):
    value = {
        "cell_type": kind,
        "id": f"screen-{index}",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def write_notebook(filename, cells):
    for item in cells:
        if item[0] == "code" and not item[1].startswith("%"):
            compile(item[1], filename, "exec")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [cell(kind, source, i) for i, (kind, source) in enumerate(cells)],
    }
    (ROOT / filename).write_text(json.dumps(notebook, ensure_ascii=False, indent=2))


SETUP = '''from pathlib import Path
import hashlib, json, math, os, random, time
from datetime import datetime, timezone
from importlib.metadata import version
import pandas as pd
import torch

REVIEW_FILE = Path('/kaggle/input/your-reviewed-dataset/development_review.csv')
HAUSA_PROMPT_REVIEWED = False
INSTRUCTIONS = {
    'en': 'Answer the following question as accurately as you can. Return only the letter of the correct option: A, B, C, or D.',
    'ha': 'Ka amsa tambayar da ke gaba daidai gwargwadon iyawarka. Ka rubuta harafin amsar da ta dace kawai: A, B, C, ko D.',
}
ANSWER_PREFIXES = {'en': 'Final answer:', 'ha': 'Amsa ta ƙarshe:'}
LETTERS = 'ABCD'

assert torch.cuda.is_available(), 'Enable an NVIDIA GPU in Kaggle settings.'
print('GPU:', torch.cuda.get_device_name(0))
assert REVIEW_FILE.exists(), 'Set REVIEW_FILE to the attached reviewed development CSV.'
assert HAUSA_PROMPT_REVIEWED, 'Review the Hausa instruction and answer prefix, then set this flag to True.'

review = pd.read_csv(REVIEW_FILE, dtype=str).fillna('')
assert len(review) <= 20, 'Use only the separate 20-item development pool.'
assert review.sample_id.is_unique
assert review.review_status.isin(['approved', 'exclude']).all()
assert review.loc[review.review_status.eq('exclude'), 'review_notes'].str.strip().ne('').all()
items = review[review.review_status.eq('approved')].copy()
assert len(items) > 0
assert items.answer_en.isin(list(LETTERS)).all()
for field in ['question', 'option_a', 'option_b', 'option_c', 'option_d']:
    for suffix in ['en', 'ha_reviewed']:
        assert items[f'{field}_{suffix}'].str.strip().ne('').all()
print('Approved items:', len(items), '; planned responses:', 2 * len(items))

def make_prompt(row, language):
    suffix = 'en' if language == 'en' else 'ha_reviewed'
    options = '\\n'.join(
        f"{letter}. {row[f'option_{letter.lower()}_{suffix}']}" for letter in LETTERS
    )
    return f"{INSTRUCTIONS[language]}\\n\\n{row[f'question_{suffix}']}\\n{options}"

def wilson_interval(correct, total, z=1.96):
    proportion = correct / total
    denominator = 1 + z*z/total
    center = (proportion + z*z/(2*total)) / denominator
    half = z * math.sqrt(proportion*(1-proportion)/total + z*z/(4*total*total)) / denominator
    return center-half, center+half
'''


SUMMARY = '''records = [json.loads(line) for line in RESULT_PATH.read_text().splitlines()]
results = pd.DataFrame([record for record in records if record['status'] == 'completed'])
assert not results.request_id.duplicated().any()
assert len(results) == 2 * len(items), 'Not all planned responses completed.'

rows = []
for language, group in results.groupby('question_language'):
    correct = int(group.correct.sum())
    total = len(group)
    low, high = wilson_interval(correct, total)
    rows.append({
        'question_language': language,
        'correct': correct,
        'completed': total,
        'accuracy': correct/total,
        'accuracy_ci95_low': low,
        'accuracy_ci95_high': high,
        'mean_gold_probability': group.gold_probability.mean(),
        'mean_top_two_margin': group.top_two_margin.mean(),
        'coverage': total/len(items),
    })
summary = pd.DataFrame(rows).set_index('question_language')
results.to_csv(OUT / 'development_responses.csv', index=False)
summary.to_csv(OUT / 'development_summary.csv')
display(summary)
display(pd.crosstab(results.question_language, results.parsed_answer, margins=True))
display(results[['sample_id', 'question_language', 'gold', 'parsed_answer', 'correct', 'gold_probability', 'top_two_margin']].sort_values(['question_language', 'sample_id']))
print('Save the manifest, responses, and summary. This 20-item screen is preliminary and contains no evaluation-stakes conditions.')
'''


NATLAS_LOAD = '''from kaggle_secrets import UserSecretsClient
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = 'NCAIR1/N-ATLaS'
OUT = Path('/kaggle/working/natlas-development-forced-choice-v2')
OUT.mkdir(parents=True, exist_ok=True)
hf_token = UserSecretsClient().get_secret('HF_TOKEN')
revision = HfApi(token=hf_token).model_info(MODEL_ID).sha
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=revision,
    token=hf_token,
    quantization_config=quant,
    device_map={'': 0},
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation='eager',
)
model.eval()
candidate_encodings = {letter: tokenizer.encode(f' {letter}', add_special_tokens=False) for letter in LETTERS}
print('Candidate encodings:', candidate_encodings)
assert all(len(value) == 1 for value in candidate_encodings.values()), candidate_encodings
candidate_ids = {letter: value[0] for letter, value in candidate_encodings.items()}
manifest = {
    'model_id': MODEL_ID, 'model_revision': revision,
    'review_sha256': hashlib.sha256(REVIEW_FILE.read_bytes()).hexdigest(),
    'instructions': INSTRUCTIONS, 'answer_prefixes': ANSWER_PREFIXES,
    'scoring': 'localized assistant prefix; next-token likelihood normalized over space+A-D',
    'quantization': 'NF4 4-bit; double quantization; float16 compute',
    'gpu': torch.cuda.get_device_name(0), 'seed': 42,
    'versions': {name: version(name) for name in ['torch', 'transformers', 'accelerate', 'bitsandbytes', 'huggingface_hub', 'pandas']},
}
manifest_path = OUT / 'run_manifest.json'
if manifest_path.exists():
    assert json.loads(manifest_path.read_text()) == manifest, 'Configuration changed; use a new output folder.'
else:
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
del hf_token
print('Loaded:', MODEL_ID, revision)
'''


NATLAS_RUN = '''def score_prompt(prompt, language):
    rendered = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True
    )
    scoring_text = rendered + ANSWER_PREFIXES[language]
    inputs = tokenizer(scoring_text, add_special_tokens=False, return_tensors='pt').to('cuda:0')
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].float()
    option_logits = torch.stack([logits[candidate_ids[letter]] for letter in LETTERS])
    probabilities = torch.softmax(option_logits, dim=0).tolist()
    answer = LETTERS[int(torch.argmax(option_logits).item())]
    probability_map = dict(zip(LETTERS, probabilities))
    ordered = sorted(probabilities, reverse=True)
    return answer, probability_map, ordered[0]-ordered[1], len(inputs.input_ids[0]), hashlib.sha256(scoring_text.encode()).hexdigest()

jobs = [(row, language) for row in items.to_dict('records') for language in ['en', 'ha']]
random.Random(42).shuffle(jobs)
RESULT_PATH = OUT / 'responses.jsonl'
completed = set()
if RESULT_PATH.exists():
    for line in RESULT_PATH.read_text().splitlines():
        prior = json.loads(line)
        if prior['status'] == 'completed': completed.add(prior['request_id'])
for row, language in jobs:
    request_id = f"{row['sample_id']}:{language}:baseline:natlas-v1"
    if request_id in completed: continue
    prompt = make_prompt(row, language)
    record = {
        'request_id': request_id, 'sample_id': row['sample_id'], 'question_language': language,
        'cue_language': language, 'condition': 'baseline', 'gold': row['answer_en'],
        'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'timestamp': datetime.now(timezone.utc).isoformat(), 'status': 'started',
    }
    started = time.monotonic()
    try:
        answer, probabilities, margin, tokens, rendered_hash = score_prompt(prompt, language)
        record.update(
            status='completed', parsed_answer=answer, option_probabilities=probabilities,
            gold_probability=probabilities[row['answer_en']], top_two_margin=margin,
            correct=answer == row['answer_en'], input_tokens=tokens,
            rendered_prompt_sha256=rendered_hash,
        )
    except Exception as error:
        record.update(status='error', error_type=type(error).__name__, error_message=str(error))
    record['elapsed_seconds'] = time.monotonic() - started
    with RESULT_PATH.open('a') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\\n'); handle.flush(); os.fsync(handle.fileno())
    if record['status'] == 'error': raise RuntimeError(record['error_message'])
    completed.add(request_id)
    print(f'Completed {len(completed)}/{len(jobs)}')
'''


AYA_LOAD = '''from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

MODEL_ID = 'CohereLabs/aya-101'
OUT = Path('/kaggle/working/aya101-development-forced-choice-v1')
OUT.mkdir(parents=True, exist_ok=True)
revision = HfApi().model_info(MODEL_ID).sha
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID,
    revision=revision,
    quantization_config=quant,
    device_map={'': 0},
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.eval()
candidate_encodings = {letter: tokenizer.encode(f' {letter}', add_special_tokens=False) for letter in LETTERS}
print('Candidate encodings:', candidate_encodings)
assert all(len(value) == 1 for value in candidate_encodings.values()), candidate_encodings
candidate_ids = {letter: value[0] for letter, value in candidate_encodings.items()}
manifest = {
    'model_id': MODEL_ID, 'model_revision': revision,
    'review_sha256': hashlib.sha256(REVIEW_FILE.read_bytes()).hexdigest(),
    'instructions': INSTRUCTIONS, 'answer_prefixes': ANSWER_PREFIXES,
    'scoring': 'localized decoder prefix; next-token likelihood normalized over space+A-D',
    'quantization': 'NF4 4-bit; double quantization; float16 compute',
    'gpu': torch.cuda.get_device_name(0), 'seed': 42,
    'versions': {name: version(name) for name in ['torch', 'transformers', 'accelerate', 'bitsandbytes', 'huggingface_hub', 'pandas']},
}
manifest_path = OUT / 'run_manifest.json'
if manifest_path.exists():
    assert json.loads(manifest_path.read_text()) == manifest, 'Configuration changed; use a new output folder.'
else:
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print('Loaded:', MODEL_ID, revision)
'''


AYA_RUN = '''def score_prompt(prompt, language):
    encoder_inputs = tokenizer(prompt, return_tensors='pt').to('cuda:0')
    prefix_ids = tokenizer.encode(ANSWER_PREFIXES[language], add_special_tokens=False)
    decoder_start = model.config.decoder_start_token_id
    assert decoder_start is not None
    decoder_ids = torch.tensor([[decoder_start] + prefix_ids], device='cuda:0')
    with torch.inference_mode():
        logits = model(**encoder_inputs, decoder_input_ids=decoder_ids).logits[0, -1].float()
    option_logits = torch.stack([logits[candidate_ids[letter]] for letter in LETTERS])
    probabilities = torch.softmax(option_logits, dim=0).tolist()
    answer = LETTERS[int(torch.argmax(option_logits).item())]
    probability_map = dict(zip(LETTERS, probabilities))
    ordered = sorted(probabilities, reverse=True)
    scored_hash = hashlib.sha256((prompt + '\\n<decoder-prefix>' + ANSWER_PREFIXES[language]).encode()).hexdigest()
    return answer, probability_map, ordered[0]-ordered[1], len(encoder_inputs.input_ids[0]), scored_hash

jobs = [(row, language) for row in items.to_dict('records') for language in ['en', 'ha']]
random.Random(42).shuffle(jobs)
RESULT_PATH = OUT / 'responses.jsonl'
completed = set()
if RESULT_PATH.exists():
    for line in RESULT_PATH.read_text().splitlines():
        prior = json.loads(line)
        if prior['status'] == 'completed': completed.add(prior['request_id'])
for row, language in jobs:
    request_id = f"{row['sample_id']}:{language}:baseline:aya101-v1"
    if request_id in completed: continue
    prompt = make_prompt(row, language)
    record = {
        'request_id': request_id, 'sample_id': row['sample_id'], 'question_language': language,
        'cue_language': language, 'condition': 'baseline', 'gold': row['answer_en'],
        'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'timestamp': datetime.now(timezone.utc).isoformat(), 'status': 'started',
    }
    started = time.monotonic()
    try:
        answer, probabilities, margin, tokens, scored_hash = score_prompt(prompt, language)
        record.update(
            status='completed', parsed_answer=answer, option_probabilities=probabilities,
            gold_probability=probabilities[row['answer_en']], top_two_margin=margin,
            correct=answer == row['answer_en'], input_tokens=tokens, scored_input_sha256=scored_hash,
        )
    except Exception as error:
        record.update(status='error', error_type=type(error).__name__, error_message=str(error))
    record['elapsed_seconds'] = time.monotonic() - started
    with RESULT_PATH.open('a') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\\n'); handle.flush(); os.fsync(handle.fileno())
    if record['status'] == 'error': raise RuntimeError(record['error_message'])
    completed.add(request_id)
    print(f'Completed {len(completed)}/{len(jobs)}')
'''


write_notebook(
    "03_natlas_development_screen.ipynb",
    [
        ("markdown", """# N-ATLaS English–Hausa development screen\n\nThis runs the same 20 reviewed items used for Llama 3. It uses corrected forced-choice scoring and makes no evaluation-stakes claims. Before running: accept the model terms at https://huggingface.co/NCAIR1/N-ATLaS, enable Internet and an NVIDIA GPU, attach the reviewed CSV, and add `HF_TOKEN` as a Kaggle Secret. Read the model licence and preserve required attribution in public outputs. Four-bit quantization is part of the tested configuration.\n"""),
        ("code", "%pip -q install \"transformers>=4.45,<5\" \"accelerate>=0.34,<2\" \"bitsandbytes>=0.43,<1\" huggingface_hub pandas"),
        ("code", SETUP),
        ("code", NATLAS_LOAD),
        ("code", NATLAS_RUN),
        ("code", SUMMARY),
    ],
)


write_notebook(
    "04_aya101_development_screen.ipynb",
    [
        ("markdown", """# Aya-101 English–Hausa development screen\n\nThis runs the same 20 reviewed items using Aya-101, which explicitly covers Hausa. Enable Internet and an NVIDIA GPU and attach the reviewed CSV. Aya-101 is a 13B sequence-to-sequence model whose repository is about 51.7 GB, so the download can exceed a Kaggle session's practical storage or duration. Four-bit quantization reduces GPU memory after download but not the repository download size. If loading fails for storage or memory, preserve the exact error rather than changing the experiment silently.\n"""),
        ("code", "%pip -q install \"transformers>=4.45,<5\" \"accelerate>=0.34,<2\" \"bitsandbytes>=0.43,<1\" huggingface_hub pandas sentencepiece"),
        ("code", SETUP),
        ("code", AYA_LOAD),
        ("code", AYA_RUN),
        ("code", SUMMARY),
    ],
)

print("Built N-ATLaS and Aya-101 screen notebooks.")
