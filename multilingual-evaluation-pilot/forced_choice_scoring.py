"""Forced-choice A-D scoring helpers for the Kaggle development notebook."""

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone

import torch


LETTERS = "ABCD"


def candidate_token_ids(tokenizer):
    """Return the single-token IDs used to score the literal answers A-D."""
    encoded = {
        letter: tokenizer.encode(letter, add_special_tokens=False)
        for letter in LETTERS
    }
    if not all(len(ids) == 1 for ids in encoded.values()):
        raise ValueError(f"A-D are not all single tokens for this tokenizer: {encoded}")
    ids = {letter: values[0] for letter, values in encoded.items()}
    if len(set(ids.values())) != 4:
        raise ValueError(f"A-D do not map to four distinct tokens: {ids}")
    return ids


def score_prompt(model, tokenizer, prompt, candidate_ids):
    """Score the next-token probability of A-D, normalized within those options."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].float()
    option_logits = torch.tensor(
        [logits[candidate_ids[letter]].item() for letter in LETTERS],
        dtype=torch.float64,
    )
    probabilities = torch.softmax(option_logits, dim=0).tolist()
    chosen_index = int(torch.argmax(option_logits).item())
    return {
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "input_tokens": int(inputs.input_ids.shape[-1]),
        "parsed_answer": LETTERS[chosen_index],
        "option_logits": dict(zip(LETTERS, option_logits.tolist())),
        "option_probabilities": dict(zip(LETTERS, probabilities)),
        "top_two_margin": float(sorted(probabilities, reverse=True)[0] - sorted(probabilities, reverse=True)[1]),
    }


def append_record(path, record):
    """Append and flush one record so an interrupted Kaggle session can resume."""
    with path.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
