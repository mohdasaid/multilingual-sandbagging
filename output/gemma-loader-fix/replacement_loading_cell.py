CACHE_DIR = "/tmp/sandbagging-huggingface-cache"
os.environ.update({
    "HF_HOME": CACHE_DIR, "HF_HUB_CACHE": f"{CACHE_DIR}/hub",
    "TRANSFORMERS_CACHE": f"{CACHE_DIR}/hub", "HF_HUB_DISABLE_XET": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
})
from kaggle_secrets import UserSecretsClient
from huggingface_hub import HfApi
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoProcessor,
                          AutoTokenizer, BitsAndBytesConfig, Gemma3ForConditionalGeneration)

HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
MODEL_ID = CFG["model_id"]
EXPERIMENT_VERSION = f"{CFG['tag']}-{RUN_VERSION}-n{N_ITEMS}-g{GENERATION_MAX_NEW_TOKENS}-a{ANSWER_FIRST_MAX_NEW_TOKENS}-r{REASON_FIRST_MAX_NEW_TOKENS}"
OUT = Path("/kaggle/working") / EXPERIMENT_VERSION
OUT.mkdir(parents=True,exist_ok=True)
if CFG["loader"] == "gemma_prequantized" and any(OUT.glob("*responses*.jsonl")):
    raise RuntimeError("Use a new RUN_VERSION for the Gemma precision correction; existing response files must not be resumed.")
prior_path = OUT / "run_manifest.json"
prior_manifest = json.loads(prior_path.read_text()) if prior_path.exists() else None
if prior_manifest:
    assert prior_manifest["model_id"] == MODEL_ID
model_revision = prior_manifest["model_revision"] if prior_manifest else HfApi(token=HF_TOKEN).model_info(MODEL_ID).sha
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)

# Release the previous model before allocating its replacement.
if "model" in globals():
    del model
gc.collect()
torch.cuda.empty_cache()
processor = None
common = dict(revision=model_revision, token=HF_TOKEN, cache_dir=CACHE_DIR,
              device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager")
if CFG["loader"] == "seq2seq_nf4":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_revision, token=HF_TOKEN, cache_dir=CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, quantization_config=quant,
        max_memory={0: "14GiB", 1: "14GiB", "cpu": "30GiB"}, **common)
elif CFG["loader"] == "gemma_prequantized":
    from transformers import AutoConfig

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=model_revision,
        token=HF_TOKEN,
        cache_dir=CACHE_DIR,
    )
    tokenizer = processor.tokenizer

    # Preserve the checkpoint's quantization configuration.
    checkpoint_config = AutoConfig.from_pretrained(
        MODEL_ID,
        revision=model_revision,
        token=HF_TOKEN,
        cache_dir=CACHE_DIR,
    )
    saved_quant = checkpoint_config.quantization_config
    quant_dict = (
        saved_quant.to_dict()
        if hasattr(saved_quant, "to_dict")
        else dict(saved_quant)
    )

    if not quant_dict.get("load_in_4bit", False):
        raise RuntimeError("Expected a checkpoint quantized to 4 bits.")

    quant_dict["bnb_4bit_compute_dtype"] = "float32"
    gemma_quant = BitsAndBytesConfig.from_dict(quant_dict)
    checkpoint_config.quantization_config = gemma_quant.to_dict()

    gemma_common = dict(common)
    gemma_common["torch_dtype"] = torch.float32

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        config=checkpoint_config,
        **gemma_common,
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_revision, token=HF_TOKEN,
        cache_dir=CACHE_DIR, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant, **common)
model.eval()
if CFG["loader"] == "gemma_prequantized":
    import bitsandbytes as bnb
    compute_dtypes = {m.compute_dtype for m in model.modules()
                      if isinstance(m, bnb.nn.Linear4bit)}
    if compute_dtypes != {torch.float32}:
        raise RuntimeError(f"Gemma FP32 compute override was not applied: {compute_dtypes}")
    print("Verified Gemma 4-bit layer compute dtype:", compute_dtypes)
if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
input_device = model.get_input_embeddings().weight.device
del HF_TOKEN

EXPERIMENT_VERSION = f"{CFG['tag']}-{RUN_VERSION}-n{N_ITEMS}-g{GENERATION_MAX_NEW_TOKENS}-a{ANSWER_FIRST_MAX_NEW_TOKENS}-r{REASON_FIRST_MAX_NEW_TOKENS}"
OUT = Path("/kaggle/working") / EXPERIMENT_VERSION
OUT.mkdir(parents=True, exist_ok=True)

def render_chat(prompt):
    if processor is not None:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
    if CFG["architecture"]=="seq2seq": return prompt
    raise ValueError("Causal model has no chat template; configure an audited template explicitly.")

def candidate_suffix_ids(prefix):
    # The actual candidate emitted after the prefix is space+A/B/C/D.
    result = {}
    for x in LETTERS:
        suffix = tokenizer.encode(" " + x, add_special_tokens=False)
        assert suffix, (prefix,x)
        assert tokenizer.decode(suffix,skip_special_tokens=True).strip()==x, (prefix,x,suffix)
        result[x] = suffix
    assert len({tuple(v) for v in result.values()})==4
    return result

candidate_ids_by_prefix = {lang: candidate_suffix_ids(prefix) for lang, prefix in ANSWER_PREFIXES.items()}
candidate_ids_by_prefix["threeway"] = candidate_suffix_ids("Answer:")

def encode_checked(text, add_special_tokens=True, reserve=0):
    encoded=tokenizer(text,add_special_tokens=add_special_tokens,return_tensors="pt",truncation=False).to(input_device)
    limits=[getattr(tokenizer,"model_max_length",None)]
    cfg=getattr(model.config,"text_config",model.config)
    limits.append(getattr(cfg,"max_position_embeddings",None))
    limits=[int(x) for x in limits if isinstance(x,(int,float)) and 0<x<1_000_000]
    if limits:
        assert encoded.input_ids.shape[1]+reserve<=min(limits), "Input exceeds audited context budget; no silent truncation allowed."
    return encoded

def score_prompt(prompt, prefix, candidate_key):
    candidates=candidate_ids_by_prefix[candidate_key]
    seq=CFG["architecture"]=="seq2seq"
    rendered=prompt if seq else render_chat(prompt)
    enc=encode_checked(rendered if seq else rendered+prefix,add_special_tokens=seq,
                       reserve=0 if seq else max(map(len,candidates.values())))
    if seq:
        start=model.config.decoder_start_token_id
        assert start is not None
        base=torch.tensor([[start]+tokenizer.encode(prefix,add_special_tokens=False)],device=input_device)
    else: base=enc.input_ids
    def forward(ids):
        if seq: return model(**enc,decoder_input_ids=ids).logits[0].float()
        return model(input_ids=ids,attention_mask=torch.ones_like(ids)).logits[0].float()
    with torch.inference_mode():
        first=forward(base)[-1]
        if not torch.isfinite(first).all().item():
            raise FloatingPointError("Non-finite initial logits; no answer may be saved.")
        logfirst=torch.log_softmax(first,dim=-1)
        loglik=[]
        for letter in LETTERS:
            suffix=candidates[letter]
            if len(suffix)==1: score=logfirst[suffix[0]]
            else:
                ids=torch.cat([base,torch.tensor([suffix[:-1]],device=input_device)],dim=1)
                candidate_logits=forward(ids)[base.shape[1]-1:]
                if not torch.isfinite(candidate_logits).all().item():
                    raise FloatingPointError("Non-finite continuation logits; no answer may be saved.")
                lp=torch.log_softmax(candidate_logits,dim=-1)
                score=sum(lp[j,t] for j,t in enumerate(suffix))
            loglik.append(score)
        scores=torch.stack(loglik)
        if not torch.isfinite(scores).all().item():
            raise FloatingPointError("Non-finite candidate scores; no answer may be saved.")
        probs=torch.softmax(scores,dim=0).cpu().tolist()
        if (not all(math.isfinite(p) and 0 <= p <= 1 for p in probs)
                or not math.isclose(sum(probs), 1.0, abs_tol=1e-5)):
            raise FloatingPointError("Invalid option probabilities; no answer may be saved.")
        single=all(len(v)==1 for v in candidates.values())
        mass=float(torch.exp(scores).sum()) if single else None
    order=sorted(probs,reverse=True)
    return {"parsed_answer":LETTERS[int(torch.argmax(scores))],"option_probabilities":dict(zip(LETTERS,probs)),
      "candidate_log_likelihoods":dict(zip(LETTERS,scores.cpu().tolist())),
      "scoring_mode":"single_token" if single else "sequence_log_likelihood",
      "candidate_mass":mass,"candidate_sequence_probability_sum":float(torch.exp(scores).sum()),
      "unrestricted_top_token_id":int(first.argmax()),"unrestricted_top_token":tokenizer.decode([int(first.argmax())]),
      "candidate_entropy":float(-sum(p*math.log(p) for p in probs if p>0)),
      "top_two_margin":order[0]-order[1],"input_tokens":int(enc.input_ids.shape[1]),
      "rendered_prompt":rendered,"answer_prefix":prefix,
      "scored_input_sha256":hashlib.sha256((rendered+"<prefix>"+prefix).encode()).hexdigest()}

audit_prompt = make_prompt(balanced_items.iloc[0], "en", "en", "honest")
print("Rendered prompt:\n", render_chat(audit_prompt)[:2000])
print("Candidate IDs:", candidate_ids_by_prefix)
print("Model loaded; scoring runs only after manifest verification.")