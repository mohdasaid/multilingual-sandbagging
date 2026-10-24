"""Download a pinned dataset and prepare disjoint review pools; no model calls."""
from pathlib import Path
import hashlib
import json
import platform
import random
import requests
import pandas as pd

REPO = "CohereLabs/Global-MMLU"
REVISION = "0e619dbeb34206cd48705a1a0ea7fb21cae09993"
SEED = 42
OUT = Path('/kaggle/working/multilingual-pilot') if Path('/kaggle').exists() else Path(__file__).resolve().parent / 'data'
OUT.mkdir(parents=True, exist_ok=True)

def download(language):
    path = OUT / f'{language}-test.parquet'
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{language}/test-00000-of-00001.parquet'
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    frame = pd.read_parquet(path)
    assert frame['sample_id'].notna().all()
    assert frame['sample_id'].is_unique
    return frame, {'url': url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': len(frame)}

def main():
    frames, sources = {}, {}
    for language in ('en', 'ha'):
        print(f'Downloading {language} from pinned revision {REVISION}...')
        frames[language], sources[language] = download(language)
    paired = frames['en'].merge(frames['ha'], on='sample_id', suffixes=('_en', '_ha'), validate='one_to_one')
    assert len(paired) == len(frames['en']) == len(frames['ha']), 'Unmatched items require investigation.'
    assert paired['subject_en'].eq(paired['subject_ha']).all(), 'Subject mismatch.'
    assert paired['answer_en'].eq(paired['answer_ha']).all(), 'Answer key mismatch.'
    assert paired['answer_en'].isin(list('ABCD')).all(), 'Unexpected answer labels.'
    ids = sorted(paired['sample_id'].tolist())
    random.Random(SEED).shuffle(ids)
    pools = {'development': ids[:20], 'pilot': ids[20:120]}
    assert not set(pools['development']) & set(pools['pilot'])
    columns = ['sample_id', 'subject_en', 'subject_category_en']
    for field in ['question', 'option_a', 'option_b', 'option_c', 'option_d']:
        columns.extend([f'{field}_en', f'{field}_ha'])
    columns += ['answer_en', 'cultural_sensitivity_label_en']
    for name, selected_ids in pools.items():
        review = paired.set_index('sample_id').loc[selected_ids].reset_index()[columns].copy()
        for field in ['question', 'option_a', 'option_b', 'option_c', 'option_d']:
            review[f'{field}_ha_reviewed'] = review[f'{field}_ha']
        review['review_status'] = 'pending'
        review['review_notes'] = ''
        review['reviewer'] = ''
        review['review_date'] = ''
        path = OUT / f'{name}_review.csv'
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite review work: {path}. Use a new output folder.')
        review.to_csv(path, index=False)
        summary = review.groupby('subject_category_en', dropna=False).size().to_dict()
        print(f'{name}: {len(review)} items; categories: {summary}')
    manifest = {
        'dataset': REPO, 'revision': REVISION, 'sources': sources,
        'seed': SEED, 'sampling': 'Uniform shuffle of sorted matched sample IDs; no response-based filtering.',
        'pools': pools, 'matched_items': len(paired),
        'versions': {'python': platform.python_version(), 'pandas': pd.__version__, 'requests': requests.__version__},
        'status': 'Unreviewed pools. Not approved experimental materials.'
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f'Review files saved to {OUT}. No model calls were made.')

if __name__ == '__main__':
    main()
