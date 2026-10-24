"""Create offline reading copies; record actual decisions in the review CSV files."""
from pathlib import Path
import csv
from html import escape

root = Path(__file__).resolve().parent / 'data'
for pool in ('development', 'pilot'):
    with (root / f'{pool}_review.csv').open() as f:
        rows = list(csv.DictReader(f))
    cards = []
    for i, row in enumerate(rows, 1):
        body = []
        for field in ('question', 'option_a', 'option_b', 'option_c', 'option_d'):
            body.append('<tr><th>' + escape(field.replace('_', ' ')) + '</th><td>' + escape(row[field+'_en']) + '</td><td lang="ha">' + escape(row[field+'_ha_reviewed']) + '</td></tr>')
        cards.append(f'<article id="item-{i}"><h2>{i}. {escape(row["sample_id"])} — {escape(row["subject_en"])}</h2><table><thead><tr><th>Field</th><th>English</th><th>Hausa</th></tr></thead><tbody>{"".join(body)}</tbody></table><p>Source answer: <strong>{escape(row["answer_en"])}</strong></p></article>')
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>English–Hausa '+pool+' review</title><style>body{font:17px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;background:#f6f7f9;color:#172234}article{background:white;padding:24px;margin:24px 0;border:1px solid #d9e0e9;border-radius:12px}table{width:100%;border-collapse:collapse;table-layout:fixed}td,th{vertical-align:top;text-align:left;padding:12px;border:1px solid #ddd;white-space:pre-wrap;overflow-wrap:anywhere}th:first-child{width:12%}h2{font-size:20px}@media print{article{break-inside:avoid}}</style><h1>English–Hausa '+pool+' review</h1><p>Reading copy only. Record corrections and decisions in <strong>'+pool+'_review.csv</strong>. These items are unreviewed. Check the question, every option, and the source answer.</p>'+''.join(cards)+'</html>'
    (root / f'{pool}_review.html').write_text(html)
    print(f'Created {pool} reading copy: {len(rows)} items')
