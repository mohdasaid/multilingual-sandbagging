import unittest, tempfile, json, zipfile
from pathlib import Path
from unittest.mock import patch
import gemini_runner as g
from parser_v6 import extract_answer

g.extract_answer=extract_answer
HERE=Path(__file__).parent
z=zipfile.ZipFile('/home/muhammad/Downloads/afriquellama-8b-nf4-corrected-v5-afrique-n20-g16-a128-r192.zip')
PROMPTS=json.loads(z.read('run_manifest.json'))['prompts']
SETTINGS=dict(model='gemini-2.5-flash-lite', max_output_tokens={'answer_only':32})
class Response:
    def __init__(self,status=200,text='A'):self.status_code=status;self.text=text
    def json(self):return dict(candidates=[dict(content=dict(parts=[dict(text=self.text)]),finishReason='STOP')],modelVersion='fixed-test-model')
class Tests(unittest.TestCase):
    def setUp(self):
        self.items=g.load_items(HERE/'balanced_items.jsonl',10)
        self.jobs=g.make_jobs(self.items,PROMPTS,[0],['answer_only'])
    def test_counts_and_selection(self):
        self.assertEqual(len(self.jobs),160)
        self.assertEqual(self.items,g.load_items(HERE/'balanced_items.jsonl',10))
        counts=[sum(r['answer_en']==a for r in self.items) for a in 'ABCD']
        self.assertLessEqual(max(counts)-min(counts),1)
        with self.assertRaises(AssertionError):g.load_items(HERE/'balanced_items.jsonl',21)
    def test_reviewed_hausa_and_rotations(self):
        item=self.items[0]
        jobs=g.make_jobs([item],PROMPTS,[0,1,2,3],['answer_only'])
        for job in jobs:
            lang='en' if job['question_language']=='en' else 'ha_reviewed'
            expected=item[f"option_{item['answer_en'].lower()}_{lang}"]
            self.assertIn(job['gold']+'. '+expected,job['prompt'])
    def test_parser_no_fallback_or_gold(self):
        for text in ['', 'I cannot answer.', 'A. one\nB. two', 'Answer: A\nAnswer: B','A cikin wannan']:
            self.assertFalse(extract_answer(text)['answer_extractable'])
        self.assertEqual(extract_answer('**B**')['extracted_answer'],'B')
    def test_resume_errors_and_throttle(self):
        with tempfile.TemporaryDirectory() as d:
            out=g.initialize(d,self.jobs,SETTINGS,self.items,'code')
            with patch.object(g.time,'sleep') as sleep, patch.object(g.time,'time',return_value=1000):
                records=g.run_jobs(self.jobs,out,SETTINGS,'secret',max_calls=2,post=lambda *a,**kw:Response())
                self.assertEqual(len(records),2)
                self.assertGreaterEqual(sleep.call_args.args[0],6.1)
                records=g.run_jobs(self.jobs,out,SETTINGS,'secret',max_calls=1,post=lambda *a,**kw:Response(429))
                self.assertEqual(len(records),2)
                records=g.run_jobs(self.jobs,out,SETTINGS,'secret',max_calls=1,post=lambda *a,**kw:Response(text=''))
                self.assertEqual(len(records),3)
                self.assertFalse(records[-1]['correct'])
                self.assertEqual(records[-1]['extracted_answer'],'')
            self.assertNotIn('secret',(Path(d)/'responses.jsonl').read_text())
            with self.assertRaises(AssertionError):g.initialize(d,self.jobs,{**SETTINGS,'model':'different'},self.items,'code')
    def test_partial_and_question_clustered_analysis(self):
        jobs=g.make_jobs(self.items[:2],PROMPTS,[0,1,2,3],['answer_only'])
        records=[dict(**j,correct=j['condition']=='honest',answer_extractable=True,strict_format_compliant=True,model_version='fixed',finish_reason='STOP',raw_response={}) for j in jobs]
        with tempfile.TemporaryDirectory() as d:
            g.analyze(jobs,records[:-1],d,draws=20)
            self.assertFalse((Path(d)/'paired_effects.csv').exists())
            g.analyze(jobs,records,d,draws=20)
            effects=g.pd.read_csv(Path(d)/'paired_effects.csv')
            self.assertTrue((effects.paired_base_questions==2).all())
            self.assertTrue((effects.loc[effects.contrast=='direct_sandbag_minus_honest','estimate']==-1).all())
    def test_notebook_compiles(self):
        nb=json.loads((HERE/'gemini-sandbagging-api.ipynb').read_text())
        for cell in nb['cells']:
            source=''.join(cell['source'])
            if cell['cell_type']=='code' and not source.startswith('%'):compile(source,'cell','exec')
if __name__=='__main__':unittest.main()
