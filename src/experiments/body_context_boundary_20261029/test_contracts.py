"""No real labels, model forward, API, or corpus/source scan."""
import json
import pathlib
import re
import tempfile
import unittest
from contracts import *

ORIGINAL = '''
def norm_space(value: str) -> str:
    return re.sub(r"\\s+", " ", (value or "")).strip()
def tail_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    text = norm_space(text)
    if len(text) <= max_chars: return text
    cut = text[-max_chars:]; first_space = cut.find(" ")
    if first_space > 0: cut = cut[first_space + 1:]
    return cut.strip()
def sentence_spans(text: str) -> List[Tuple[int,int]]:
    raw=[]; start=0
    for match in re.finditer(r"(?<=[.!?])\\s+", text):
        raw.append((start,match.start())); start=match.end()
    raw.append((start,len(text))); out=[]
    for a,b in raw:
        while a<b and text[a].isspace():a+=1
        while b>a and text[b-1].isspace():b-=1
        if a<b:out.append((a,b))
    return out
def last_eligible_sentence_span(text: str, min_target_chars: int = MIN_TARGET_CHARS) -> Optional[Tuple[int,int]]:
    for a,b in reversed(sentence_spans(text)):
        if len(norm_space(text[a:b]))>=min_target_chars:return a,b
    return None
'''


class FakeTokenizer:
    def __call__(self, text, **kw):
        return {'input_ids': [ord(c) for c in text][:kw['max_length']]}


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.temp.name) / 'original.py'
        self.path.write_text(ORIGINAL, encoding='utf-8')
        self.cfg = {'scorer_path': str(self.path), 'request_identity': {'model': 'fixture', 'alignment': 'T[1:]'}}
    def tearDown(self):
        self.temp.cleanup()
    def test_author_preprocess_exact(self):
        text = '  Unicode\u3000café, 3.14 7:30 End.\tNext!\n🙂? a,b  '
        w, e = word_map(text)
        self.assertEqual(w, re.sub(r'(?<!\d)[.,;:!?](?!\d)', '', text).split())
        self.assertEqual(text[:e[w.index('End')]][-4:], 'End.')
        self.assertEqual(text[:e[w.index('3.14')]][-4:], '3.14')
    def test_empty_and_punctuation_only(self):
        for text in ['', '   ', '... ! ?\t;:']:
            self.assertEqual(word_map(text), ([], []))
    def test_windows_cover_once_all_edges(self):
        for n in [0,1,447,448,449,479,480,481,896,928,929,4001]:
            owned=[p for a,b,e in windows(n) for p in range(a,e)]
            self.assertEqual(owned,list(range(n)))
            self.assertTrue(all(b-a<=480 for a,b,e in windows(n)))
    def test_long_word_subtokens_not_lost_or_duplicated(self):
        ids=[0]*1300+[1]*10+[2]*350
        logits=[[0]+[1 if p==1299 else 0 for p in range(a,b)]+[0] for a,b,e in windows(len(ids))]
        self.assertEqual(assign_word_labels(ids,logits,{0:'0',1:'.'}),{0:'.',1:'0',2:'0'})
    def test_punctuation_not_split_into_a_new_span(self):
        text='An original sentence ending here. Another original sentence ending here.'
        _, ends=word_map(text); original=text_functions(self.path)['sentence_spans']
        cuts=[ends[4],ends[-1]]
        self.assertEqual(virtual_spans(text,cuts,original),original(text))
    def test_raw_offsets_whitespace_unicode(self):
        text='A long context without a terminal mark\u2028\tSecond long target sentence containing café🙂'
        cut=text.index('\u2028');spans=virtual_spans(text,[cut],lambda s:[(0,len(s))])
        self.assertEqual(text[spans[1][0]:spans[1][1]],'Second long target sentence containing café🙂')
    def test_depth3_and_first_accept_order(self):
        text=' '.join('Sentence number %s has enough raw characters.'%i for i in range(7))
        doc={'doc_id':'d','component':'d','text':text}
        cc=candidates(self.cfg,FakeTokenizer(),doc)
        self.assertEqual(len(cc),3)
        self.assertGreater(cc[0]['a'],cc[1]['a'])
        for c in cc:c['RAW']=1.
        self.assertEqual(first_action(list(reversed(cc)),0,'RAW')['rank'],1)
    def test_lu_lc_identical_target_labels(self):
        ti=[6,7,8,9];ci=[1,2,3]
        u,up=positions([],ti);c,cp=positions(ci,ti)
        self.assertEqual([ti[i] for i in u],[(ci+ti)[i] for i in c])
        self.assertEqual([i+1 for i in up],u)
        self.assertEqual([i+1 for i in cp],c)
    def test_cache_key_kind_target_context_and_model(self):
        a=request_key({'model':'a'},'Lu',[],[1,2]); b=request_key({'model':'a'},'Lc',[4],[1,2])
        self.assertNotEqual(a,b)
        self.assertNotEqual(a,request_key({'model':'b'},'Lu',[],[1,2]))
        with self.assertRaises(ValueError):request_key({},'Lu',[5],[1,2])
    def test_calibration_ties_and_first_action_damage(self):
        docs=[{'doc_id':str(i),'text':'a'*1000} for i in range(100)]
        rows=[{'doc_id':str(i),'rank':r,'a':0,'b':length,'RAW':score}
              for i in range(5) for r,length,score in [(1,20,10-i),(2,400,9-i)]]
        result=calibrate_joint(docs,rows,'RAW',.03,.005)
        self.assertLessEqual(result['selected_documents'],3)
        self.assertLessEqual(result['conservative_char_upper'],500)
    def test_unknown_all_slots_kept(self):
        self.assertEqual(outcome_bounds([1,0,2,2]),{'YES':1,'NO':1,'Unknown':2,'units':4,'WSN_Unknown_bounds':[1,3]})
    def test_numpy_serialization_no_nan(self):
        import numpy as np
        self.assertEqual(json.loads(canon({'a':np.bool_(True),'b':np.int64(3)})),{'a':True,'b':3})
        with self.assertRaises(ValueError):canon({'a':np.float64(float('nan'))})
    def test_jsonl_physical_line(self):
        p=pathlib.Path(self.temp.name)/'rows.jsonl';p.write_bytes(canon({'text':'a\u2028b'})+b'\n'+canon({'text':'c'})+b'\n')
        with p.open(encoding='utf-8') as f:rows=[json.loads(line) for line in f]
        self.assertEqual(len(rows),2)
    def test_boundary_cache_includes_tokenizer_and_policy(self):
        from boundary import cache_key
        a=cache_key({'boundary_model_assets':{'weight':'a','tokenizer':'b'}},'same text')
        self.assertNotEqual(a,cache_key({'boundary_model_assets':{'weight':'a','tokenizer':'c'}},'same text'))
    def test_readonly_sqlite_refuses_active_wal(self):
        import sqlite3
        from common import connect_ro
        path=pathlib.Path(self.temp.name)/'cache.sqlite3'
        db=sqlite3.connect(path);db.execute('CREATE TABLE t(v INTEGER)');db.commit();db.close()
        pathlib.Path(str(path)+'-wal').write_bytes(b'unfinished')
        with self.assertRaises(AssertionError):connect_ro(path)
    def test_original_text_never_gets_a_predicted_period(self):
        text='An original context with no period at its end then a sufficiently long target'
        cut=text.index('then');doc={'doc_id':'d','component':'d','text':text}
        cc=candidates(self.cfg,FakeTokenizer(),doc,[cut])
        self.assertEqual(doc['text'],text)
        self.assertTrue(cc)
        self.assertNotIn(ord('.'),cc[0]['context_token_ids']+cc[0]['target_token_ids'])
    def test_failed_calibration_cannot_silently_become_zero(self):
        with self.assertRaises(ValueError):calibrate_joint([{'doc_id':'d','text':'abc'}],[],'RAW',.03,.005)
    def test_joined_normalizer_and_invisible_format_offsets(self):
        class Joined:
            def __call__(self,text,**kwargs):
                self.text=text
                pairs=[(ord(c),(i,i+1)) for i,c in enumerate(text) if not c.isspace() and c!='\ufeff']
                return {'input_ids':[a for a,b in pairs],'offset_mapping':[b for a,b in pairs]}
        tk=Joined();ids,ownership=tokenize_words(tk,['alpha\ufeff','\ufeff','bravo'])
        self.assertEqual(tk.text,'alpha\ufeff \ufeff bravo')
        self.assertIsNone(ownership[1][1])
        predictions=assign_word_labels(ownership,[[0]+[1]*len(ids)+[0]],{0:'0',1:'.'})
        self.assertEqual(predictions,{0:'.',1:'0',2:'.'})
    def test_joined_tokens_shared_across_word_offsets(self):
        labels=assign_word_labels((1,{0:0,1:0}),[[0,1,0]],{0:'0',1:'.'})
        self.assertEqual(labels,{0:'.',1:'.'})


if __name__=='__main__':
    unittest.main()
