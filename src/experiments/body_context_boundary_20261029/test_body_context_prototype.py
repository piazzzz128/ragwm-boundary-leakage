import unittest
from body_context_prototype import classify_context,eligible


class BodyContext(unittest.TestCase):
    def setUp(self):self.title='Public article';self.text=self.title+'\nFirst normal body sentence. Another normal sentence.'
    def test_title_body_boundary_has_no_body_context(self):
        a=len(self.title)+1
        self.assertEqual(classify_context(self.text,self.title,a,len(self.text)),'NO_BODY_CONTEXT')
    def test_internal_title_boundary_rejected(self):
        self.assertEqual(classify_context(self.text,self.title,7,len(self.text)),'WITHIN_TITLE_METADATA')
    def test_body_boundary_eligible(self):
        a=self.text.index('Another')
        self.assertTrue(eligible(self.text,self.title,a,len(self.text)))
    def test_append_after_nonempty_body_eligible(self):
        a=len(self.text)+1;text=self.text+' '+('Synthetic fixture only, no generated watermark.')
        self.assertTrue(eligible(text,self.title,a,len(text)))
    def test_whitespace_before_first_body_token_not_context(self):
        text=self.title+'\n \t  Body words and another sufficiently long target.'
        a=text.index('Body')
        self.assertFalse(eligible(text,self.title,a,len(text)))
    def test_wrong_or_missing_metadata_stops(self):
        for title in ['Wrong title','',None]:
            with self.assertRaises(ValueError):eligible(self.text,title,20,len(self.text))
    def test_no_original_text_modification(self):
        original=self.text
        eligible(self.text,self.title,self.text.index('Another'),len(self.text))
        self.assertEqual(self.text,original)


if __name__=='__main__':unittest.main()
