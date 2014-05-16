'''
Created on Apr 21, 2014

@author: Prateek
'''
import unittest
from mywork.task2 import *

class Test(unittest.TestCase):


    def test_ignorewords(self):
        words = [u'']
        ignore_words_list = ignoreWordsList(words)
        assert len(ignore_words_list) != 0
        words = ['mac']
        ignore_words_list = ignoreWordsList(words)
        assert cmp(sorted(ignore_words_list),sorted(['macintosh', 'mackintosh', 'mac', 'mack'])) == 0
        words = ['mac', 'lotion']
        ignore_words_list = ignoreWordsList(words)
        assert cmp(sorted(ignore_words_list),sorted(['macintosh', 'mackintosh', 'mac', 'mack', 'lotion', 'application'])) == 0
        words = []
        ignore_words_list = ignoreWordsList(words)
        assert len(ignore_words_list) == 0
        
    def test_usualAndunusualwords(self):
        words = [u'']
        usual,unusual = usualAndUnusualWords(words)
        assert len(usual) == len(unusual) == 0
        
    def test_find_groups(self):
        pass
    
    def test_tuple_with_list(self):
        pass
    
    def test_WordContext(self):
        pass
    
    def test_populateStopWords(self):
        pass
    
    def test_labllist(self):
        pass
    
    def test_topicslist(self):
        pass
    
    def test_topic_label_list(self):
        pass
    
    def test_performSVD(self):
        pass
    
    def test_accumulate(self):
        pass
    
    def test_tfidf_info(self):
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()