'''
Created on Apr 21, 2014

@author: Prateek
'''
import unittest
from mywork.task2 import *

class Test(unittest.TestCase):


    def testignorewords(self):
        words = [u'']
        ignore_words_list = ignoreWordsList(words)
        assert len(ignore_words_list) == 0
        words = ['mac']
        ignore_words_list = ignoreWordsList(words)
        assert cmp(sorted(ignore_words_list),sorted(['macintosh', 'mackintosh', 'mac', 'mack'])) == 0
        words = ['mac', 'lotion']
        ignore_words_list = ignoreWordsList(words)
        assert cmp(sorted(ignore_words_list),sorted(['macintosh', 'mackintosh', 'mac', 'mack', 'lotion', 'application'])) == 0
        words = []
        ignore_words_list = ignoreWordsList(words)
        assert len(ignore_words_list) == 0
        
    def testusualnunusualwords(self):
        words = [u'']
        usual,unusual = usualAndUnusualWords(words)
        assert len(usual) == len(unusual) == 0
        
    def testfindgroups(self):
        
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()