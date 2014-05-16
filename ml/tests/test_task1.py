'''
Created on Apr 21, 2014

@author: Prateek
'''
import unittest
from mywork.task1 import *

class Test(unittest.TestCase):

    def setUp(self):
        self.data = ["Netis Wireless N300 AP Router Repeater $15 + Free Shipping", 
"Architectural Digest Magazine $5 per year", 
"Saints Row: The Third (Xbox 360 Digital Download Game) Free (Xbox Live Gold Membership Required)",
"ARMA: Cold War Assault (PC Digital Download) Free",
"500GB Sony PlayStation 4 Console (Pre-Owned) $300 + Free Shipping"]

#     def tearDown(self):
#         pass

    def testWordCounts(self):
        print self.data
        sorted = wordCounts(self.data)
        print sorted
    
    def testTypesOfObject(self):
        pass
    
    def testBigramMetrics(self):
        pass

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()