# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:53:54 2021

@author: Eya
"""

#on test le model demand
import unittest
import model_demand as d

class TestConsumption(unittest.TestCase):
    f = d.import_csv("DE.csv",  ";", True)
    sig = d.Optimize_sigmoid(f)
    sig.optimize()
    def get_consumptionTest(self):
        #je teste si la temperature ne vaut pas 40, car valeur interdite
        self.assertNotEquals(self.temperature,40)
   
   
   
if __name__ == '__main__':
    unittest.main()