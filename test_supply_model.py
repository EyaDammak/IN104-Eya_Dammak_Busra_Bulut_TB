# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:53:54 2021

@author: Eya
"""

import supply_model as s
import model_demand as d
import unittest
import numpy as np
import pandas as pd


class TestPrediction(unittest.TestCase):
    dict_storage=pd.read_excel('storage_data.xlsx',sheet_name=None)
    df_price=pd.read_excel('price_data.xlsx')

    def testLaggedNW(self): #on teste que Lagged_NW et NW est bien décalé de 1jour
        self.dict_storage_join=s.preparer_le_dict_à_la_prediction(self.dict_storage,self.df_price)
        for key in self.dict_storage_join:
            Lagged=self.dict_storage_join[key]['Lagged_NW'].values
            NW=self.dict_storage_join[key]['Net withdrawal'].values
            for i in range (1,1629):
                self.assertEqual(Lagged[i+1],NW[i])


    def test_binaryNW(self):
        self.dict_storage_join=s.preparer_le_dict_à_la_prediction(self.dict_storage,self.df_price)
    #je teste si dans Net Withdrawal binary ya que des 1 et 0
        for key in self.dict_storage_join:
            P=s.Prediction(self.dict_storage_join,key)
            P=P.logistic_reg()
            y_pred_binary=list(P.y_pred_train) + list(P.y_pred) #c'est la colonne net withdrawal binary prédite
            y_pred_binary=np.array(y_pred_binary)
            for i in range(len(y_pred_binary)):
                self.assertEqual(y_pred_binary[i],0) or self.assertEqual(y_pred_binary[i],1)
       
           

if __name__ == '__main__':
    unittest.main()