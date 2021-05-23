# -*- coding: utf-8 -*-
"""
Created on Fri May 21 17:55:41 2021

@author: dondu
"""

#himport Consumption
import supply_model as s
import model_demand as d
import numpy as np
import pandas as pd
import dfply 
from dfply import *



#demand side of model

f = d.import_csv("C:/Users/dondu/Documents/IN104/Balance/De.csv",  ";", True)
sig = d.Optimize_sigmoid(f)
sig.optimize() #la temperature
c = sig.create_Consumption()
c.sigmoid()

#création d'un Dataframe
n=len(sig.Date)
D=[]
for i in range (n):
    D.append(c.get_consumption(sig.T[i]))

dfconso=pd.DataFrame({"Date":sig.Date,"Demand":D})
dict_storage=pd.read_excel('storage_data.xlsx',sheet_name=None)
df_price=pd.read_excel('price_data.xlsx')
dict_storage_join=s.preparer_le_dict_à_la_prediction(dict_storage,df_price)

NW=[]*n
for key in dict_storage_join.keys():
    P=s.Prediction(dict_storage_join,key)
    P.logistic_reg()
    #P.comparaison()
    #if(P.choix==0):
      #  P.random_forest()
    #else:
     #   P.logistic_reg()
#on a comme attribut le vecteur Net withdraal Binary
    y_pred_binary=list(P.y_pred_train) + list(P.y_pred)
    y_pred_binary=np.array(y_pred_binary)
    dfy_pred_binary=pd.DataFrame({"Date":dict_storage_join[key]['Date'],"y_pred_binary":y_pred_binary})
    dict_storage_join[key] = pd.concat((dict_storage_join[key], dfy_pred_binary), axis = 1)
    dict_storage_join[key] = pd.merge(dfy_pred_binary, dict_storage[key], on='Date')
    
    Temp = dict_storage_join[key] >> mask(y_pred_binary > 0)
    
    #Créer X_reg avec temp : c'est X_mod
    P.regression_lineaire()
#Vous avez maintenant un vecteur y_pred_num : y_mod_pred
    Temp = pd.concat((Temp['Date '],P.y_mod_pred), axis = 1)
    dict_storage_join[key] = pd.merge(dict_storage_join[key], Temp, on = 'Date', how  = 'outer')
    

        
        

    
    
    
    


