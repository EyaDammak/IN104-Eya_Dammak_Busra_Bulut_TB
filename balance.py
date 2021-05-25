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

f = d.import_csv("DE.csv",  ";", True)
sig = d.Optimize_sigmoid(f)
sig.optimize() #self.T ; self.coef ; self.model_ajust_conso ; self.__cov ; self.__corr ; self.__nrmse ...
c = sig.create_Consumption()
c.sigmoid() #self.model_conso

#création d'un Dataframe
n=len(sig.Date)
D=[]
for i in range (n):
    D.append(c.get_consumption(sig.T[i]))

df_demand=pd.DataFrame({"Date":sig.Date,"Demand":D})
dict_storage=pd.read_excel('storage_data.xlsx',sheet_name=None)
df_price=pd.read_excel('price_data.xlsx')
dict_storage_join=s.preparer_le_dict_à_la_prediction(dict_storage,df_price)

df_metrics={}
for key in dict_storage_join.keys():
    P=s.Prediction(dict_storage_join,key)
    P.logistic_reg() #self.metrics_lr ; self.y_pred
 
#on a comme attribut le vecteur Net withdraal Binary
    y_pred_binary=list(P.y_pred_train) + list(P.y_pred)
    y_pred_binary=np.array(y_pred_binary)
    dfy_pred_binary=pd.DataFrame({"gasDayStartedOn":dict_storage_join[key]['gasDayStartedOn'],"y_pred_binary":y_pred_binary})
    #dict_storage_join[key] = pd.concat((dict_storage_join[key], dfy_pred_binary), axis = 1)
    dict_storage_join[key] = pd.merge(dict_storage[key],dfy_pred_binary, on='gasDayStartedOn',how='outer')
    colonne=dict_storage_join[key]['y_pred_binary']
    Temp = dict_storage_join[key]  >> mask(colonne > 0)
    
    #Créer X_reg avec temp : c'est X_mod
    P.regression_lineaire(1, dict_storage_join) #self.metrics_lineaire ; self.Y_mod_pred ; self.a ; self.b

    df_metrics[key]={'metrics_lr':P.metrics_lr, 'metrics_lineaire':P.metrics_lineaire}
    df_Temp=pd.DataFrame({"gasDayStartedOn":Temp['gasDayStartedOn'],"y_mod_pred":P.Y_mod_pred})
    Temp=pd.merge(Temp,df_Temp, on='gasDayStartedOn',how='outer')
#Vous avez maintenant un vecteur y_pred_num : y_mod_pred
    #Temp = pd.concat((Temp['Date '],P.y_mod_pred), axis = 1)
    
    dict_storage_join[key]=pd.merge(dict_storage_join[key], Temp, on='gasDayStartedOn', how='outer')


N=len(dict_storage_join['SF - UGS Rehden']['y_mod_pred'].values)
sum=[0]*N
for key in dict_storage_join.keys():
    dict_storage_join[key]['y_mod_pred']=dict_storage_join[key]['y_mod_pred'].fillna(0)
    L=dict_storage_join[key]['y_mod_pred'].values
    N=len(L)
    for i in range (N):
        sum[i]=sum[i]+L[i]


supply=pd.DataFrame({"Date":dict_storage_join['SF - UGS Rehden']['gasDayStartedOn'],'offre':sum})

df_decision=pd.merge(supply,df_demand, on='Date',how='outer')
df_decision=df_decision.fillna(0)


df_decision['Decision']=df_decision['offre']
for k in range (N):
    O=df_decision['offre'].values
    D=df_decision['Demand'].values
    if O[k]<D[k]:
        df_decision['Decision'][k]='buy'
    elif (O[k]==D[k]):
        df_decision['Decision'][k]='flat'
    else:
        df_decision['Decision'][k]='sell'

test=[]
for i in range (N):
    if (df_decision['Decision'][i]=='sell'):
        test.append(i)
print(test)
        
        

    ##################################VRAIES VALEURS###########################################################


N=len(dict_storage_join['SF - UGS Rehden']['Net withdrawal_x' ].values)
sum_true=[0]*N
for key in dict_storage_join.keys():
    dict_storage_join[key]['Net withdrawal_x ']=dict_storage_join[key]['Net withdrawal_x'].fillna(0)
    L=dict_storage_join[key]['Net withdrawal_x'].values
    N=len(L)
    for i in range (N):
        sum_true[i]=sum_true[i]+L[i]


supply=pd.DataFrame({"Date":dict_storage_join['SF - UGS Rehden']['gasDayStartedOn'],'offre':sum_true})

df_decision_true=pd.merge(supply,df_demand, on='Date',how='outer')
df_decision_true=df_decision_true.fillna(0)


df_decision_true['Decision']=df_decision_true['offre']
for k in range (N):
    O=df_decision_true['offre'].values
    D=df_decision_true['Demand'].values
    if O[k]<D[k]:
        df_decision_true['Decision'][k]='buy'
    elif (O[k]==D[k]):
        df_decision_true['Decision'][k]='flat'
    else:
        df_decision_true['Decision'][k]='sell'

test2=[]
for i in range (N):
    if (df_decision_true['Decision'][i]=='sell'):
        test2.append(i)

    
proportion=0
n2=0
for i in range(N):
    predict=df_decision['Decision'].values
    if predict[i]!=0:
        n2=n2+1

for k in range (N):
    predict=df_decision['Decision'].values
    true=df_decision_true['Decision'].values
    if (predict[k]==true[k] and predict[k]!=0):
        proportion=proportion+1
        
proportion=proportion/n2
    
#######################TABLEAU DE METRICS##########
#initialisation du dictionnaire
"""
df_metrics={'SF - UGS Rehden':None, 'SF - UGS Kraak':None, 'SF - UGS Stassfurt':None, 'SF - UGS Harsefeld':None, 
            'SF - UGS Breitbrunn':None, 'SF - UGS Epe Uniper H-Gas':None, 'SF - UGS Eschenfelden':None,
            'SF - UGS Inzenham-West':None,'SF - UGS Bierwang':None, 'SF - UGS Jemgum H (EWE)':None, 
            'SF - UGS Peckensen':None, 'SF - UGS Etzel ESE (Uniper Ener':None}

for key in df_metrics:
    P=s.Prediction(dict_storage_join,key)
    P.logistic_reg()
    #P.regression_lineaire(1,dict_storage_join)
    df_metrics[key]={'metrics_lr':P.metrics_lr, 'metrics_lineaire':P.metrics_lineaire}
"""

    



