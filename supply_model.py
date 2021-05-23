import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import os
import dfply 
from dfply import * 
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression



def preparer_le_dict_à_la_prediction(dict_storage,df_price):
###création de colonnes  
            #on vérifie que f soit bien un type Dataframe
        if isinstance(dict_storage, dict):
            for key in dict_storage: #on parcourt les clés
                if isinstance(dict_storage[key],pd.DataFrame):
                    if 'withdrawal' and 'injection' in dict_storage[key]:
                        dict_storage[key]=dict_storage[key].fillna(0);
                        dict_storage[key]['Net withdrawal'] = dict_storage[key]['withdrawal'] - dict_storage[key]['injection']
                        L=dict_storage[key]['Net withdrawal'].values
                        dict_storage[key]['Lagged_NW']=dict_storage[key]['withdrawal'] - dict_storage[key]['injection']
                        N=len(dict_storage[key]['Lagged_NW']) #nombre de lignes
                        #on créé la colonne décalée
                        dict_storage[key]['Lagged_NW'][0]=0
                        for i in range(N):
                            dict_storage[key]['Lagged_NW'][i] = L[i-1]
                            #on créé la colonne randomly
                            dict_storage[key]['Net withdrawal binary']= dict_storage[key]['withdrawal']
                        for i in range(N):
                            if(L[i]>0):
                                dict_storage[key]['Net withdrawal binary'][i]=1
                            else:
                                dict_storage[key]['Net withdrawal binary'][i]=0
                        liste=dict_storage[key]['full'].values
                        for i in range (N):
                            if liste[i]>45:
                                dict_storage[key]['FSW1']=liste[i]-45
                                dict_storage[key]['FSW2']=0
                            else:
                                dict_storage[key]['FSW1']=0
                                dict_storage[key]['FSW2']=45-liste[i]
                    else:
                        print("Le dataframe n'a pas les bonnes colonnes")
                        return()
                else:
                    print("Au moins un des Dataframes du dictionnaire not initialized since dict_storage[key] is not a Dataframe")
                    return()
        else:
            print("le dictionnaire not initialized since dict_storage is not a Dict")
            return()
            #on joint les dictionnaires
        df_price=df_price.rename(columns={'Date':'gasDayStartedOn'})
        for key in dict_storage:
            dict_storage[key] = pd.merge(df_price, dict_storage[key], on='gasDayStartedOn')
        return(dict_storage)  

class Prediction:
    #cette classe permet, à partir du dictionnaire préparé, d'afficher les résultats de la prédiction ; en prenant la méthode Logistic Regression par défaut
    
    def __init__(self,dic,storage):
        self.dict_storage_join=dic
        self.key=storage
        
        
    def logistic_reg(self):
        self.metrics_lr={}
       
        X=self.dict_storage_join[self.key].iloc[:,[1,2,3,4,16,18,19]].values
        y=self.dict_storage_join[self.key].iloc[:,17].values
        #mise à l'échelle
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
        
        lr=LogisticRegression(random_state=0,solver='liblinear')
        lr.fit(X_train, y_train)
        print(lr.coef_)
        print(lr.intercept_)
        
        #on rajoute cette ligne pour la partie Balance
        self.y_pred_train=lr.predict(X_train)
        
        self.y_pred = lr.predict(X_test)
        cm=confusion_matrix(y_test, self.y_pred)
        probs=lr.predict_proba(X_test)[:,1] 
        self.metrics_lr[self.key]={'recall': metrics.recall_score(y_test, self.y_pred), 'neg_recall': cm[1,1]/(cm[0,1] + cm[1,1]), 'confusion': cm, 'precision': metrics.precision_score(y_test, self.y_pred), 'neg_precision':cm[1,1]/cm.sum(axis=1)[1], 'roc': metrics.roc_auc_score(y_test, probs),'class_mod': lr}
        return(self.y_pred)
    
    
    
    def random_forest(self):
        self.metrics_rf={}
        
        X=self.dict_storage_join[self.key].iloc[:,[1,2,3,4,16,18,19]].values
        y=self.dict_storage_join[self.key].iloc[:,17].values
        #mise à l'échelle
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
            
        
        rf = RandomForestClassifier(n_estimators=100, 
                                       bootstrap = True,
                                       max_features = 'sqrt')
        # Fit on training data
        rf.fit(X_train, y_train)
        # Actual class predictions
        self.rf_predictions = rf.predict(X_test)
        # Probabilities for each class
        rf_probs = rf.predict_proba(X_test)[:, 1]
        
        # Calculate roc auc
        roc_value = roc_auc_score(y_test, rf_probs)
        cm=confusion_matrix(y_test, y_pred)
        
        self.metrics_rf[self.key]={'recall': metrics.recall_score(y_test, y_pred), 'neg_recall': cm[1,1]/(cm[0,1] + cm[1,1]), 'confusion': cm, 'precision': metrics.precision_score(y_test, y_pred), 'neg_precision':cm[1,1]/cm.sum(axis=1)[1], 'roc': metrics.roc_auc_score(y_test, rf_probs),'class_mod': rf}
        return(self.rf_predictions)
    

    

    def comparaison(self):
        roc_lr=self.metrics_lr[self.key]['roc']
        roc_rf=self.metrics_rf[self.key]['roc']
       
        if roc_lr>roc_rf: #retourne 0 : si la méthode de random forest est meilleure
            print('la meilleure méthode est random forest')
            self.choix=0
            return 0
        else: #retourne 1 : si la méthode de LR est meilleure
            print('la meilleure méthode est logistic regression')
            self.choix=1
            return 1
        
        
        
    
    def regression_lineaire(self,choix_modele):  #choix_modele=0 si on est dans le supply-model 
        self.metrics_lineaire={}
        L=[]
        if choix_modele==0: #nopus sommes dans le supply_modele          
            NWB=self.dict_storage_join[self.key]['Net withdrawal binary']
        else:
            NWB=self.y_pred_train+self.y_pred  #vecteur de y_pred complet
        f = self.dict_storage_join[self.key] >> mask(NWB > 0)
        print(f)
        #print(dict_storage_join[self.key])
            
        #X et Y "mod"ifiés car absence de lignes dans la dataframe
        X_mod=f.iloc[:,[1,2,3,4,16,18,19]].values
            #on prend le NW , ici le but ets de prédire la quantité qu'on doit soutirer c'est pour ça aussi qu'on a prit que 1, c'est parceque l'offre demande que ça
        Y_mod=f.iloc[:,15].values
        rg=LinearRegression()
        X_train,X_test,Y_train,Y_test=train_test_split(X_mod,Y_mod,random_state=1)
        rg.fit(X_train,Y_train)
        
        if choix_modele==0:#nopus sommes dans le supply_modele     
            self.Y_mod_pred=rg.predict(X_test)
        else:
            self.Y_mod_pred=rg.predict(X_mod) 
        rmse = np.sqrt(mean_squared_error(self.Y_mod_pred,Y_test))
        nrmse = rmse/(np.max(Y_test) - np.min(Y_test))
        anrmse = rmse/np.mean(Y_test)
        corr=r2_score(self.Y_mod_pred,Y_test)
        self.metrics_lineaire[self.key] = {'r2': metrics.r2_score(Y_test, self.Y_mod_pred), 'rmse': rmse, 'nrmse': nrmse, 'anrmse': anrmse, 'cor': corr, 'l_reg': rg}
        self.a=rg.coef_
        self.b=rg.intercept_
        return self.metrics_lineaire
    

        
        
        
    #def __str__(self):
        
#if __name__ == '__main__':
    #dict_storage=pd.read_excel('storage_data.xlsx',sheet_name=None)
    #df_price=pd.read_excel('price_data.xlsx')
    #dict_storage_join=preparer_le_dict_à_la_prediction(dict_storage,df_price)
    #prediction=Prediction(dict_storage_join)
    #prediction.
    