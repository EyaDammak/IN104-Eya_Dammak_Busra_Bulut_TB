# -*- coding: utf-8 -*-
"""
Created on Fri May 21 17:58:00 2021

@author: dondu
"""

"""
Created on Tue Apr 27 16:40:41 2021
@author: dondu
"""

import pandas as pd
import numpy as np
import os
from dfply import *
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#This function sets the working directory
def set_wd(wd='C:/cygwin64/home/busra/IN104/Model_Demand/De_corrige.csv'):
    os.chdir(wd)

#This function imports a csv file and has the option to plot its value columns as a function of the first column
def import_csv(f_name, delimeter, plot ):
    f = pd.read_csv(f_name, sep=delimeter )
    f=f.rename(columns ={'Date (CET)':'Date'})
    f['Date'] = pd.to_datetime(f['Date']) #on modifie le type de la colonne de date
    fig, axe = plt.subplots(3,1) #3lignes et 1 colonnes
    f.plot(x='Date', y='Actual', ax=axe[0], color='g')
    plt.title("La température actuelle en fonction des jours")
    f.plot(x='Date', y='Normal', ax=axe[1], color='r')
    plt.title("La température normale en fonction des jours")
    f.plot(x='Date', y='LDZ', ax=axe[2], color='b')
    plt.title("La consommation de gaz en fonction des jours")
    return f



#f=import_csv(f_name = "C:/cygwin64/home/busra/IN104/Model_Demand/DE_corrige.csv", delimeter = ";", plot = True)

#This function creates a scatter plot given a DataFrame and an x and y column
def scatter_plot(dataframe, x , y , col):
    plt.scatter(dataframe[x], dataframe[y], c=col)


#This function is the sigmoid function for gas consumption as a function of temperature
def h(t, a, b, c, d):
    return(d+a/(1+(b/(t-40))**c))



#The following function takes sigmoid parameters values and a temperature vector as input and plots the sigmoid, can compare it with the actual values
def consumption_sigmoid(dataframe,guess_value, plot):
    #dataframe=f, a = 900, b = -35, c = 6, d = 300, plot = True
    a=guess_value[0]
    b=guess_value[1]
    c=guess_value[2]
    d=guess_value[3]
    h_hat = []
    T=dataframe['Actual'].values
    for t in T:
        h_hat.append(h(t, a, b, c, d)) #je calcule la consommation pour chaque température

    if plot:
        dataframe.plot(x='Actual', y = 'LDZ', color = 'b') #la vraie courbe
        t_h_hat=np.array(h_hat)
        T_t=np.array(T)
        #je les met sous forat tableau pour pouvoir les dessiner
        plt.plot(T_t, t_h_hat, c='r') 
        plt.title("La consommation en fonction de la Température")
        xlabel=("La température")
        ylabel=("La consommation")
        plt.show()
    return(h_hat)

#try_1 = [500,-25,2,100]
#p=consumption_sigmoid(f, try_1, True)

#using Scipy pour determiner les paramètres optimaux du model
#optimiser_les_parametres(dataframe, guess_value,fonction):
#T= f['Actual'].values
#real_conso =f['LDZ'].values
#c, cov= curve_fit(h,T,real_conso,try_1)
#print(c)
#[798.90008115 -37.14129894   5.6214797  101.03853919]
#model=consumption_sigmoid(f, c, True)

#on teste avec 

#pour tester si problème avec la valeur interdite 40
def is_40(T):
    for i in T:
        if (i==40):
            print("il y a un 40")
            return 0 
        else:
            return 1


#pour mesurer si les valeurs fit bien
#on utilise les metric suivant 
#calculate goodness of fit parameters: correlation, root mean square error (RMSE), Average normalised RMSE, normalized RMSE
#averaged normalized RMSE is RMSE/(average value of real consumption)
#normalized RMSE is RMSE/(max value of real consumption - min value of real consumption)
#The following function gets the fit metrics list between 2 sigmoids
def get_fit_metrics(model, real_conso):
    if(len(model) != len(real_conso)):
        print("Difference in length between Fit and Real Consumption vectors")
    else:
        corr, _ = pearsonr(model, real_conso)
        #print('corrélation : ',corr)
        rmse = np.sqrt(mean_squared_error(model,real_conso))
        #print('rmse : ', rmse)
        nrmse = rmse/(np.max(real_conso) - np.min(real_conso))
        #print('nrmse : ', nrmse)
        anrmse = rmse/np.mean(real_conso)
        #print('anrmse : ',anrmse)
    return [corr,rmse,nrmse,anrmse]

#Any other metric we could use ?
#on calcule le coefficient de determination 
#update on ne peut pas calculer R^2 car ce n'est pas un model linéaire
#print('R^2 : ', r2_score(model,real_conso))

#The following class is the cosumption class it takes sigmoid parameters as well as a temperature as input
class Consumption:
    #Initialize class
    def __init__(self, g):
        self.a=g[0]
        self.b=g[1]
        self.c=g[2]
        self.d=g[3]
                

    #calculate the consumption given a temperature
    def get_consumption(self, temperature):
        self.temperature=temperature
        self.h=self.d+self.a/(1+(self.b/(self.temperature-40))**self.c)
        return(self.h)
        
        
    #T est la liste de température, variable gloable dans la classe
    T=np.linspace(-39,40,1000)
    #problème de rencontre car 40 est la valeur interdite
        
    #get the sigmoid considering a temperature between -40 and 39, use the function consumption_sigmoid above
    def sigmoid(self): 
            self.model_conso = []
            for t in Consumption.T:
                self.model_conso.append(h(t,self.a,self.b,self.c,self.d)) #je calcule la consommation pour chaque température
            
    #This is what the class print if you use the print function on it
    def __str__(self):
         #je met la consommation sous format tableau pour pouvoir les dessiner
        t_model_conso=np.array(self.model_conso)
        plt.plot(Consumption.T, t_model_conso, c='r') 
        plt.title("La consommation en fonction de la Température")
        xlabel=("La température")
        ylabel=("La consommation")
        plt.show()
        return("on a tracé la consommation en fonction de la Température" )

#The following class optimizes the parameters of the sigmoid and returns an object of class consumption
class Optimize_sigmoid:
    #Initialize guess values that are common to all instances of the clasee
    #je les définit sous format variable de la classe car, peu importe l'objet, on initialise avec les mêmes valeurs de paramètres
    guess_a = 500
    guess_b =-25
    guess_c =2
    guess_d=100

    def __init__(self, dataframe):
        self.__f=dataframe
        #on vérifie que f soit bien un type Dataframe
        if isinstance(self.__f, pd.DataFrame): 
            if 'Actual' and 'LDZ' in self.__f.columns:
                    self.__f=self.__f.rename(columns ={'Date (CET)':'Date'})
                    self.__f['Date'] = pd.to_datetime(self.__f['Date']) #on modifie le type de la colonne de date
                    self.Date=self.__f['Date']
            else:
                print("Class not initialized since f does not contain Actual and LDZ column names")
        else:
            print("Class not initialized since f is not a DataFrame")

    #optimize and return metrics use functions h, consumption_sigmoid defined above as well as get_fit_metrics
    def optimize(self):
        if self.__f is not None:
            self.T= self.__f['Actual'].values
            real_conso =self.__f['LDZ'].values
            guess_values=[Optimize_sigmoid.guess_a,Optimize_sigmoid.guess_b,Optimize_sigmoid.guess_c, Optimize_sigmoid.guess_d]
            #on trouve les meilleurs parametres
            self.coef, self.__cov = curve_fit(h, self.T, real_conso, guess_values) 
            #on construit le modele de consommation avec les paramètres optimaux
            self.model_ajust_conso = consumption_sigmoid(self.__f,self.coef, True)
            #on calcule les métrics pour s'assurer de la proximité entre le model et la vrai courbe de consommation
            self.__corr, self.__rmse, self.__nrmse, self.__anrmse = get_fit_metrics( self.model_ajust_conso, real_conso)
        else:
            print("Class not initialized")

    #this function returns the fit metrics calculated above
    def fit_metrics(self):
        if self.__f is not None:
            return f"la corrélation vaut {self.__corr}, the root mean square error : {self.__rmse}, the normalized root mean square error :{self.__nrmse}, the average normalized rmse :{self.__anrmse}"
           #return(self.__corr, self.__rmse, self.__nrmse, self.__anrmse)
        else:
            print("optimize method is not yet run")
            

    #This function creates the class consumption
    def create_Consumption(self):
        if self.__f is not None:
            #je définit la classe consommation avec les paramètres optimaux
            h=Consumption(self.coef)
            return h
        else:
            print("optimize method is not yet run")

    #This is what the class print if you use the print function on it
    def __str__(self):
        if self.__f is not None:
            self.__f.plot(x='Actual', y = 'LDZ', color = 'b') #la vraie courbe
            t_model = np.array(self.model_ajust_conso ) 
            T_t=np.array(self.T)
            #je les met sous format tableau pour pouvoir les dessiner
            plt.plot(T_t, t_model, c='r') 
            plt.title("La consommation en fonction de la Température")
            xlabel=("La température")
            ylabel=("La consommation")
            plt.show()
            return("on a tracé la consommation en fonction de la Température")
            
        else:
            return("optimize method is not yet run")
        

#If you have filled correctly the following code will run without an issue        
'''if __name__ == '__main__':

    #set working directory
    #set_wd('C:/cygwin64/home/busra/IN104/Model_Demand/DE_corrige.csv')

    #1) import consumption data and plot it
    f = import_csv("C:/cygwin64/home/busra/IN104/Model_Demand/DE_corrige.csv",  ";", True)

    #2) work on consumption data (non-linear regression)
    #2)1. Plot consumption as a function of temperature    

    scatter_plot(f,"Actual", y = "LDZ", col = "red")        

    #2)2. optimize the parameters
    sig = Optimize_sigmoid(f)
    sig.optimize()
    c = sig.create_Consumption()
    print(sig)


    #2)3. check the new fit

    # These are the 3 ways to access a protected attribute, it works the same for a protected method
    # An attribute/method is protected when it starts with 2 underscores "__"
    # Protection is good to not falsy create change
     
    print(sig.fit_metrics())
    c.sigmoid()
    print(c)
'''
    