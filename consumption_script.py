import pandas as pd
import numpy as np
import os
from dfply import * 
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#This function sets the working directory
def set_wd(wd='C:/Users/Hichem/Documents/IN104/projet/DE.csv'):
    os.chdir(wd)

#This function imports a csv file and has the option to plot its value columns as a function of the first column
def import_csv(f_name="C:/Users/Hichem/Documents/IN104/projet/DE.csv", delimeter=";", plot=True):
    f=pd.read_csv(f_name,sep=delimeter)
    f=f.rename(columns={'Date (CET)':'Date'})
    f['Date']=pd.to_datetime(f['Date'])
    fig,axes=plt.subplots(3,1)
    f.plot(x='Date',y='Normal',ax=axes[0],color='r')
    f.plot(x='Date',y='Actual',ax=axes[1],color='g')
    f.plot(x='Date',y='LDZ',ax=axes[2],color='b')
    plt.title("Consommation de gaz en fonction des jours")
    plt.show()
    return f 

f=import_csv(f_name="C:/Users/Hichem/Documents/IN104/projet/DE.csv", delimeter=";", plot=True) 



def h(t, a, b, c, d):
    return(d+a/(1+(b/(t-40))**c))




#This function creates a scatter plot given a DataFrame and an x and y column
def scatter_plot(dataframe = f, x = "Actual", y = "LDZ", col = "red"):
    plt.scatter(dataframe[x],dataframe[y],c=col)
    
    
#This function is the sigmoid function for gas consumption as a function of temperature
def consumption_sigmoid(dataframe,guess_value, plot):
    #dataframe=f, a = 900, b = -35, c = 6, d = 300, plot = True
    a=guess_value[0]
    b=guess_value[1]
    c=guess_value[2]
    d=guess_value[3]
    h_hat = []
    T=dataframe['Actual'].values #liste de températures
    for t in T:
        h_hat.append(h(t, a, b, c, d)) #je calcule la consommation pour chaque température

    if plot:
        dataframe.plot(x='Actual', y = 'LDZ', color = 'b') #la vraie courbe
        t_h_hat=np.array(h_hat)
        T_t=np.array(T)
        #je les met sous format tableau pour pouvoir les dessiner
        plt.plot(T_t, t_h_hat, c='r')
        plt.title("La consommation en fonction de la Température")
        xlabel=("La température")
        ylabel=("La consommation")
        plt.show()
        #if real_conso is not None you plot it as well
        if not isinstance(f['LDZ'], type(None)):
            if(len(f['Actual']) != len(f['LDZ'])):
                print("Difference in length between Temperature and Real Consumption vectors")
            # add title and legend and show plot
    return(h_hat)

try_1 = [798.90008115, -37.14129894 ,  5.6214797,  101.03853919]
p=consumption_sigmoid(f, try_1, True)

#using Scipy pour determiner les paramètres optimaux du model
#optimiser_les_parametres(dataframe, guess_value,fonction):
T= f['Actual'].values
#print(T)
C =f['LDZ'].values
#print(C)
c, cov= curve_fit(h,T,C,try_1)
print(c)

#on teste avec

#pour tester si problème avec la valeur interdite 40


'''
def is_40(T):
    for i in T:
        if (i==40):
            print("il y a un 40")
            return 0
        else:
            return 1
'''

model=consumption_sigmoid(f, c, True)
real_conso=C
from sklearn.metrics import r2_score
print('R^2:',r2_score(model,real_conso))

rmse = np.sqrt(mean_squared_error(model,real_conso))
print('RMSE :',rmse)
nrmse = rmse/(np.max(real_conso) - np.min(real_conso))
print('NRMSE :',nrmse)
anrmse = rmse/np.mean(real_conso)
print('ANRMSE :',anrmse)

'''
#The following function gets the fit metrics list between 2 sigmoids
def get_fit_metrics(h_hat, real_conso):
    if(len(h_hat) != len(real_conso)):
        print("Difference in length between Fit and Real Consumption vectors")
    else:
        
        
    return []

#The following class is the cosumption class it takes sigmoid parameters as well as a temperature as input
class consumption:
    #Initialize class
    def __init__(self, a, b, c, d):
        

    #calculate the consumption given a temperature
    def get_consumption(self, temperature):
        

    #get the sigmoid considering a temperature between -40 and 39, use the function consumption_sigmoid above
    def sigmoid(self, p):

    #This is what the class print if you use the print function on it
    def __str__(self):
        
        return t

#The following class optimizes the parameters of the sigmoid and returns an object of class consumption
class optimize_sigmoid:
    #Initialize guess values that are common to all instances of the clasee
    __guess_a, __guess_b, __guess_c, __guess_d

    def __init__(self, f):
        if isinstance(f, pd.DataFrame):
            if 'Actual' and 'LDZ' in f.columns:
                
            else:
                print("Class not initialized since f does not contain Actual and LDZ column names")
        else:
            print("Class not initialized since f is not a DataFrame")

    #optimize and return metrics use functions h, consumption_sigmoid defined above as well as get_fit_metrics
    def optimize(self):
        if self.__f is not None:
            self.__coef, self.__cov = curve_fit(
                h,

                )
            
            s = consumption_sigmoid(

                plot = True
                

                )
            
            self.__corr, self.__rmse, self.__nrmse, self.__anrmse = get_fit_metrics(s, self.__f['LDZ'])
        else:
            print("Class not initialized")

    #this function returns the fit metrics calculated above
    def fit_metrics(self):
        if  is not None:
            return 
        else:
            print("optimize method is not yet run")

    #This function creates the class consumption
    def create_consumption(self):
        if  is not None:
            return 
        else:
            print("optimize method is not yet run")

    #This is what the class print if you use the print function on it
    def __str__(self):
        if is not None:

        else:
            t = "optimize method is not yet run"
        return t

#If you have filled correctly the following code will run without an issue        
if __name__ == '__main__':

    #set working directory
    set_wd()

    #1) import consumption data and plot it
    conso = import_csv()

    #2) work on consumption data (non-linear regression)
    #2)1. Plot consumption as a function of temperature    

    scatter_plot()        

    #2)2. optimize the parameters
    sig = optimize_sigmoid(conso)
    sig.optimize()
    c = sig.create_consumption()
    print(sig)


    #2)3. check the new fit

    # These are the 3 ways to access a protected attribute, it works the same for a protected method
    # An attribute/method is protected when it starts with 2 underscores "__"
    # Protection is good to not falsy create change
    
    print(
            [
            sig.__dict__['_optimize_sigmoid__corr'],
            sig.__dict__['_optimize_sigmoid__rmse'],
            sig.__dict__['_optimize_sigmoid__nrmse'],
            sig.__dict__['_optimize_sigmoid__anrmse']
            ]
        )

    print(
            [
            sig._optimize_sigmoid__corr,
            sig._optimize_sigmoid__rmse,
            sig._optimize_sigmoid__nrmse,
            sig._optimize_sigmoid__anrmse
            ]
        )

    print(
            [
            getattr(sig, "_optimize_sigmoid__corr"),
            getattr(sig, "_optimize_sigmoid__rmse"),
            getattr(sig, "_optimize_sigmoid__nrmse"),
            getattr(sig, "_optimize_sigmoid__anrmse")
            ]
        )
    
    print(sig.fit_metrics())
    c.sigmoid(True)
    print(c)
    
    #3) If time allows do TSA on actual temperature
    #3)1. Check trend (and Remove it)
    #3)2. Check Seasonality (Normal Temperature)
    #3)3. Model stochastic part that is left with ARIMA
    #3)4. Use this to forecast consumption over N days
'''
