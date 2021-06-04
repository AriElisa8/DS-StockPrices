# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:15:17 2020

DATA SCIENCE COURSE FINAL PROJECT: STOCK PRICES PREDICTION MODEL 

@author: Arianna Hernandez

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
import joblib

"""
PHASE 1: DATASET UPLOAD AND CLEANING
"""
#Dataset upload
dataframe=pd.read_csv("StockPrices.csv")

#First column rename
dataframe=dataframe.rename(columns={"Attributes": "Date"})

#Auxiliar lists to clean de columns names
columnName=list(dataframe.columns) #First row - first part of name
columnName2=list(dataframe.iloc[0,:]) #Second row - second part of column name

#For loop to concat column names
new_name=[]       
[new_name.append(element +" " +element2) for element,element2 in zip(columnName, columnName2)]
new_name[0]="Date"

#Dictionary to join former name with new name, followed by collumn rename
dic_cols=dict(zip(columnName,new_name))
dataframe=dataframe.rename(columns=(dic_cols))

dataframe=dataframe.iloc[2:,:] # modified dataframe to start from third row
dataframe.Date=pd.to_datetime(dataframe.Date,utc=False) # change date column format to datetime

#Aux variables cleaning...
del(columnName)
del(columnName2)
del(dic_cols)
del(new_name)


"""
PHASE 2: DATA VISUAL EXPLORATION
"""

#Study of adjusted closing prices for APPLE, General Electric, Google, IBM and Microsoft

adjClosing=dataframe.iloc[:,:6]

# Moving Average with a 100 days window for each company...

adjClosing["Moving Average Apple"]=adjClosing["Adj Close AAPL"].rolling(100).mean()
adjClosing["Moving Average GOOGLE"]=adjClosing["Adj Close.2 GOOG"].rolling(100).mean()
adjClosing["Moving Average GE"]=adjClosing["Adj Close.1 GE"].rolling(100).mean()
adjClosing["Moving Average IBM"]=adjClosing["Adj Close.3 IBM"].rolling(100).mean()
adjClosing["Moving Average MSFT"]=adjClosing["Adj Close.4 MSFT"].rolling(100).mean()


#Moving Average Graph for each Company.
def MovingAverageGraphic(date, movingAverage, dataset, movingAverageColor, title, xlabel ,  ylabel):
        fig= plt.figure(figsize=(12,6))
        fig=sns.lineplot(x=date, y=movingAverage, data=dataset, color=movingAverageColor)
        fig=plt.title(title, fontsize=22, fontstyle='normal',color=movingAverageColor )
        fig=plt.xlabel(xlabel, fontsize=18)
        fig=plt.ylabel(ylabel,fontsize=18)
        return plt.show()
            
#APPLE
MovingAverageGraphic('Date', 'Moving Average Apple', adjClosing, "darkcyan", "APPLE Closing Price Moving Average", 'Year' , "Adjusted Closing Price")

#GENERAL ELECTRIC
MovingAverageGraphic('Date', 'Moving Average GE', adjClosing,"darkslategrey", "GE Closing Price Moving Average", 'Year' , "Adjusted Closing Price")

#GOOGLE
MovingAverageGraphic('Date', 'Moving Average GOOGLE', adjClosing,"darkred", "GOOGLE Closing Price Moving Average", 'Year' , "Adjusted Closing Price")

#IBM
MovingAverageGraphic('Date', 'Moving Average IBM', adjClosing,"darkslategrey", "IBM Closing Price Moving Average", 'Year' , "Adjusted Closing Price")

#MICROSOFT
MovingAverageGraphic('Date', 'Moving Average MSFT', adjClosing,"darkcyan", "MICROSOFT Closing Price Moving Average", 'Year' , "Adjusted Closing Price")


"""
PHASE 2.1: DATA EXPLORATORY ANALYSIS
"""
#PARTE 2
def stockReturn(dataframe,companyClosing,companyReturn):
    
    companyData=dataframe.loc[:,["Date",companyClosing]]
    companyData["day_shifted"]=companyData[companyClosing].shift(1)
    companyData=companyData.astype({companyClosing: float, "day_shifted":float})
    companyData[companyReturn]=companyData[companyClosing]/companyData["day_shifted"] - 1
    del(companyData["day_shifted"])
    del(companyData[companyClosing])
    return companyData

#Return calculation for each company
data_apple= stockReturn(dataframe,"Close AAPL", "Retorno AAPL")
data_GE= stockReturn(dataframe,"Close.1 GE", "Retorno GE")    
data_GOOG= stockReturn(dataframe,'Close.2 GOOG', "Retorno GOOG")   
data_IBM= stockReturn(dataframe,'Close.3 IBM', "Retorno IBM")   
data_MSFT= stockReturn(dataframe,'Close.4 MSFT', "Retorno MSFT")   

#Returns merged into one dataframe
returnsDataframe=pd.merge(data_apple,data_GE, on="Date")
returnsDataframe=pd.merge(returnsDataframe,data_GOOG, on="Date")
returnsDataframe=pd.merge(returnsDataframe,data_IBM, on="Date")
returnsDataframe=pd.merge(returnsDataframe,data_MSFT, on="Date")

#Selection of Moving Average Columns from adjClosing df.
adjClosing=adjClosing.loc[:,['Date',"Moving Average Apple","Moving Average GE","Moving Average GOOGLE", "Moving Average IBM", "Moving Average MSFT"]]
# M_Avg and return merge
dataframe=pd.merge(dataframe,adjClosing, on='Date')
dataframe=pd.merge(dataframe,returnsDataframe, on='Date')

#Final df to csv file
dataframe.to_csv('Companies_MovingAverage_and_Returns.csv')

del(adjClosing)
del(data_apple)
del(data_GE)
del(data_GOOG)
del(data_IBM)
del(data_MSFT)

#Return Graphs


def returnGraph(data,xcol, ycol, title,xlabel,ylabel ,color):
    fig6 = plt.figure(figsize=(12,6))
    fig6=sns.lineplot(x=xcol, y=ycol, data=data, color=color)
    fig6=plt.title(title, fontsize=22, fontstyle='normal',color=color)
    fig6=plt.xlabel(xlabel, fontsize=18)
    fig6=plt.ylabel(ylabel,fontsize=18)
    return plt.show()
   

    
#APPLE
returnGraph(returnsDataframe,'Date', "Retorno AAPL", "APPLE Returns (2006 - 2020)",'Year',"Returns", "darkcyan")
#GENERAL ELECTRIC
returnGraph(returnsDataframe,'Date', "Retorno GE", "GENERAL ELECTRIC Returns (2006 - 2020)",'Year',"Returns", "darkslategrey")
#GOOGLE
returnGraph(returnsDataframe,'Date',"Retorno GOOG", "GOOGLE Returns (2006 - 2020)",'Year',"Returns", "darkred")
#IBM
returnGraph(returnsDataframe,'Date',"Retorno IBM", "IBM Returns (2006 - 2020)",'Year',"Returns", "darkslategrey")
#MICROSOFT
returnGraph(returnsDataframe,'Date',"Retorno MSFT", "MICROSOFT Returns (2006 - 2020)",'Year',"Returns","darkcyan")


#Color Heat Map of returns
corr_matrix=returnsDataframe.corr()
fig11=plt.figure(figsize=(8,8))
fig11=sns.heatmap(corr_matrix, cmap='icefire')
plt.show()


#Returns Pairplot
fig12=sns.pairplot(returnsDataframe.iloc[:,1:])
del(returnsDataframe)
del(corr_matrix)




"""
PHASE 3: SUPPORT VECTOR MACHINE REGRESSION MODEL

COMPANY: GOOGLE
"""
#google dataframe from original dataframe
googleDf=dataframe.loc[:,['Date','Adj Close.2 GOOG','Volume.2 GOOG','High.2 GOOG',
                           'Low.2 GOOG','Open.2 GOOG','Close.2 GOOG'] ]
googleDf.iloc[:,1:]=googleDf.iloc[:,1:].astype(float)

# Dayly index and percentage change for google stock prices...

googleDf["daylyIndex"]=(googleDf['High.2 GOOG']-googleDf['Low.2 GOOG'])/googleDf['Close.2 GOOG']

googleDf["percentage_change"]=(googleDf['Close.2 GOOG'] - googleDf['Open.2 GOOG'])/googleDf['Open.2 GOOG']

#last five days for  final comprobation after trainning.
last5_df=googleDf.iloc[-5:,:]


#final Google DataFrame for trainning...
googleDf=googleDf.iloc[:-5,:]
googleDf.to_csv('Google_dataframe.csv')
googleDf.Date=pd.to_datetime(googleDf.Date)

#Parameters selected to train the model...
#the idea is to use volume, daily index and percentage change of stocks to predict de adjusted closing price.
parameters=googleDf.loc[:,['Date','Volume.2 GOOG', 'daylyIndex','percentage_change']].values
target=googleDf.loc[:,'Adj Close.2 GOOG'].values.reshape(-1,1)

#last five days Parameters selected for  final comprobation after trainning.
last5_parameters=last5_df.loc[:,['Date','Volume.2 GOOG', 'daylyIndex','percentage_change']].values
last5_target=last5_df.loc[:,'Adj Close.2 GOOG'].values.reshape(-1,1)



#Parameters Standardization
sc = StandardScaler()
parameters[:,1:]= sc.fit_transform(parameters[:,1:])

sctarget=StandardScaler()
target=sctarget.fit_transform(target.reshape(-1,1))
last5_parameters[:,1:]= sc.fit_transform(last5_parameters[:,1:])
last5_target=sctarget.fit_transform(last5_target)


#ENTRENAMIENTO: Encontrando el mejor modelo
# Creamos rejilla de posibles parametros
parametros_rejilla = {'kernel':('rbf','poly'), 'C': [1e5], 'epsilon': [100, 10, 1]}

# Creamos el buscador para los parametros optimos
regresor_rejilla = GridSearchCV(SVR(), parametros_rejilla,n_jobs=-1, verbose=1)

# Entrenamos 
regresor_rejilla.fit(parameters[:,1:], target.ravel())

# Mejores parámetros
print(regresor_rejilla.best_params_)
print(regresor_rejilla.best_estimator_)

#Prediccion target
target_pred = regresor_rejilla.predict(parameters[:,1:])

#Devolvemos a los valores iniciales para graficar
target=sctarget.inverse_transform(target)
target_pred=sctarget.inverse_transform(target_pred)


#Grafica de Modelos
sns.lineplot(parameters[:,0], target[:,0], color = "darkred")
sns.lineplot(parameters[:,0],target_pred, color = 'teal')
plt.suptitle('Prediccion Precio de Cierre Google (2006-2020)')
plt.xlabel('Year')
plt.ylabel('Adjusted Closing Price')
plt.show() 


#PREDICCION ULTIMOS 5 DIAS

last5_target_pred= regresor_rejilla.predict(last5_parameters_sinfecha)
#devolvemos a valores originales para graficar.
last5_target_pred=sc.inverse_transform(last5_target_pred)
last5_target.inverse_transform(last5_target)

fig15=plt.figure(figsize=(12,6))
sns.lineplot(last5_parameters[:,0],last5_target_pred, color = 'teal', ls='--')
sns.lineplot(x=last5_parameters[:,0],y=last5_target.ravel(), color = "darkred")
plt.suptitle('Prediccion Precio de Cierre Google Noviembre 2020)')
plt.title('Ultimos 5 dias')
plt.xlabel('Year')
plt.ylabel('Adjusted Closing Price')
plt.xticks(rotation=70)
plt.show() 


#Metricas
def metricas(y_true, y_pred):

    mae=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 

    print('Error Medio Absoluto (MAE): ', round(mae,4))
    print('Error Medio Cuadratico (MSE):: ', round(np.sqrt(mse),4))

print('Metricas Data Entrenamiento')
metrica_modelo=metricas(target,target_pred)
print('Metricas ultimos 5 días de Noviembre')
metrica_ultimos_dias=metricas(last5_target,last5_target_pred)


#guardamos el modelo y las predicciones


joblib.dump(regresor_rejilla, 'regresionSVR.pkl')

parameters=pd.DataFrame(parameters,index=list(range(len(parameters))),columns=['Date','Volume.2 GOOG', 'daylyIndex','percentage_change'] )
last5_parameters=pd.DataFrame(last5_5, index=list(range(len(last5_parameters), len(last5_parameters)+5)),columns=['Date','Volume.2 GOOG', 'daylyIndex','percentage_change']  )

parametersGOOGLE=parameters.append(last5_parameters)


target_pred=pd.DataFrame(target.pred.reshape(-1,1), index=list(range(len(target_pred))), columns='Precio de Cierre Predicho')
last5_target_pred=pd.DataFrame(target.pred.reshape(-1,1), index=list(range(len(last5_target_pred),len(last5_target_pred)+5)), columns='Precio de Cierre Predicho')

targetGOOGLE=target.append(last5_target_pred)

parametersGOOGLE['Precio de Cierre Predicho']=targetGOOGLE

parametersGOOGLE.to_csv('Dataframe_de_prediccion.csv')




"""
ESTE ES EL MODELO CON EL C MAS ALTO DEL QUE OBTUVE UN RESULTADO
"""

#Entrenamos el Modelo
regresorSVR = SVR( kernel="rbf", C=1e4, epsilon=0.05)
regresorSVR.fit(parameters[:,1:], target.ravel())

target_pred= regresorSVR.predict(parameters[:,1:])

#Devolvemos a los valores iniciales para graficar
target=sctarget.inverse_transform(target)
target_pred=sctarget.inverse_transform(target_pred)


#Grafica de Modelos
import matplotlib.patches as mpatches
fig14=plt.figure(figsize=(12,6))
sns.lineplot(parameters[:,0],target_pred, color = 'teal', ls='--')
sns.lineplot(parameters[:,0], target.ravel(), color = "darkred")
plt.suptitle('Prediccion Precio de Cierre Google (2006-2020)')
plt.xlabel('Year')
plt.ylabel('Adjusted Closing Price')

real=mpatches.Patch(color='darkred', label='Data Real')
pred=mpatches.Patch(color='teal', label='Predicción')
plt.legend(handles=[real,pred])
plt.show() 




#PREDICCION ULTIMOS 5 DIAS

last5_target_pred= regresorSVR.predict(last5_parameters[:,1:])
#devolvemos a valores originales para graficar.
last5_target_pred=sctarget.inverse_transform(last5_target_pred.ravel())
last5_target=sctarget.inverse_transform(last5_target.reshape(-1,1))


fig15=plt.figure(figsize=(12,6))
sns.lineplot(last5_parameters[:,0],last5_target_pred.ravel(), color = 'teal', ls='--')
sns.lineplot(last5_parameters[:,0],last5_target.ravel(), color = "darkred")
plt.suptitle('Prediccion Precio de Cierre Google')
plt.title('16 al 20 Novimebre 2020')
plt.xlabel('Día')
plt.ylabel('Adjusted Closing Price')
plt.xticks(rotation=70)
real=mpatches.Patch(color='darkred', label='Data Real')
pred=mpatches.Patch(color='teal', label='Predicción')
plt.legend(handles=[real,pred])
plt.show()  

 

#Metricas
def metricas(y_true, y_pred):

    mae=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 

    print('Error Medio Absoluto (MAE): ', round(mae,4))
    print('Error Medio Cuadratico (MSE):: ', round(np.sqrt(mse),4))

print('Metricas Data Entrenamiento')
metrica_modelo=metricas(target,target_pred)
print('Metricas ultimos 16 al 20 de Noviembre 2020')
metrica_ultimos_dias=metricas(last5_target,last5_target_pred)


#guardamos el modelo y las predicciones



joblib.dump(regresorSVR, 'regresionSVR005error.pkl')

parameters=pd.DataFrame(parameters,index=list(range(len(parameters))),columns=['Date','Volume.2 GOOG', 'daylyIndex','percentage_change'] )
last5_parameters=pd.DataFrame(last5_parameters, index=list(range(len(parameters),len(parameters)+5)),columns=['Date','Volume.2 GOOG', 'daylyIndex','percentage_change'])

parametersGOOGLE=parameters.append(last5_parameters)


target_pred=pd.DataFrame(target_pred, index=list(range(len(target_pred))), columns=['Precio de Cierre Predicho'])
last5_target_pred=pd.DataFrame(last5_target_pred, index=list(range(len(target_pred),len(target_pred)+5 )), columns=['Precio de Cierre Predicho'])

targetGOOGLE=target_pred.append(last5_target_pred)

parametersGOOGLE['Precio de Cierre Predicho']=targetGOOGLE

parametersGOOGLE.to_csv('Dataframe_de_prediccion.csv')






