# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:15:17 2020

PROYECTO: FINANZAS

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

#cargamos dataset, cambiamos el nombre de la primera columna
dataframe=pd.read_csv("stocks.csv")
dataframe=dataframe.rename(columns={"Attributes": "Date"})

#creamos listas de los numbres de las columnas
columnas=list(dataframe.columns)
#creo lista de la segunda fila del dataset
columnas2=list(dataframe.iloc[0,:])
columnas2[0]=""

#Hacemos loop para unir nombre de columnas con fila 1 en un solo nombre.
nueva_col=[]       
[nueva_col.append(element +" " +element2) for element,element2 in zip(columnas, columnas2)]
nueva_col[0]="Date" #reajustamos el nombre de esta columna para que no quede con espacio al final

#creo un diccionario para asociar el nombre de las columnas con el nuevo nombre
dic_cols=dict(zip(columnas,nueva_col))
#renombro las columnas
dataframe=dataframe.rename(columns=(dic_cols))

#elimino las dos primeras filas y cambio la columna Date a formato Datetime
dataframe=dataframe.iloc[2:,:]
dataframe.Date=pd.to_datetime(dataframe.Date,utc=False)

del(columnas)
del(columnas2)
del(dic_cols)
del(nueva_col)

#1 Dataset de los Precios de Cierre Ajustado por empresa

cierre_ajustado=dataframe.iloc[:,:6]

# Calculo de media movil con una ventana de 100 dias..

cierre_ajustado["Media Movil Apple"]=cierre_ajustado["Adj Close AAPL"].rolling(100).mean()
cierre_ajustado["Media Movil GE"]=cierre_ajustado["Adj Close.1 GE"].rolling(100).mean()
cierre_ajustado["Media Movil GOOGLE"]=cierre_ajustado["Adj Close.2 GOOG"].rolling(100).mean()
cierre_ajustado["Media Movil IBM"]=cierre_ajustado["Adj Close.3 IBM"].rolling(100).mean()
cierre_ajustado["Media Movil MSFT"]=cierre_ajustado["Adj Close.4 MSFT"].rolling(100).mean()


#Graficamos las medias moviles para cada empresa
#APPLE
fig1= plt.figure(figsize=(12,6))
fig1=sns.lineplot(x='Date', y='Media Movil Apple', data=cierre_ajustado, color="darkcyan")
fig1=plt.title("Media Móvil Precio de Cierre de APPLE", fontsize=22, fontstyle='normal',color="darkcyan" )
fig1=plt.xlabel('Año', fontsize=18)
fig1=plt.ylabel("Precio de Cierre Ajustado",fontsize=18)
plt.show()

#GENERAL ELECTRIC
fig2= plt.figure(figsize=(12,6))
fig2=sns.lineplot(x='Date', y='Media Movil GE', data=cierre_ajustado, color="darkslategrey")
fig2=plt.title("Media Móvil Precio de Cierre Acciones de GE", fontsize=22, fontstyle='normal',color="darkslategrey")
fig2=plt.xlabel('Año', fontsize=18)
fig2=plt.ylabel("Precio de Cierre Ajustado",fontsize=18)
plt.show()

#GOOGLE
fig3= plt.figure(figsize=(12,6))
fig3=sns.lineplot(x='Date', y='Media Movil GOOGLE', data=cierre_ajustado, color="darkred")
fig3=plt.title("Media Móvil Precio de Cierre Acciones de GOOGLE", fontsize=22, fontstyle='normal',color="darkred")
fig3=plt.xlabel('Año', fontsize=18)
fig3=plt.ylabel("Precio de Cierre Ajustado",fontsize=18)
plt.show()

#IBM
fig4= plt.figure(figsize=(12,6))
fig4=sns.lineplot(x='Date', y='Media Movil IBM', data=cierre_ajustado, color="darkslategrey")
fig4=plt.title("Media Móvil Precio de Cierre Acciones de IBM", fontsize=22, fontstyle='normal',color="darkslategrey")
fig4=plt.xlabel('Año', fontsize=18)
fig4=plt.ylabel("Precio de Cierre Ajustado",fontsize=18)
plt.show()

#MICROSOFT
fig5= plt.figure(figsize=(12,6))
fig5=sns.lineplot(x='Date', y='Media Movil MSFT', data=cierre_ajustado, color="darkcyan")
fig5=plt.title("Media Móvil Precio de Cierre Acciones de MICROSOFT", fontsize=22, fontstyle='normal',color="darkcyan")
fig5=plt.xlabel('Año', fontsize=18)
fig5=plt.ylabel("Precio de Cierre Ajustado",fontsize=18)
plt.show()


#PARTE 2
def data_retornos(dataframe,cierre_empresa,retorno_empresa):
    
    data_empresa=dataframe.loc[:,["Date",cierre_empresa]]
    data_empresa["day_shifted"]=data_empresa[cierre_empresa].shift(1)
    data_empresa=data_empresa.astype({cierre_empresa: float, "day_shifted":float})
    data_empresa[retorno_empresa]=data_empresa[cierre_empresa]/data_empresa["day_shifted"] - 1
    del(data_empresa["day_shifted"])
    del(data_empresa[cierre_empresa])
    return data_empresa

#Retornos de Cada Empresa
data_apple= data_retornos(dataframe,"Close AAPL", "Retorno AAPL")
data_GE= data_retornos(dataframe,"Close.1 GE", "Retorno GE")    
data_GOOG= data_retornos(dataframe,'Close.2 GOOG', "Retorno GOOG")   
data_IBM= data_retornos(dataframe,'Close.3 IBM', "Retorno IBM")   
data_MSFT= data_retornos(dataframe,'Close.4 MSFT', "Retorno MSFT")   

#Dataframe de los retornos de las 5 empresas
dataframe_retornos=pd.merge(data_apple,data_GE, on="Date")
dataframe_retornos=pd.merge(dataframe_retornos,data_GOOG, on="Date")
dataframe_retornos=pd.merge(dataframe_retornos,data_IBM, on="Date")
dataframe_retornos=pd.merge(dataframe_retornos,data_MSFT, on="Date")

#limpiamos Cierre Ajustado
cierre_ajustado=cierre_ajustado.loc[:,['Date',"Media Movil Apple","Media Movil GE","Media Movil GOOGLE", "Media Movil IBM", "Media Movil MSFT"]]
#Unimos las columnas calculadas
dataframe=pd.merge(dataframe,cierre_ajustado, on='Date')
dataframe=pd.merge(dataframe,dataframe_retornos, on='Date')
#Salvamos Dataframe final
dataframe.to_csv('Acciones_MM_y_Ret_Empresas.csv')

del(cierre_ajustado)
del(data_apple)
del(data_GE)
del(data_GOOG)
del(data_IBM)
del(data_MSFT)

#Grafico de los retornos

#APPLE
fig6 = plt.figure(figsize=(12,6))
fig6=sns.lineplot(x='Date', y="Retorno AAPL", data=dataframe_retornos, color="darkcyan")
fig6=plt.title("Retornos de APPLE (2006 - 2020)", fontsize=22, fontstyle='normal',color="darkcyan")
fig6=plt.xlabel('Año', fontsize=18)
fig6=plt.ylabel("Retornos",fontsize=18)
plt.show()

#GENERAL ELECTRIC
fig7 = plt.figure(figsize=(12,6))
fig7=sns.lineplot(x='Date', y="Retorno GE", data=dataframe_retornos, color="darkslategrey")
fig7=plt.title("Retornos de GENERAL ELECTRIC (2006 - 2020)", fontsize=22, fontstyle='normal',color="darkslategrey")
fig7=plt.xlabel('Año', fontsize=18)
fig7=plt.ylabel("Retornos",fontsize=18)
plt.show()

#GOOGLE
fig8 = plt.figure(figsize=(12,6))
fig8=sns.lineplot(x='Date', y="Retorno GOOG", data=dataframe_retornos, color="darkred")
fig8=plt.title("Retornos de GOOGLE (2006 - 2020)", fontsize=22, fontstyle='normal',color="darkred")
fig8=plt.xlabel('Año', fontsize=18)
fig8=plt.ylabel("Retornos",fontsize=18)
plt.show()

#IBM
fig9 = plt.figure(figsize=(12,6))
fig9=sns.lineplot(x='Date', y="Retorno IBM", data=dataframe_retornos, color="darkslategrey")
fig9=plt.title("Retornos de IBM (2006 - 2020)", fontsize=22, fontstyle='normal',color="darkslategrey")
fig9=plt.xlabel('Año', fontsize=18)
fig9=plt.ylabel("Retornos",fontsize=18)
plt.show()

#MICROSOFT

fig10 = plt.figure(figsize=(12,6))
fig10=sns.lineplot(x='Date', y="Retorno MSFT", data=dataframe_retornos, color="darkcyan")
fig10=plt.title("Retornos de MICROSOFT (2006 - 2020)", fontsize=22, fontstyle='normal',color="darkcyan")
fig10=plt.xlabel('Año', fontsize=18)
fig10=plt.ylabel("Retornos MICROSOFT",fontsize=18)
plt.show()


#Mapa de Calor para los retornos de las 5 empresas
corr_matrix=dataframe_retornos.corr()
fig11=plt.figure(figsize=(8,8))
fig11=sns.heatmap(corr_matrix, cmap='icefire')
plt.show()


#Pairplot de los retornos
fig12=sns.pairplot(dataframe_retornos.iloc[:,1:])
del(dataframe_retornos)
del(corr_matrix)

"""
PARTE 2: Modelo de Regresion


EMPRESA: GOOGLE
"""

google_df=dataframe.loc[:,['Date','Adj Close.2 GOOG','Volume.2 GOOG','High.2 GOOG',
                           'Low.2 GOOG','Open.2 GOOG','Close.2 GOOG'] ]
google_df.iloc[:,1:]=google_df.iloc[:,1:].astype(float)

google_df["indice_diario"]=(google_df['High.2 GOOG']-google_df['Low.2 GOOG'])/google_df['Close.2 GOOG']

google_df["cambio_porcentual"]=(google_df['Close.2 GOOG'] - google_df['Open.2 GOOG'])/google_df['Open.2 GOOG']

last5_df=google_df.iloc[-5:,:]

google_df=google_df.iloc[:-5,:]

google_df.to_csv('dataframeGOOGLE.csv')

google_df.Date=pd.to_datetime(google_df.Date)

#Matrices para entrenar y probar el modelo
X=google_df.loc[:,['Date','Volume.2 GOOG', 'indice_diario','cambio_porcentual']].values
Y=google_df.loc[:,'Adj Close.2 GOOG'].values.reshape(-1,1)


last5_X=last5_df.loc[:,['Date','Volume.2 GOOG', 'indice_diario','cambio_porcentual']].values
last5_Y=last5_df.loc[:,'Adj Close.2 GOOG'].values.reshape(-1,1)



#ESTANDARIZAMOS
sc = StandardScaler()
X_sinfecha= sc.fit_transform(X[:,1:])
Y=sc.fit_transform(Y)
last5_X_sinfecha= sc.fit_transform(last5_X[:,1:])
last5_Y=sc.fit_transform(last5_Y)
X[:,1:]=X_sinfecha
last5_X[:,1:]=last5_X_sinfecha

#ENTRENAMIENTO: Encontrando el mejor modelo
# Creamos rejilla de posibles parametros
parametros_rejilla = {'kernel':('rbf', 'poly'), 'C': [1e10], 'epsilon': [10, 1, 0.5, 0.1]}

# Creamos el buscador para los parametros optimos
regresor_rejilla = GridSearchCV(SVR(), parametros_rejilla,n_jobs=-1, verbose=1)

# Entrenamos 
regresor_rejilla.fit(X[:,1:], Y.ravel())

# Mejores parámetros
print(regresor_rejilla.best_params_)
print(regresor_rejilla.best_estimator_)

#Prediccion Y
Y_pred = regresor_rejilla.predict(X[:,1:])

#Devolvemos a los valores iniciales para graficar
Y=sc.inverse_transform(Y)
Y_pred=sc.inverse_transform(Y_pred)


#Grafica de Modelos
sns.lineplot(X[:,0], Y[:,0], color = "darkred")
sns.lineplot(X[:,0],Y_pred, color = 'teal')
plt.suptitle('Prediccion Precio de Cierre Google (2006-2020)')
plt.xlabel('Año')
plt.ylabel('Precio de Cierre Ajustado')
plt.show() 


#PREDICCION ULTIMOS 5 DIAS

last5_Y_pred= regresor_rejilla.predict(last5_X_sinfecha)
#devolvemos a valores originales para graficar.
last5_Y_pred=sc.inverse_transform(last5_Y_pred)

fig15=plt.figure(figsize=(12,6))
sns.lineplot(last5_X[:,0],last5_Y_pred, color = 'teal', ls='--')
sns.lineplot(x=last5_X[:,0],y=last5_Y.ravel(), color = "darkred")
plt.suptitle('Prediccion Precio de Cierre Google Noviembre 2020)')
plt.title('Ultimos 5 dias')
plt.xlabel('Año')
plt.ylabel('Precio de Cierre Ajustado')
plt.xticks(rotation=70)
plt.show() 


#Metricas
def metricas(y_true, y_pred):

    mae=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 

    print('Error Medio Absoluto (MAE): ', round(mae,4))
    print('Error Medio Cuadratico (MSE):: ', round(np.sqrt(mse),4))

print('Metricas Data Entrenamiento')
metrica_modelo=metricas(Y,Y_pred)
print('Metricas ultimos 5 días de Noviembre')
metrica_ultimos_dias=metricas(last5_Y,last5_Y_pred)


#guardamos el modelo y las predicciones


joblib.dump(regresor_rejilla, 'regresionSVR.pkl')

X=pd.DataFrame(X,index=list(range(len(X))),columns=['Date','Volume.2 GOOG', 'indice_diario','cambio_porcentual'] )
last5_X=pd.DataFrame(last5_5, index=list(range(len(last5_X), len(last5_X)+5)),columns=['Date','Volume.2 GOOG', 'indice_diario','cambio_porcentual']  )

XGOOGLE=X.append(last5_X)


Y_pred=pd.DataFrame(Y.pred.reshape(-1,1), index=list(range(len(Y_pred))), columns='Precio de Cierre Predicho')
last5_Y_pred=pd.DataFrame(Y.pred.reshape(-1,1), index=list(range(len(last5_Y_pred),len(last5_Y_pred)+5)), columns='Precio de Cierre Predicho')

YGOOGLE=Y.append(last5_Y_pred)

XGOOGLE['Precio de Cierre Predicho']=YGOOGLE

XGOOGLE.to_csv('Dataframe_de_prediccion.csv')




""""
ESTE ES EL MODELO CON EL C MAS ALTO DEL QUE OBTUVE UN RESULTADO
"""

#Entrenamos el Modelo
regresorSVR = SVR( kernel="rbf", C=100000, epsilon=0.3)
regresorSVR.fit(X[:,1:], Y.ravel())

Y_pred = regresorSVR.predict(X[:,1:])

#Devolvemos a los valores iniciales para graficar
Y=sc.inverse_transform(Y)
Y_pred=sc.inverse_transform(Y_pred)

#Grafica de Modelos
fig14=plt.figure(figsize=(12,6))
sns.lineplot(X[:,0],Y_pred, color = 'teal', ls='--')
sns.lineplot(X[:,0], Y.ravel, color = "darkred")
plt.suptitle('Prediccion Precio de Cierre Google (2006-2020)')
plt.xlabel('Año')
plt.ylabel('Precio de Cierre Ajustado')
plt.show() 


#PREDICCION ULTIMOS 5 DIAS

last5_Y_pred= regresorSVR.predict(last5_X_sinfecha).reshape(-1,1)
#devolvemos a valores originales para graficar.
last5_Y_pred=sc.inverse_transform(last5_Y_pred).reshape(-1,1)

fig15=plt.figure(figsize=(12,6))
sns.lineplot(last5_X[:,0],last5_Y_pred, color = 'teal', ls='--')
sns.lineplot(last5_X[:,0],last5_Y.ravel(), color = "darkred")
plt.suptitle('Prediccion Precio de Cierre Google Noviembre 2020)')
plt.title('Ultimos 5 dias')
plt.xlabel('Año')
plt.ylabel('Precio de Cierre Ajustado')
plt.xticks(rotation=70)
plt.show() 


#Metricas
def metricas(y_true, y_pred):

    mae=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 

    print('Error Medio Absoluto (MAE): ', round(mae,4))
    print('Error Medio Cuadratico (MSE):: ', round(np.sqrt(mse),4))

print('Metricas Data Entrenamiento')
metrica_modelo=metricas(Y,Y_pred)
print('Metricas ultimos 5 días de Noviembre')
metrica_ultimos_dias=metricas(last5_Y,last5_Y_pred)


#guardamos el modelo y las predicciones



joblib.dump(regresorSVR, 'regresionSVR.pkl')

X=pd.DataFrame(X,index=list(range(len(X))),columns=['Date','Volume.2 GOOG', 'indice_diario','cambio_porcentual'] )
last5_X=pd.DataFrame(last5_X, index=list(range(len(X),len(X)+5)),columns=['Date','Volume.2 GOOG', 'indice_diario','cambio_porcentual'])

XGOOGLE=X.append(last5_X)


Y_pred=pd.DataFrame(Y_pred, index=list(range(len(Y_pred))), columns=['Precio de Cierre Predicho'])
last5_Y_pred=pd.DataFrame(last5_Y_pred, index=list(range(len(Y_pred),len(Y_pred)+5 )), columns=['Precio de Cierre Predicho'])

YGOOGLE=Y_pred.append(last5_Y_pred)

XGOOGLE['Precio de Cierre Predicho']=YGOOGLE

XGOOGLE.to_csv('Dataframe_de_prediccion.csv')






