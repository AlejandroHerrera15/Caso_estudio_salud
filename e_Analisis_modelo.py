import numpy as np
import joblib ### para cargar array
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

x_train = joblib.load('salidas\\x_train.pkl')
y_train = joblib.load('salidas\\y_train.pkl')
x_test = joblib.load('salidas\\x_test.pkl')
y_test = joblib.load('salidas\\y_test.pkl')


#### Escalar ######################
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') 

x_train /=255 
x_test /=255

#Modelo con clases desbalanceadas con tunnig de 3 hyperparametros
##### cargar modelo  ######
modelo=tf.keras.models.load_model('salidas\\fc_model.h5')
#Distribucion de probabilidad de las predicciones del modelo
probabilidades=modelo.predict(x_test)
sns.histplot(probabilidades, legend=False, bins=15)
pred_test=(probabilidades>0.50).astype('int')
#pred_test=(modelo.predict(x_test)>=0.5080).astype('int')
print(metrics.classification_report(y_test, pred_test))
cm=metrics.confusion_matrix(y_test,pred_test)
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['normal', 'tumor'])
disp.plot()

probabilidadest=modelo.predict(x_train)
sns.histplot(probabilidadest, bins=15,legend=False)
plt.show()
pred_testt=(probabilidadest>0.50).astype('int')
print(metrics.classification_report(y_train, pred_testt))






#Modelo con clases desbalanceados con tunnig de 5 hyperparametros
##### cargar modelo 2 ######
modelo2=tf.keras.models.load_model('salidas\\fc_model+.h5')
modelo2.summary()
#Distribucion de probabilidad de las predicciones del modelo
probabilidades2=modelo2.predict(x_test)
sns.histplot(probabilidades2, bins=15,legend=False)
pred_test2=(probabilidades2>0.50).astype('int')
#pred_test=(modelo.predict(x_test)>=0.5080).astype('int')
print(metrics.classification_report(y_test, pred_test2))
cm=metrics.confusion_matrix(y_test,pred_test2)
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['normal', 'tumor'])
disp.plot()

probabilidades2t=modelo2.predict(x_train)
sns.histplot(probabilidades2t, bins=15,legend=False)
plt.show()
pred_test2t=(probabilidades2t>0.50).astype('int')
print(metrics.classification_report(y_train, pred_test2t))



#Modelo con balanceo de clases con smote y tunnig de 5 hyperparametros
##### cargar modelo 3 ######
modelo3=tf.keras.models.load_model('salidas\\fc_modelres.h5')
modelo3.summary()
#Distribucion de probabilidad de las predicciones del modelo
probabilidades3=modelo3.predict(x_test)
sns.histplot(probabilidades3,legend=False, bins=15)
pred_test3=(probabilidades3>0.50).astype('int')
#pred_test=(modelo.predict(x_test)>=0.5080).astype('int')
print(metrics.classification_report(y_test, pred_test3))
cm=metrics.confusion_matrix(y_test,pred_test3)
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['normal', 'tumor'])
disp.plot()

probabilidades3t=modelo3.predict(x_train)
sns.histplot(probabilidades3t, bins=15,legend=False)
plt.show()
pred_test3t=(probabilidades3t>0.50).astype('int')
print(metrics.classification_report(y_train, pred_test3t))



#Para este modelo se reducio el numero de pixeles por imagen
#ademas se buscaron mas imagenes en internet.


##### cargar modelo 4 ######
modelo4=tf.keras.models.load_model('salidas\\fc_model4.h5')
modelo4.summary()
#Distribucion de probabilidad de las predicciones del modelo
probabilidades4=modelo4.predict(x_test)
sns.histplot(probabilidades4, bins=15,legend=False)
plt.show()
pred_test4=(probabilidades4>0.50).astype('int')
#pred_test=(modelo.predict(x_test)>=0.5080).astype('int')
print(metrics.classification_report(y_test, pred_test4))
cm=metrics.confusion_matrix(y_test,pred_test4, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['tumor', 'normal'])
disp.plot()



#Para entrenamiento
probabilidades4t=modelo4.predict(x_train)
sns.histplot(probabilidades4t, bins=15,legend=False)
plt.show()
pred_test4t=(probabilidades4t>0.50).astype('int')
print(metrics.classification_report(y_train, pred_test4t))


#Se generan en la grafica dos grupos a partir de las predicciones
#Se observara la matriz de confusion para ambos grupos con dos thresholds
#El primer el threshold es 90% y el otro es 10%

#Threshold
sns.histplot(probabilidades4, bins=15,legend=False)
plt.axvline(x=0.10, color='red', linestyle='--')
plt.axvline(x=0.90, color='red', linestyle='--')
plt.show()


#Para el grupo 1
pred_test4_1=(probabilidades4>0.10).astype('int')
print(metrics.classification_report(y_test, pred_test4_1))
cm=metrics.confusion_matrix(y_test,pred_test4_1, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['tumor', 'normal'])
disp.plot()


#Para el grupo 2
pred_test4_2=(probabilidades4>0.90).astype('int')
print(metrics.classification_report(y_test, pred_test4_2))
cm=metrics.confusion_matrix(y_test,pred_test4_2, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['tumor', 'normal'])
disp.plot()