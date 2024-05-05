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


##### cargar modelo  ######
modelo=tf.keras.models.load_model('salidas\\fc_model.h5')
#Distribucion de probabilidad de las predicciones del modelo
probabilidades=modelo.predict(x_test)
sns.histplot(probabilidades, legend=False)
pred_test=(probabilidades>0.85).astype('int')
#pred_test=(modelo.predict(x_test)>=0.5080).astype('int')
print(metrics.classification_report(y_test, pred_test))
cm=metrics.confusion_matrix(y_test,pred_test)
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['normal', 'tumor'])
disp.plot()

##### cargar modelo 2 ######
modelo2=tf.keras.models.load_model('salidas\\fc_modelres.h5')
#Distribucion de probabilidad de las predicciones del modelo
probabilidades2=modelo2.predict(x_test)
sns.histplot(probabilidades2, bins=50,legend=False)
pred_test2=(probabilidades2>0.50).astype('int')
#pred_test=(modelo.predict(x_test)>=0.5080).astype('int')
print(metrics.classification_report(y_test, pred_test2))
cm=metrics.confusion_matrix(y_test,pred_test2)
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['normal', 'tumor'])
disp.plot()
