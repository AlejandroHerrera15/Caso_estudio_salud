import numpy as np
import joblib ### para cargar array
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
from sklearn.ensemble import RandomForestClassifier  ### para analizar modelo
import pandas as pd
from sklearn.linear_model import LogisticRegression
import cv2 ### para leer imagenes jpeg
from matplotlib import pyplot as plt #
from tensorflow.keras.utils import to_categorical

#Carga de bases procesadas
x_train = joblib.load('salidas\\x_train.pkl')
y_train = joblib.load('salidas\\y_train.pkl')
x_test = joblib.load('salidas\\x_test.pkl')
y_test = joblib.load('salidas\\y_test.pkl')

#Preprocesamiento
x_train=x_train.astype('float32') #Vienen en entero por lo que  hay que cambiar el formato 
x_test=x_test.astype('float32')
np.max(x_train)
np.max(x_test)

x_train /=255 
x_test /=255

x_train.shape
x_test.shape

#Confimar numero de datos por imagen
np.product(x_train[1].shape)

#Validación valores unicos para la y
np.unique(y_train, return_counts=True)
np.unique(y_test, return_counts=True)

#convertir a una dimension las bases de datos para el random forest y la regresión logística
x_train1d=x_train.reshape(x_train.shape[0],np.product(x_train[1].shape))
x_test1d=x_test.reshape(x_test.shape[0],np.product(x_train[1].shape))
x_train1d.shape
x_test1d.shape

#Regresion logistica
rl=LogisticRegression()
modelorl=rl.fit(x_train1d, y_train)
predrltrain=modelorl.predict(x_train1d)
print(metrics.classification_report(y_train, predrltrain))

predrltest=modelorl.predict(x_test1d)
print(metrics.classification_report(y_test, predrltest))

#Random Foerst Classifier
rf=RandomForestClassifier()
rf.fit(x_train1d, y_train)

preddtctrain=rf.predict(x_train1d)
print(metrics.classification_report(y_train, preddtctrain))

preddtctest=rf.predict(x_test1d)
print(metrics.classification_report(y_test, preddtctest))

#Los resultados se analizan en el informe

#Ahora construimos una red neuronal full connected
#Red neuronal simple
rd_model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

#Optimizador y funcion de perdida
rd_model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy", "recall", "precision"])
#Entrenamiento
rd_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
m1=rd_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
m1

#Grafico para observar sobreajuste
plt.plot(m1.history['precision'])
plt.plot(m1.history['val_precision'])
plt.title('Precisión del modelo')
plt.xlabel('Tiempo de entrenamiento - Epochs')
plt.ylabel('Precisión')
plt.legend(['train', 'val'])
plt.show()

#Evaluar el modelo
test_loss, test_acc, test_recall, test_precision = rd_model.evaluate(x_test, y_test, verbose=2)
print("Test recall:", test_recall)

#Matriz de confusión para test 
#evaluar algun porcentaje como filtro
pred_test=(rd_model.predict(x_test) > 0.50 ).astype("int")
#para train 
pred_testt=(rd_model.predict(x_train) > 0.50 ).astype("int")

cm=metrics.confusion_matrix(y_test, pred_test)
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['no_tumor', 'tumor'])
disp.plot()
#metricas
print(metrics.classification_report(y_test, pred_test)) #metricas para test
print(metrics.classification_report(y_train, pred_testt)) #metricas para train
#Guardar modelo
rd_model.save('path_to_my_model.h5') 

#Por la presencia de sobreajuste se propone regualarizar la red neuronal
fuerza=0.003 
dropout_rate = 0.2
rd_model2=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
    tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(fuerza)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(fuerza)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
#Optimizador y funcion de perdida
rd_model2.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy", "recall", "precision"])
#Entrenamiento
rd_model2.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
m2=rd_model2.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

#Evaluar el modelo
test_loss, test_acc, test_recall, test_precision = rd_model2.evaluate(x_test, y_test, verbose=2)
print("Test recall:", test_recall)

#evaluar algun porcentaje como filtro y matriz de confusion
pred_test2=(rd_model2.predict(x_test)> 0.50 ).astype("int")
#para train 
pred_testt2=(rd_model.predict(x_train) > 0.50 ).astype("int")
cm=metrics.confusion_matrix(y_test, pred_test2)
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['no_tumor', 'tumor'])
disp.plot()

#metricas para test y train
print(metrics.classification_report(y_test, pred_test2))
print(metrics.classification_report(y_train, pred_testt2))

#Grafico de sobreajuste
plt.plot(m2.history['precision'])
plt.plot(m2.history['val_precision'])
plt.title('Precisión del modelo tuneado')
plt.xlabel('Tiempo de entrenamiento - Epochs')
plt.ylabel('Precisión')
plt.legend(['train', 'val'])
plt.show()
