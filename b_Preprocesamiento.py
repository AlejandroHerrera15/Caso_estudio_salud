import numpy as np
import cv2 ### para leer imagenes jpeg
from matplotlib import pyplot as plt ## para gráfciar imágnes
import a_funciones as fn#### funciones personalizadas, carga de imágenes
import joblib ### para descargar array
##### ver ejemplo de imágenes cargadas ######
img1 = cv2.imread('Data/Training/no_tumor/Tr-no_0011.jpg')
##### ver ejemplo de imágenes cargadas ######
plt.imshow(img1)
plt.title('no_tumor')
plt.show()
img1.shape
###### representación numérica de imágenes ####
img1.shape
img1.max() ### máximo valor de intensidad en un pixel
img1.min() ### mínimo valor de intensidad en un pixel
np.prod(img1.shape) ### 5 millones de observaciones cada imágen
#### dado que se necesitarían muchas observaciones (imágenes para entrenar)
#### un modelo con tantas observaciones y no tenemos, vamos a reescalar las imágenes
img1_r = cv2.resize(img1 ,(120, 120))
plt.imshow(img1_r)
plt.title('no_tumor')
plt.show()
np.prod(img1_r.shape)
################################################################
######## Código para cargar todas las imágenes #############
####### reducir su tamaño y convertir en array ################
################################################################
width = 120#tamaño para reescalar imágen
num_classes = 2 #clases variable respuesta
trainpath = 'Data/Training/'
testpath = 'Data/Testing/'

x_train, y_train= fn.img2data(trainpath, width) #R, un in train
x_test, y_test = fn.img2data(testpath, width) #Run in test
np.shape(x_train)
#### convertir salidas a numpy array ####
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
np.unique(y_train, return_counts = True)
np.unique(y_test, return_counts = True)

x_train.shape
x_test.shape
x_test.shape
y_test.shape

np.prod(x_train[1].shape)
y_train.shape

####### salidas del preprocesamiento bases listas ######

joblib.dump(x_train, "salidas\\x_train.pkl")
joblib.dump(y_train, "salidas\\y_train.pkl")
joblib.dump(x_test, "salidas\\x_test.pkl")
joblib.dump(y_test, "salidas\\y_test.pkl")


