import numpy as np
import pandas as pd
import cv2 ### para leer imagenes jpeg
### pip install opencv-python
import a_funciones as fn#### funciones personalizadas, carga de imágenes
import tensorflow as tf
import openpyxl

import sys
sys.executable
sys.path

if __name__=="__main__":

    #### cargar datos ####
    path = 'C:/Users/Asus/Caso_estudio_salud/despliegue/'
    x, _, files= fn.img3data(path) #cargar datos de despliegue
    x=np.array(x) ##imagenes a predecir
    
    
    x=x.astype('float')######convertir para escalar
    x/=255######escalar datos


    files2= [name.rsplit('.', 1)[0] for name in files] ### eliminar extension a nombre de archivo

    modelo=tf.keras.models.load_model('salidas\\fc_model4.h5') ### cargar modelo
    prob=modelo.predict(x)


    clas=["tumor" if x > 0.90 else "no tumor" if x < 0.10 else "revision" for x in prob ]

    res_dict={
        "paciente": files2,
        "clas": clas   
    }
    
    resultados=pd.DataFrame(res_dict)

    resultados.to_excel('C:/Users/Asus/Caso_estudio_salud/salidas/clasificados.xlsx', index=False)