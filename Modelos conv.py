import numpy as np
import joblib ### para cargar array
########Paquetes para NN #########
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
import pandas as pd
####instalar paquete !pip install keras-tuner
import keras_tuner as kt
from tensorflow.keras.utils import to_categorical

### cargar bases_procesadas ####

x_train = joblib.load('salidas\\x_train.pkl')
y_train = joblib.load('salidas\\y_train.pkl')
x_test = joblib.load('salidas\\x_test.pkl')
y_test = joblib.load('salidas\\y_test.pkl')

x_train[0]
x_train=x_train.astype("float32")
x_test=x_test.astype("float32")
x_train.max()
x_train.min()

x_train/=255
x_test/=255

x_train.shape
x_test.shape

np.product(x_train[15].shape) ## cantidad de variables por imagen

np.unique(y_train, return_counts=True)
np.unique(y_test, return_counts=True)

#y_train1=to_categorical(y_train)
#y_test1=to_categorical(y_test)
#Red convolucional
model_cov=tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model_cov.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", "recall", "precision"])
#Entrenamiento
model_cov.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))

model_cov.summary()


#Evaluar el modelo
test_loss, test_acc, test_recall, test_precision = model_cov.evaluate(x_test, y_test, verbose=2)
print("Test recall:", test_recall)
model_cov.predict(x_test)
#Matriz de confusión
#evaluar algun porcentaje como filtro____pred_test=(rd_model.predict(x_test) > 0.50 ).astype("int")
pred_test=(model_cov.predict(x_test) > 0.20 ).astype("int")
#pred_test.shape #El modelo tiene una probabilidad mayor a 80% en todas las clases

cm=metrics.confusion_matrix(y_test, pred_test)
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['no_tumor', 'tumor' ])
disp.plot()
print(metrics.classification_report(y_test, pred_test))


#Se le puede bajar el numero de filtros para disminuir el sobreajuste
# o disminuir epochs o capas
#Aplicar regularizacion
#Tunning
##### función con definicion de hiperparámetros a afinar
hp = kt.HyperParameters()

def build_model(hp):
    
    dropout_rate=hp.Float('DO', min_value=0.1, max_value= 0.5, step=0.05)
    reg_strength = hp.Float("rs", min_value=0.0001, max_value=0.005, step=0.001)
    optimizer = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop']) ### en el contexto no se debería afinar
   
    ####hp.Int
    ####hp.Choice
    

    model= tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    
  
    
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=0.001)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
   
    model.compile(
        optimizer=opt, loss="categorical_crossentropy", metrics=["recall"],
    )
    
    
    return model



###########

tuner = kt.RandomSearch(
    hypermodel=build_model,
    hyperparameters=hp,
    tune_new_entries=False, ## solo evalúe los hiperparámetros configurados
    objective=kt.Objective("val_recall", direction="max"),
    max_trials=10,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld"
)



tuner.search(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=100)

fc_best_model = tuner.get_best_models(num_models=1)[0]
tuner.results_summary()
fc_best_model.summary()


#################### Mejor redes ##############
fc_best_model.save('salidas\\fc_model.h5')
cnn_model.save('salidas\\cnn_model.h5')


cargar_modelo=tf.keras.models.load_model('salidas\\cnn_model.h5')
cargar_modelo.summary()




