import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K


K.clear_session()

#Donde estan las imagenes que se van a usar
data_entrenamiento= './data/entrenamiento'
data_validacion= './data/validacion'

#Parametros
epocas= 20#Veces que se va a iterar 
altura, longitud= 100,100#Tamaño de la imagen en pixeles
batch_size = 32#Numero de imagenes por iteracion
pasos =1000#Numero de veces que se va  a procesar la informacion en cada epoca
pasos_Validacion = 200#Al final de cada epoca se prueba con estos pasosdevalidacion
filtrosConv1=32#Despues de cada convolucion profundidad de la imagen
filtrosConv2=64#Despues de cada convolucion profundidad de la imagen
tamano_filtro1=(3,3)#tamano del filtro en la convolucion altura longitud
tamano_filtro2=(2,2)#Tamano del filtro en la convolucion altura y longitud, parametros
tamano_pool = (2,2)#Tamano del pool que se va a usar para identificar imagenes
clases=3 #Gato perro y gorila
lr =0.0005 #learning rate


#Preprocesamiento de imagenes
entrenamiento_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True
)
validacion_datagen = ImageDataGenerator(
        rescale=1./255
)
 
imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
        data_entrenamiento,
        target_size=(altura,longitud),
        batch_size=batch_size,
        class_mode='categorical'#Categorica etiquetas perro gato gorila
)

imagen_validacion = validacion_datagen.flow_from_directory(
        data_validacion,
        target_size=(altura,longitud),
        batch_size=batch_size,
        class_mode='categorical'#Categorica etiquetas perro gato gorila
)

#Crea una red convolucional
cnn=Sequential()#Varias capas apiladas

cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding='same', input_shape=(altura, longitud,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())#Una dimension con toda la informacion de la red
cnn.add(Dense(256,activation='relu'))#256 neuronas
cnn.add(Dropout(0.5))#Apagar el 50 porciento de neuronas cada paso,para que aprenda varios caminos
cnn.add(Dense(clases, activation='softmax'))

cnn.compile(loss='categorical_crossentropy', optimizer= optimizers.Adam(lr=lr), metrics=['accuracy'])


cnn.fit_generator(imagen_entrenamiento,steps_per_epoch=pasos, epochs=epocas, validation_data=imagen_validacion, validation_steps=pasos_Validacion)


dir= './modelo/'

if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')  






