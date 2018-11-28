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