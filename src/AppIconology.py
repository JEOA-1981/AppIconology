#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jesús Eduardo Oliva Abarca
"""

import streamlit as st
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import imagehash
import tensorflow as tf
from PIL import Image, ImageOps
from torchvision import models, transforms

from const import CLASES, COLORES
from settings import UMBRAL_CONFIANZA_DEFECTO, IMAGEN_DEFECTO, MODELO, PROTOTXT


def main():
   st.title('Caso de estudio: Visión computacional para el estudio de las artes plásticas')
   opcion = st.sidebar.selectbox(label= 'Selecciona una opción',
                                  options= ['Bienvenida', 'Identificación básica de objetos',
                                            'Detección de bordes y contornos',
                                            'Clasificación automatizada de obras de la plástica'])
   if opcion == 'Bienvenida':
       bienvenida()
   elif opcion == 'Identificación básica de objetos':
       identificacion_objetos()
   elif opcion == 'Detección de bordes y contornos':
       deteccion_contornos_bordes()
   elif opcion == 'Clasificación automatizada de obras de la plástica':
       clasificacion()

def bienvenida():
    st.markdown("""Esta aplicación web ha sido desarrollada por Jesús Eduardo Oliva Abarca, como parte de un proyecto general de investigación 
    que parte del enfoque de la analítica cultural de Lev Manovich, el cual aborda las aplicaciones de las herramientas, métodos y técnicas
    de la ciencia de datos para el estudio de conjuntos de datos culturales masivos.
    En esta aplicación, el usuario puede examinar el funcionamiento de algunas aplicaciones generales de la visión computacional, como lo 
    son la identificación de objetos, la detección de bordes y contornos, y la clasificación automatizada de imágenes. Aunque la aplicación está pensada
    para operar con imágenes de obras artísticas, funciona también con imágenes de diferentes tipos y clases.""")
    st.markdown("""El propósito de esta aplicación es comprobar si las técnicas de procesamiento de la visión computacional son de utilidad 
    para la identificación de los elementos esenciales en una obra de arte de la plástica, y, con base en ello, facilitar y agilizar el análisis de la 
    imagen artística para los y las interesadas en este tema.""")
    st.markdown("""Es importante indicar que la identificación de objetos se realiza con un modelo pre-entrenado, por lo que su uso para el examen de imágenes
    artísticas es todavía muy inexacto, y ante todo, es un ensayo para un modelo mejorado, cuyo entrenamiento se realice con base en imágenes del ámbito 
    artístico; en lo que respecta a la sección de clasificación de imágenes, el modelo para esta parte se entrenó empleando el conjunto de datos provisto por el usuario 'Danil', 
    de Kaggle (https://www.kaggle.com/thedownhill/art-images-drawings-painting-sculpture-engraving).
    Sin embargo, el sistema de clasificación resultante es todavía inexacto, y requiere ser re-entreando con ejemplos más diversificados, o, en su defecto,
    ajustar los parámetros de entrenamiento según las características de las imágenes a clasificar.""")
    st.markdown("""
    Cualquier duda o comentario: 
        
    jeduardo.oliv@gmail.com""")
    
    st.markdown('https://github.com/JEOA-1981')
    st.markdown('https://www.linkedin.com/in/jes%C3%BAs-eduardo-oliva-abarca-78b615157/')
    st.markdown('## **Nota: esta aplicación se encuentra aún en fase de desarrollo, su uso recomendado es meramente como una herramienta de análisis**')

@st.cache
def procesamiento(imagen):
    blob = cv2.dnn.blobFromImage(cv2.resize(imagen, (300, 300)), 0.007843, (300, 300), 127.5)
    red = cv2.dnn.readNetFromCaffe(PROTOTXT, MODELO)
    red.setInput(blob)
    detecciones = red.forward()
    return detecciones

@st.cache
def anotar(imagen, detecciones, umbral_confianza= UMBRAL_CONFIANZA_DEFECTO):
    (h, w) = imagen.shape[:2]
    etiquetas = []
    for i in np.arange(0, detecciones.shape[2]):
        confianza = detecciones[0, 0, i, 2]

        if confianza > umbral_confianza:
            idx = int(detecciones[0, 0, i, 1])
            caja = detecciones[0, 0, i, 3:7] * np.array([w, h, w, h])
            (inicioX, inicioY, finX, finY) = caja.astype("int")
            etiqueta = f"{CLASES[idx]}: {round(confianza * 100, 2)}%"
            etiquetas.append(etiqueta)
            cv2.rectangle(imagen, (inicioX, inicioY), (finX, finY), COLORES[idx], 2)
            y = inicioY - 15 if inicioY - 15 > 15 else inicioY + 15
            cv2.putText(imagen, etiqueta, (inicioX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORES[idx], 2
            )
    return imagen, etiquetas

def identificacion_objetos():
   st.subheader('Identificación de objetos con MobileNet SSD')
   with st.beta_expander('¿Qué es MobileNet SSD'):
        st.info('''El modelo MobileNet SSD se basa en una red neuronal de Aprendizaje Profundo (*Deep Learning*) diseñada para una la detección de objetos
        mediante la localización de atributos diferenciadores y su separación con base a "cajas" contenedoras predefinidas. Si se desea profundizar en este tema,
        puede consultarse el siguiente artículo, Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.-Y., & Berg, A. C. (2016). 
        SSD: Single Shot MultiBox Detector. ArXiv:1512.02325 [Cs], 9905, 21–37. https://doi.org/10.1007/978-3-319-46448-0_2''')
   archivo = st.file_uploader('Sube una imagen (se admiten archivos .png, .jpg y .jpeg)', type=["png", "jpg", "jpeg"])
   umbral_confianza = st.slider('Umbral de confianza', 0.0, 1.0, UMBRAL_CONFIANZA_DEFECTO, 0.05)

   if archivo is not None:
       imagen = np.array(Image.open(archivo))

   else:
       imagen_defecto = IMAGEN_DEFECTO
       imagen = np.array(Image.open(imagen_defecto))

   detecciones = procesamiento(imagen)
   imagen, etiquetas = anotar(imagen, detecciones, umbral_confianza)

   st.image(imagen, caption= 'Imagen con objetos identificados', use_column_width= True,)

   st.write(etiquetas)

def cargar_imagen(archivo):
    imagen = cv2.imread(archivo)
    return imagen

def deteccion_contornos_bordes():
    st.subheader('Detección de bordes y contornos')
    with st.beta_expander('Información sobre bordes y contornos'):
        st.info('''La detección de bordes se basa en el algoritmo de Canny, desarrollado por John F. Canny en 1986. En la visión computacional, el algoritmo
        lleva a cabo primero un filtrado y reducción del "ruido", o variaciones aleatorias de la luminosidad y del color en la imagen; después, se localizan
        los gradiente de intensidad, esto es, las zonas de mayor concentración o intensidad luminosa o cromática, y finalmente, se remueven y se suprimen los
        píxeles innecesarios en la imagen. Para mayor información, véase: https://docs.opencv.org/3.4/da/d5c/tutorial_canny_detector.html''')
        st.info('''Los contornos, por su parte, se definen como las líneas en las que convergen todos los puntos, de similar intensidad, que 
        conforman la figura de un objeto. En el estudio de imágenes artísticas, la detección de contornos podría ser de utilidad para analizar las formas y las
        texturas predominantes en una obra, o en un conjunto de piezas artísticas. Para mayor información, véase: https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html''')
    archivo = st.file_uploader('Sube una imagen (se admiten archivos .png, .jpg y .jpeg)', type= ['png', 'jpg', 'jpeg'])
    if archivo is not None:
        imagen = np.array(Image.open(archivo))
    
    else:
        imagen_defecto = IMAGEN_DEFECTO
        imagen = np.array(Image.open(imagen_defecto))
        
    y = st.slider('Cambia los valores para incrementar o disminuir la detección de contornos',min_value = 50,max_value = 255)  
    
    boton_uno, boton_dos = st.beta_columns(2)
    with boton_uno:
        if st.checkbox('Detectar bordes', key= 0):
            bordes = cv2.Canny(imagen,100,200)
            st.image([imagen, bordes])
    
    with boton_dos:
        if st.checkbox('Detectar contornos', key= 1):
            imagen_cv2 = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
            reticula, trillado = cv2.threshold(imagen_cv2,y,255,0)
            contornos, jerarquia = cv2.findContours(trillado, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
            img = cv2.drawContours(imagen, contornos, -1, (0,255,0), 3)
 
        
            st.image(trillado, use_column_width= True, clamp= True)
            st.image(img, use_column_width= True, clamp= True)
       
def importacion_prediccion(datos_imagenes, modelo):
    tamano = (75, 75)
    imagen = ImageOps.fit(datos_imagenes, tamano, Image.ANTIALIAS)
    imagen = np.asarray(imagen)
    img = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    img_redimensionada = (cv2.resize(img, dsize= (75, 75), interpolation= cv2.INTER_CUBIC))/255.0
    img_reformada = img_redimensionada[np.newaxis, ...]
    prediccion = modelo.predict(img_reformada)  
    return prediccion

modelo = tf.keras.models.load_model('modelo_clasificación.hdf5')


def clasificacion():
    st.subheader('Clasificación de imágenes de obras de la plástica')
    with st.beta_expander('Especificaciones sobre el modelo de clasificación'):
        st.info('''Este modelo fue entrenado a partir del *corpora* provisto por el usuario "Danil", en el sitio web Kaggle 
        (https://www.kaggle.com/thedownhill/art-images-drawings-painting-sculpture-engraving). Las cinco categorías empleadas para el entrenamiento corresponden
        a cinco formas expresivas de las artes plásticas: el dibujo, el grabado, la iconografía (imágenes correspondientes a una narrativa cosmogónica o 
        tradiciones míticas), la pintura y la escultura. El modelo, cabe señalar, es inexacto, por lo que se incluye en esta aplicación con fines meramente
        ilustrativos.''')
    st.info('Las categorías de la clasificación son: a) dibujo, b) grabado, c) iconografía, d) pintura, e) escultura')
    archivo = st.file_uploader('Por favor, suba un archivo de imagen (.png, .jpg)', type= ['jpg', 'png'])

    if archivo is None:
        st.text('No has subido ninguna imagen')
    else:
        imagen = Image.open(archivo)
        st.image(imagen, use_column_width= True)
        prediccion = importacion_prediccion(imagen, modelo)
        
        if np.argmax(prediccion) == 0:
            st.write('Es un dibujo')
        elif np.argmax(prediccion) == 1:
            st.write('Es un grabado')
        elif np.argmax(prediccion) == 2:
            st.write('Es una iconografía')
        elif np.argmax(prediccion) == 3:
            st.write('Es una pintura')
        else:
            st.write('Es una escultura')
            
        st.text('Probabilidad (0: dibujo, 1: grabado, 2: iconografía, 3: pintura, 4: escultura)')
        st.write(prediccion)
        st.info("""Los valores numéricos que se muestran corresponden a las probabilidades con las que el modelo clasifica a una imagen como perteneciente
        a las categorías establecidas. Para entender estas cifras, se considera la posición del valor respecto al 0: un ejemplo sería el de un valor de 0.7801,
        que correspondería a un porcentaje de 78%, mientras que una cifra de 0.0426 sería el equivalentea 4%.""")


if __name__ == '__main__':
    main()








