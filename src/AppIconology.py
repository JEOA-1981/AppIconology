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
from settings import UMBRAL_CONFIANZA_DEFECTO, IMAGEN_DEMO, MODELO, PROTOTXT


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
    st.markdown("""Es importante indicar que, en lo que respecta a la sección de clasificación de imágenes, el modelo para esta parte se entrenó
    empleando el conjunto de datos provisto por el usuario 'Danil', de Kaggle, en el siguiente 
    enlace: https://www.kaggle.com/thedownhill/art-images-drawings-painting-sculpture-engraving. Sin embargo, el sistema de clasificación resultante
    es todavía inexacto, y requiere ser re-entreando con ejemplos más diversificados, o, en su defecto, ajustar los parámetros de entrenamiento según las
    características de las imágenes a clasificar.""")
    st.markdown("""
    Cualquier duda o comentario: 
        
    jeduardo.oliv@gmail.com""")
    
    st.markdown('https://github.com/JEOA-1981')
    st.markdown('https://www.linkedin.com/in/jes%C3%BAs-eduardo-oliva-abarca-78b615157/')

@st.cache
def process_image(image):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODELO)
    net.setInput(blob)
    detections = net.forward()
    return detections

@st.cache
def annotate_image(
    image, detections, confidence_threshold=UMBRAL_CONFIANZA_DEFECTO
):
    # loop over the detections
    (h, w) = image.shape[:2]
    labels = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = f"{CLASES[idx]}: {round(confidence * 100, 2)}%"
            labels.append(label)
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORES[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORES[idx], 2
            )
    return image, labels

def identificacion_objetos():
   st.title('Detección de objetos con MobileNet SSD')
   img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
   confidence_threshold = st.slider(
       "Confidence threshold", 0.0, 1.0, UMBRAL_CONFIANZA_DEFECTO, 0.05
   )

   if img_file_buffer is not None:
       image = np.array(Image.open(img_file_buffer))

   else:
       demo_image = IMAGEN_DEMO
       image = np.array(Image.open(demo_image))

   detections = process_image(image)
   image, labels = annotate_image(image, detections, confidence_threshold)

   st.image(
       image, caption=f"Processed image", use_column_width=True,
   )

   st.write(labels)

def load_image(filename):
    image = cv2.imread(filename)
    return image

def deteccion_contornos_bordes():
    archivo = st.file_uploader('Sube una imagen (se admiten archivos .png, .jpg y .jpeg)', type= ['png', 'jpg', 'jpeg'])
    if archivo is not None:
        imagen = np.array(Image.open(archivo))
    
    else:
        imagen_demo = IMAGEN_DEMO
        imagen = np.array(Image.open(imagen_demo))
        
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


if __name__ == '__main__':
    main()








