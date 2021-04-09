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
   deteccion_objetos()
   photo()
   clasificacion()

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

def deteccion_objetos():
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

def photo():
    st.header("Thresholding, Edge Detection and Contours")
    
    if st.button('See Original Image of Tom'):
        
        original = Image.open('images/imagen01.jpg')
        st.image(original, use_column_width=True)
        
    image = cv2.imread('images/demo.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    x = st.slider('Change Threshold value',min_value = 50,max_value = 255)
    ret,thresh1 = cv2.threshold(image,x,255,cv2.THRESH_BINARY)
    thresh1 = thresh1.astype(np.float64)
    st.image(thresh1, use_column_width=True,clamp = True)
    
    st.text("Press the button below to view Canny Edge Detection Technique")
    if st.button('Canny Edge Detector'):
        image = load_image("images/demo.jpg")
        edges = cv2.Canny(image,50,300)
        cv2.imwrite('edges.jpg',edges)
        st.image(edges,use_column_width=True,clamp=True)
      
    y = st.slider('Change Value to increase or decrease contours',min_value = 50,max_value = 255)     
    
    if st.button('Contours'):
        im = load_image("images/demo.jpg")
          
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,y,255,0)
        #image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        img = cv2.drawContours(im, contours, -1, (0,255,0), 3)
 
        
        st.image(thresh, use_column_width=True, clamp = True)
        st.image(img, use_column_width=True, clamp = True)
        
            
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








