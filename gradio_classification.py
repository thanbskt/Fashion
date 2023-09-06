
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import gradio as gr
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.applications.xception import preprocess_input

model = load_model('/home/thanos/Desktop/Sustainable_fashion/repos/Fashion/model_class_polyvore_vong_v1_NO_prob.h5')


class_names={0: 'Accessories',
 1: 'Bags',
 2: 'Bottoms',
 3: 'Onepiece',
 4: 'Outwear',
 5: 'Shoes',
 6: 'Tops'}



def get_classification(path_example):
    # Create a ColorThief object with the input image
    img = image.load_img(path_example, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    probas = model.predict(x)[0]
    predicted_classes = probas.argsort()[-2:][::-1] 
    resutl1 = 'First Class name: {}, class number: {}\n'.format(class_names[predicted_classes[0]],predicted_classes[0])
    result2 = 'Second Class name: {}, class number: {}\n'.format(class_names[predicted_classes[1]],predicted_classes[1])
    # Return the dominant image
    return resutl1,result2

title = "Classifications for cloth images"
description = "Takes an image as input and the category of the cloth"
inputs = [gr.Image(type='filepath')]
outputs = [gr.Textbox(label='First Prediction'),gr.Textbox(label='Second Prediction')]
allow_flagging = 'manual'
flagging_options = ['First prediction is wrong']

demo = gr.Interface(fn=get_classification,inputs=inputs,outputs=outputs, allow_flagging=allow_flagging,flagging_options=flagging_options )
demo.launch()

