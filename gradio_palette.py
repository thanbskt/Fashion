from PIL import Image
import numpy as np
import requests
from colorthief import ColorThief
import matplotlib.pyplot as plt
import gradio as gr
import math
from matplotlib.gridspec import GridSpec
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from collections import OrderedDict


colors ={
"white": (255,255,255),
"black": (0,0,0),
"grey" : (128,128,128),
"beige": (245,245,220),
"brown": (111,59,22),
"blue":  (0,0,128),
"light blue": (137,207,240),
"green": (34,139,34),
"acid green": (203,255,187),
"khaki": (101,98,8),
"red": (255,0,0),
"burgundy": (128,0,32),
"pink": (255,158,181),
"hot pink": (255,28,142),
"purple": (144,48,146),
"lilac":(255,217,255),
"orange":(255,165,0),
"yellow":(255,249,108)}


color_values = list(colors.values())

def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)

def color_distance_cie2000(rgb1,rgb2):
    rgb1 = np.array(rgb1)/255
    rgb2 = np.array(rgb2)/255
    color1_rgb  = sRGBColor(rgb1[0],rgb1[1],rgb1[2])
    color2_rgb  = sRGBColor(rgb2[0],rgb2[1],rgb2[2])
    color1_lab = convert_color(color1_rgb, LabColor)
    color2_lab = convert_color(color2_rgb, LabColor)
    delta_e = delta_e_cie2000(color1_lab, color2_lab)
    return delta_e

# This function takes as input a random color and returns the closest one from the list of colors above 
def get_color_cluster(input_color):
    distances = {}
    for color_name, color_value in colors.items():
        distance = color_distance_cie2000(input_color, color_value)
        distances[color_name] = distance
    sorted_dict = dict(sorted(distances.items(), key=lambda x: x[1]))
    closest_color, closest_value = list(sorted_dict.items())[0]
    return closest_color,closest_value

url = "https://huggingface.co/datasets/nateraw/background-remover-files/resolve/main/twitter_profile_pic.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
image.save('twitter_profile_pic.jpg')


def color_thief(file):
    
    img = Image.open(file)
    img = img.resize((224,224))
    ct = ColorThief(file)
    dominant_colors = ct.get_palette(color_count=2) 
    
   
    
    result = []
    for i in dominant_colors:
        cluster_name, distance = get_color_cluster(i)
        result.append(cluster_name)
        
    result = list(OrderedDict.fromkeys(result))
    
    
    fig = plt.figure(layout="constrained")
    gs = GridSpec(3, 1, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])


    ax1.imshow(img)
    ax1.set_title('Image')

    ax2.imshow([[dominant_colors[i] for i in range(len(dominant_colors))]])
    ax2.set_title('Dominant Colors')

    ax3.imshow([[colors[result[i]] for i in range(len(result))]])
    ax3.set_title('Thier corresponding cluster')
    fig.suptitle("Color detection and clustering, metric:CIE 2000")
    plt.show()
    
    return fig
title = "Dominant color extraction"
description = "Takes an image as input and returns the dominant color and the most similar colors in a given palette"
inputs = [gr.Image(type='filepath',image_mode = "RGBA")]
outputs = gr.Plot()
examples = [["twitter_profile_pic.jpg"]]

allow_flagging = 'manual'
flagging_options = ['Image in displayed wrong','Dominant Color is wrong','First Clustering is wrong', 'Second Clustering is wrong']

demo = gr.Interface(color_thief, 
                    inputs=inputs, 
                    outputs=outputs,
                    examples=examples,
                    title=title,
                    description=description,
                    allow_flagging = allow_flagging,
                    flagging_options = flagging_options)
demo.launch(share=True)
