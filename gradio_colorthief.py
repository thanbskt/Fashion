import numpy as np
from PIL import Image
import requests
import gradio as gr 
from colorthief import ColorThief
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec




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


url = "https://huggingface.co/datasets/nateraw/background-remover-files/resolve/main/twitter_profile_pic.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
image.save('twitter_profile_pic.jpg')


def color_thief(file):
    
    img = Image.open(file)
    img = img.resize((224,224))
    ct = ColorThief(file)
    dominant_color = ct.get_color(quality=1) 
    
    distances = {}
    for color_name, color_value in colors.items():
        distance = color_distance_cie2000(dominant_color, color_value)
        distances[color_name] = distance
    sorted_dict = dict(sorted(distances.items(), key=lambda x: x[1]))
    first_name, first_value = list(sorted_dict.items())[0]
    second_name, second_value = list(sorted_dict.items())[1]
    smallest = list(distances.values()).index(min(distances.values()))
    
    
    
    
    
    fig = plt.figure(layout="constrained")
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])
    ax5 = fig.add_subplot(gs[2,:])

    ax1.imshow(img)
    ax1.set_title('Image')


    ax2.imshow([[dominant_color]])
    ax2.set_title('Dominant Color')

    ax3.imshow([[colors[first_name]]])
    ax3.set_title('First Match')

    ax4.imshow([[colors[second_name]]])
    ax4.set_title('Second match')

    ax5.imshow([[color_values[i] for i in range(len(color_values))]])
    ax5.set_title('Basic colors')
    fig.suptitle("Color detection and clustering")
    
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