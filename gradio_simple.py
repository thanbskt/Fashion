from PIL import Image
import gradio as gr

import requests
url = "https://huggingface.co/datasets/nateraw/background-remover-files/resolve/main/twitter_profile_pic.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
image.save('twitter_profile_pic.jpg')

url = "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image.save('obama.jpg')

def sepia(input_img):
    sepia_img = input_img
    return sepia_img

demo = gr.Interface(sepia, gr.Image(shape=(200, 200),type="pil",image_mode="RGBA"),"image",examples=[["obama.jpg"],["twitter_profile_pic.jpg"]])
demo.launch(share=True,auth=("admin", "pass1234"))