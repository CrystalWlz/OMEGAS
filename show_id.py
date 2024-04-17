import gradio as gr
from PIL import Image
import os

def open_image(filepath):
    img = Image.open(filepath).convert('L')
    return img

def get_pixel_value(img, x, y):
    pixel_value = img.getpixel((x, y))
    return f"Pixel Value at ({x}, {y}): {pixel_value}"

def process_image(img, x, y):
    img = open_image(img.name)
    pixel_value = get_pixel_value(img, x, y)
    return pixel_value

def select_image(folder_path):
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.endswith('.png') or f.endswith('.jpg')]
    return gr.inputs.Dropdown(choices=image_files)

image = gr.inputs.Image(shape=(None, None))
number = gr.inputs.Number()

iface = gr.Interface(fn=process_image, inputs=[select_image("your_folder_path"), "number", "number"], outputs="text")
iface.launch()
