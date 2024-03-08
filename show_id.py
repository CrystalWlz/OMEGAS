import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

img = None  # 初始化 img 为 None


def open_image(canvas):
    global img
    filepath = filedialog.askopenfilename()
    img = Image.open(filepath).convert('L')
    tk_img = ImageTk.PhotoImage(img)
    canvas.config(width=img.width, height=img.height) 
    canvas.create_image(0, 0, anchor='nw', image=tk_img)
    canvas.image = tk_img  # keep a reference to the image
    return img

def get_pixel_value(img, x, y):
    pixel_value = img.getpixel((x, y))
    return pixel_value

def main():
    window = tk.Tk()
    window.title("Pixel Value Viewer")

    canvas = tk.Canvas(window, width=500, height=500)
    canvas.pack()

    def open_and_draw_image():
        img = open_image(canvas)
        canvas.bind('<Button-1>', lambda event: show_value(event, img))

    open_button = tk.Button(window, text="Open Image", command=open_and_draw_image)
    open_button.pack()

    value_label = tk.Label(window, text="")
    value_label.pack()

    def show_value(event, img):
        x, y = event.x, event.y
        value = get_pixel_value(img, x, y)
        value_label.config(text=f"Pixel Value at ({x}, {y}): {value}")

    window.mainloop()

if __name__ == "__main__":
    main()
