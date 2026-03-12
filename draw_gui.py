# Mouse Draw Digit Recognition GUI using Tkinter (VS Code Ready)
# ------------------------------------------------------------
# Requirements:
# pip install tensorflow pillow numpy
# Trained model file: mnist_cnn_model.h5 (from Kaggle)

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

# Create main window
root = tk.Tk()
root.title("Draw Digit Recognition")
root.geometry("400x520")
root.resizable(False, False)

# Title label
title = tk.Label(root, text="Draw a Digit (0–9)", font=("Arial", 18, "bold"))
title.pack(pady=10)

# Canvas settings
CANVAS_SIZE = 280
canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
canvas.pack()

# Image for drawing
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
draw = ImageDraw.Draw(image)

# Drawing function
last_x, last_y = None, None

def draw_digit(event):
    global last_x, last_y
    x, y = event.x, event.y
    if last_x is not None:
        canvas.create_line(last_x, last_y, x, y, width=8, fill="black", capstyle=tk.ROUND)
        draw.line([last_x, last_y, x, y], fill=0, width=8)
    last_x, last_y = x, y

# Reset drawing coordinates

def reset(event):
    global last_x, last_y
    last_x, last_y = None, None

# Clear canvas

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=255)
    result_label.config(text="Predicted Digit: ")

# Predict digit

def predict_digit():
    # Resize to MNIST size
    img = image.resize((28, 28))

    # Convert to numpy array
    img_array = np.array(img)

    # Invert colors (MNIST style)
    img_array = 255 - img_array

    # Normalize
    img_array = img_array / 255.0

    # Reshape for CNN
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    result_label.config(text=f"Predicted Digit: {digit}")


# Bind mouse events
canvas.bind("<B1-Motion>", draw_digit)
canvas.bind("<ButtonRelease-1>", reset)

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

predict_btn = tk.Button(btn_frame, text="Predict", width=12, command=predict_digit)
predict_btn.grid(row=0, column=0, padx=5)

clear_btn = tk.Button(btn_frame, text="Clear", width=12, command=clear_canvas)
clear_btn.grid(row=0, column=1, padx=5)

# Result label
result_label = tk.Label(root, text="Predicted Digit: ", font=("Arial", 16))
result_label.pack(pady=15)

# Start GUI
root.mainloop()
