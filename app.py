import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

# Create main window
root = tk.Tk()
root.title("Handwritten Digit Recognition")
root.geometry("400x500")
root.resizable(False, False)

# Title label
title = tk.Label(root, text="MNIST Digit Predictor", font=("Arial", 18, "bold"))
title.pack(pady=10)

# Canvas for image display
canvas = tk.Canvas(root, width=200, height=200, bg="white")
canvas.pack(pady=10)

img_on_canvas = None
loaded_image = None

# Function to upload image
def upload_image():
    global img_on_canvas, loaded_image

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    if not file_path:
        return

    img = Image.open(file_path).convert("L")  # Convert to grayscale
    img = img.resize((200, 200))
    loaded_image = img

    img_tk = ImageTk.PhotoImage(img)
    canvas.delete("all")
    img_on_canvas = canvas.create_image(100, 100, image=img_tk)
    canvas.image = img_tk

# Function to predict digit
def predict_digit():
    if loaded_image is None:
        messagebox.showwarning("Warning", "Please upload an image first")
        return

    # Resize to MNIST size
    img = loaded_image.resize((28, 28))

    # Convert to numpy array
    img_array = np.array(img)

    # 🔥 IMPORTANT: Invert colors (MNIST style)
    img_array = 255 - img_array

    # Normalize
    img_array = img_array / 255.0

    # Reshape for CNN
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    result_label.config(text=f"Predicted Digit: {digit}")


# Upload button
upload_btn = tk.Button(root, text="Upload Image", command=upload_image, width=20)
upload_btn.pack(pady=10)

# Predict button
predict_btn = tk.Button(root, text="Predict Digit", command=predict_digit, width=20)
predict_btn.pack(pady=10)

# Result label
result_label = tk.Label(root, text="Predicted Digit: ", font=("Arial", 16))
result_label.pack(pady=20)

# Exit button
exit_btn = tk.Button(root, text="Exit", command=root.quit, width=20)
exit_btn.pack(pady=10)

# Start GUI loop
root.mainloop()
