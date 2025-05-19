import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog, Label, Button, Frame

from PIL import Image, ImageTk

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image

import lime
from lime import lime_image

from skimage.segmentation import mark_boundaries

import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg




# Constants
IMAGE_SIZE = 224
dataset_path = r'../maize-leaf-dataset'
classes = os.listdir(dataset_path)
classifier = keras.models.load_model('../mobilenetv2_v1_44_0.996.h5')
detector = keras.models.load_model('maize_detector_V1_23_0.990.h5')

# Globals
uploaded_image_path = None
original_img_display = None
lime_img_display = None
img_array_normalized = None
predicted_index_global = None
probabilities_global = None
bar_canvas = None

# GUI Setup
root = tk.Tk()
root.title("LEAFLENS")
root.geometry("1100x750")
root.configure(bg="#e8f5e9")

# Fonts
BUTTON_FONT = ("Nunito", 12)
LABEL_FONT = ("Nunito", 14)

# Frames
title_frame = Frame(root, bg="#e8f5e9")
title_frame.pack(pady=0)

top_frame = Frame(root, bg="#e8f5e9")
top_frame.pack(pady=10)

img_frame = Frame(root, bg="#e8f5e9")
img_frame.pack(pady=0)

bottom_frame = Frame(root, bg="#e8f5e9")
bottom_frame.pack(pady=0)

# Functions
def upload_image():
    global uploaded_image_path, original_img_display, lime_img_display, bar_canvas
    uploaded_image_path = filedialog.askopenfilename()
    if uploaded_image_path:
        img = Image.open(uploaded_image_path).resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        original_img_display.configure(image=img_tk)
        original_img_display.image = img_tk
        prediction_label.config(text="")
        probabilities_label.config(text="")
        lime_img_display.configure(image="")
        get_explanation_btn.pack_forget()
        if bar_canvas:
            bar_canvas.get_tk_widget().destroy()
        predict_btn.config(state=tk.NORMAL)

# Pre-filter function to check if the uploaded image is maize
def is_maize_leaf(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_array_normalized = img_array_expanded / 255.0

        prediction = detector.predict(img_array_normalized)[0][0]
        result = int(round(prediction))  # 0 = Not maize, 1 = Maize

        print(f"Prefilter Prediction: {'Maize' if result == 1 else 'Not maize'} ({prediction:.2%})")
        return result == 1

    except Exception as e:
        print("Error in maize pre-filter:", str(e))
        return False

def predict():
    global img_array_normalized, predicted_index_global, probabilities_global, bar_canvas
    if not uploaded_image_path:
        messagebox.showwarning("No Image", "Please upload an image before predicting.")
        return

    # Prefilter step - check if it's a maize leaf
    if not is_maize_leaf(uploaded_image_path):
        messagebox.showerror("Invalid Input",
                                 "This does not appear to be a maize leaf. Please upload a valid image.")
        return

    img = image.load_img(uploaded_image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array_normalized = img_array / 255.0
    img_array_expanded = np.expand_dims(img_array_normalized, axis=0)

    predictions = classifier.predict(img_array_expanded)
    probabilities_global = predictions[0]
    predicted_index_global = np.argmax(predictions[0])
    predicted_class = classes[predicted_index_global]
    confidence = predictions[0][predicted_index_global]

    prediction_label.config(text=f"Prediction: {predicted_class} ({confidence:.2%})")

    # Sort class probabilities
    sorted_indices = np.argsort(probabilities_global)[::-1]
    sorted_probs = probabilities_global[sorted_indices]
    sorted_classes = [classes[i] for i in sorted_indices]

    # Draw sorted bar chart
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['#66bb6a', '#ffa726', '#29b6f6', '#ab47bc'] * (len(classes) // 4 + 1)
    bars = ax.barh(sorted_classes, sorted_probs, color=colors[:len(classes)])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Confidence", fontsize=10)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{sorted_probs[i]:.2%}", va='center', fontsize=9)
    plt.tight_layout()

    if bar_canvas:
        bar_canvas.get_tk_widget().destroy()
    bar_canvas = FigureCanvasTkAgg(fig, master=bottom_frame)
    bar_canvas.draw()
    bar_canvas.get_tk_widget().pack(pady=10)

    get_explanation_btn.pack(pady=10)


def explain():
    global lime_img_display
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array_normalized.astype('double'),
        classifier.predict,
        top_labels=4,
        hide_color=0,
        num_samples=700
    )

    temp, mask = explanation.get_image_and_mask(
        predicted_index_global,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    lime_img_np = mark_boundaries(temp, mask, color=(1, 1, 0))  # Yellow

    lime_img_np = (lime_img_np * 255).astype(np.uint8)
    lime_img_pil = Image.fromarray(lime_img_np)
    lime_img_pil = lime_img_pil.resize((250, 250))
    lime_img_tk = ImageTk.PhotoImage(lime_img_pil)

    lime_img_display.configure(image=lime_img_tk)
    lime_img_display.image = lime_img_tk

# Widgets
title_label = Label(title_frame, text="Maize Disease Classifier 🌽", bg="#e8f5e9",
                    font=("Nunito", 20, "bold"), fg="#2e7d32")
title_label.pack()

upload_btn = Button(top_frame, text="Upload Image", command=upload_image,
                    bg="#a5d6a7", fg="black", font=BUTTON_FONT, width=18)
upload_btn.pack(side=tk.LEFT, padx=10)

predict_btn = Button(top_frame, text="Predict", command=predict,
                     bg="#81c784", fg="black", font=BUTTON_FONT, width=18)
predict_btn.pack(side=tk.LEFT, padx=10)

get_explanation_btn = Button(top_frame, text="Get Explanation", command=explain,
                             bg="#ffcc80", fg="black", font=BUTTON_FONT, width=18)

original_img_display = Label(img_frame, bg="#e8f5e9")
original_img_display.pack(side=tk.LEFT, padx=30)

lime_img_display = Label(img_frame, bg="#e8f5e9")
lime_img_display.pack(side=tk.RIGHT, padx=30)

prediction_label = Label(bottom_frame, text="", bg="#e8f5e9", font=LABEL_FONT, fg="#2e7d32")
prediction_label.pack(pady=10)

probabilities_label = Label(bottom_frame, text="", bg="#e8f5e9", font=("Nunito", 11), justify="left")
probabilities_label.pack()

# Start GUI
root.mainloop()