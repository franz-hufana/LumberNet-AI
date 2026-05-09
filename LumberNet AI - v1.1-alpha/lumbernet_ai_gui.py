import os
import json
import numpy as np
import tkinter as tk

from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from tensorflow.keras.models import load_model


# =========================================================
# FILE PATHS
# =========================================================
MODEL_V2_PATH = "lumberNet_AI_MobileNetV2.keras"
MODEL_V3_PATH = "lumberNet_AI_MobileNetV3Large.keras"
CLASS_NAMES_PATH = "class_names.json"


# =========================================================
# LOAD MODELS AND LABELS
# =========================================================
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

model_v2 = load_model(MODEL_V2_PATH)
model_v3 = load_model(MODEL_V3_PATH)

print("Both MobileNetV2 and MobileNetV3Large loaded successfully.")


# =========================================================
# DISPLAY LABEL NAME
# =========================================================
def display_class_name(class_name):
    if class_name == "no defect":
        return "natural patterns"
    return class_name


# =========================================================
# PREPROCESS IMAGE
# =========================================================
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# =========================================================
# MODEL PREDICTION
# =========================================================
def predict_with_model(model, img_array):
    predictions = model.predict(img_array, verbose=0)

    probs = predictions[0]
    predicted_index = np.argmax(probs)
    predicted_class = class_names[predicted_index]
    confidence = probs[predicted_index] * 100

    return predicted_class, confidence, probs


def compare_models(image_path):
    img_array = preprocess_image(image_path)

    v2_class, v2_conf, v2_probs = predict_with_model(model_v2, img_array)
    v3_class, v3_conf, v3_probs = predict_with_model(model_v3, img_array)

    if v2_conf > v3_conf:
        recommended = "MobileNetV2"
    elif v3_conf > v2_conf:
        recommended = "MobileNetV3Large"
    else:
        recommended = "Tie"

    return {
        "v2_class": v2_class,
        "v2_conf": v2_conf,
        "v2_probs": v2_probs,
        "v3_class": v3_class,
        "v3_conf": v3_conf,
        "v3_probs": v3_probs,
        "recommended": recommended
    }


# =========================================================
# GUI FUNCTIONS
# =========================================================
def update_topk_box(text_widget, probs, top_k=4):
    text_widget.config(state="normal")
    text_widget.delete("1.0", tk.END)

    top_indices = np.argsort(probs)[::-1][:top_k]

    for rank, i in enumerate(top_indices, start=1):
        text_widget.insert(
            tk.END,
            f"Top {rank}: {display_class_name(class_names[i]):<16} - {probs[i] * 100:.2f}%\n"
        )

    text_widget.config(state="disabled")


def display_full_image(image_path):
    frame_width = 440
    frame_height = 380

    img = Image.open(image_path).convert("RGB")

    img_ratio = img.width / img.height
    frame_ratio = frame_width / frame_height

    if img_ratio > frame_ratio:
        new_width = frame_width
        new_height = int(frame_width / img_ratio)
    else:
        new_height = frame_height
        new_width = int(frame_height * img_ratio)

    img = img.resize((new_width, new_height), Image.LANCZOS)

    background = Image.new("RGB", (frame_width, frame_height), (249, 250, 251))
    offset = ((frame_width - new_width) // 2, (frame_height - new_height) // 2)
    background.paste(img, offset)

    img_display = ImageTk.PhotoImage(background)
    image_label.config(image=img_display, text="")
    image_label.image = img_display


def load_and_predict(image_path):
    if not os.path.exists(image_path):
        messagebox.showerror("File Error", "Image file not found.")
        return

    try:
        display_full_image(image_path)
        file_name_label.config(text=os.path.basename(image_path))

        results = compare_models(image_path)

        v2_prediction_label.config(text=display_class_name(results["v2_class"]))
        v2_confidence_label.config(text=f"{results['v2_conf']:.2f}%")

        v3_prediction_label.config(text=display_class_name(results["v3_class"]))
        v3_confidence_label.config(text=f"{results['v3_conf']:.2f}%")

        update_topk_box(v2_probs_box, results["v2_probs"], top_k=4)
        update_topk_box(v3_probs_box, results["v3_probs"], top_k=4)

        recommended_label.config(
            text=f"Recommended Model: {results['recommended']}"
        )

    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))


def browse_image():
    file_path = filedialog.askopenfilename(
        title="Select Wood Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All Files", "*.*")
        ]
    )

    if file_path:
        load_and_predict(file_path)


def drop_image(event):
    file_path = event.data.strip()

    if file_path.startswith("{") and file_path.endswith("}"):
        file_path = file_path[1:-1]

    load_and_predict(file_path)


# =========================================================
# GUI DESIGN
# =========================================================
root = TkinterDnD.Tk()
root.title("LumberNet AI - Wood Defect Classification System")
root.geometry("1150x720")
root.configure(bg="#F4F6F8")
root.resizable(False, False)


# =========================================================
# HEADER
# =========================================================
header = tk.Frame(root, bg="#1F2937", height=80)
header.pack(fill="x")

title = tk.Label(
    header,
    text="LumberNet AI",
    bg="#1F2937",
    fg="white",
    font=("Segoe UI", 26, "bold")
)
title.pack(side="left", padx=30, pady=18)

subtitle = tk.Label(
    header,
    text="Wood Surface Defect Classification System",
    bg="#1F2937",
    fg="#D1D5DB",
    font=("Segoe UI", 12)
)
subtitle.pack(side="left", pady=28)


# =========================================================
# MAIN CONTAINER
# =========================================================
main = tk.Frame(root, bg="#F4F6F8")
main.pack(fill="both", expand=True, padx=25, pady=20)


# =========================================================
# LEFT PANEL - IMAGE INPUT
# =========================================================
left_panel = tk.Frame(main, bg="white", width=500, height=580)
left_panel.pack(side="left", fill="both", padx=(0, 15))
left_panel.pack_propagate(False)

input_title = tk.Label(
    left_panel,
    text="Image Input",
    bg="white",
    fg="#111827",
    font=("Segoe UI", 18, "bold")
)
input_title.pack(anchor="w", padx=20, pady=(20, 5))

input_desc = tk.Label(
    left_panel,
    text="Drag and drop a wood image or choose a file.",
    bg="white",
    fg="#6B7280",
    font=("Segoe UI", 10)
)
input_desc.pack(anchor="w", padx=20)

drop_area = tk.Label(
    left_panel,
    text="Drop Image Here",
    bg="#EEF2F7",
    fg="#374151",
    relief="ridge",
    borderwidth=2,
    font=("Segoe UI", 14, "bold"),
    height=3
)
drop_area.pack(fill="x", padx=20, pady=15)

drop_area.drop_target_register(DND_FILES)
drop_area.dnd_bind("<<Drop>>", drop_image)

browse_button = tk.Button(
    left_panel,
    text="Browse Image",
    command=browse_image,
    bg="#2563EB",
    fg="white",
    activebackground="#1D4ED8",
    activeforeground="white",
    relief="flat",
    font=("Segoe UI", 11, "bold"),
    height=2
)
browse_button.pack(fill="x", padx=20)

file_name_label = tk.Label(
    left_panel,
    text="No image selected",
    bg="white",
    fg="#6B7280",
    font=("Segoe UI", 9)
)
file_name_label.pack(pady=8)

image_frame = tk.Frame(left_panel, bg="#F9FAFB", width=440, height=380)
image_frame.pack(padx=20, pady=10)
image_frame.pack_propagate(False)

image_label = tk.Label(
    image_frame,
    text="Image preview will appear here",
    bg="#F9FAFB",
    fg="#9CA3AF",
    font=("Segoe UI", 11)
)
image_label.pack(expand=True)


# =========================================================
# RIGHT PANEL - RESULTS
# =========================================================
right_panel = tk.Frame(main, bg="#F4F6F8")
right_panel.pack(side="right", fill="both", expand=True)

result_title = tk.Label(
    right_panel,
    text="Comparative Classification Results",
    bg="#F4F6F8",
    fg="#111827",
    font=("Segoe UI", 18, "bold")
)
result_title.pack(anchor="w", pady=(0, 10))


# =========================================================
# RESULT CARD FUNCTION
# =========================================================
def create_model_card(parent, model_name):
    card = tk.Frame(parent, bg="white", height=245)
    card.pack(fill="x", pady=8)
    card.pack_propagate(False)

    title = tk.Label(
        card,
        text=model_name,
        bg="white",
        fg="#111827",
        font=("Segoe UI", 15, "bold")
    )
    title.pack(anchor="w", padx=20, pady=(15, 8))

    metrics_frame = tk.Frame(card, bg="white")
    metrics_frame.pack(fill="x", padx=20)

    pred_title = tk.Label(metrics_frame, text="Prediction", bg="white", fg="#6B7280", font=("Segoe UI", 9))
    pred_title.grid(row=0, column=0, sticky="w", padx=(0, 60))

    conf_title = tk.Label(metrics_frame, text="Confidence", bg="white", fg="#6B7280", font=("Segoe UI", 9))
    conf_title.grid(row=0, column=1, sticky="w")

    pred_value = tk.Label(metrics_frame, text="-", bg="white", fg="#111827", font=("Segoe UI", 13, "bold"))
    pred_value.grid(row=1, column=0, sticky="w", padx=(0, 60), pady=(3, 10))

    conf_value = tk.Label(metrics_frame, text="-", bg="white", fg="#111827", font=("Segoe UI", 13, "bold"))
    conf_value.grid(row=1, column=1, sticky="w", pady=(3, 10))

    topk_label = tk.Label(
        card,
        text="Defect Probabilities",
        bg="white",
        fg="#374151",
        font=("Segoe UI", 10, "bold")
    )
    topk_label.pack(anchor="w", padx=20)

    probs_box = tk.Text(
        card,
        height=5,
        bg="#F9FAFB",
        fg="#111827",
        relief="flat",
        font=("Consolas", 10)
    )
    probs_box.pack(fill="x", padx=20, pady=8)
    probs_box.config(state="disabled")

    return pred_value, conf_value, probs_box


v2_prediction_label, v2_confidence_label, v2_probs_box = create_model_card(
    right_panel,
    "MobileNetV2"
)

v3_prediction_label, v3_confidence_label, v3_probs_box = create_model_card(
    right_panel,
    "MobileNetV3Large"
)


# =========================================================
# RECOMMENDED MODEL CARD
# =========================================================
recommend_card = tk.Frame(right_panel, bg="#E0F2FE", height=75)
recommend_card.pack(fill="x", pady=12)
recommend_card.pack_propagate(False)

recommended_label = tk.Label(
    recommend_card,
    text="Recommended Model: -",
    bg="#E0F2FE",
    fg="#075985",
    font=("Segoe UI", 16, "bold")
)
recommended_label.pack(anchor="w", padx=20, pady=22)


# =========================================================
# FOOTER
# =========================================================
footer = tk.Label(
    root,
    text="Input Size: 224×224 | Models: MobileNetV2 and MobileNetV3Large | Output Classes: Crack, Fuzz, Knots, Natural Patterns",
    bg="#F4F6F8",
    fg="#6B7280",
    font=("Segoe UI", 9)
)
footer.pack(pady=(0, 8))


root.mainloop()