import os
import json
import cv2
import numpy as np
import tkinter as tk
import tensorflow as tf

from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False


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

model_v2 = load_model(MODEL_V2_PATH, compile=False)
model_v3 = load_model(MODEL_V3_PATH, compile=False)

print("Models loaded successfully.")


# =========================================================
# LABEL DISPLAY
# =========================================================
def display_class_name(class_name):
    if class_name == "no defect":
        return "natural patterns"
    return class_name


# =========================================================
# IMAGE PREPROCESSING
# =========================================================
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img).astype("float32")
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# =========================================================
# MODEL PREDICTION
# =========================================================
def predict_with_model(model, img_array):
    predictions = model.predict(img_array, verbose=0)

    probs = predictions[0]
    predicted_index = int(np.argmax(probs))
    predicted_class = class_names[predicted_index]
    confidence = float(probs[predicted_index] * 100)

    return predicted_class, confidence, probs, predicted_index


def compare_models(image_path):
    img_array = preprocess_image(image_path)

    v2_class, v2_conf, v2_probs, v2_index = predict_with_model(model_v2, img_array)
    v3_class, v3_conf, v3_probs, v3_index = predict_with_model(model_v3, img_array)

    if v2_conf >= v3_conf:
        recommended_model = "MobileNetV2"
        selected_model = model_v2
        selected_index = v2_index
        selected_class = v2_class
        selected_conf = v2_conf
    else:
        recommended_model = "MobileNetV3"
        selected_model = model_v3
        selected_index = v3_index
        selected_class = v3_class
        selected_conf = v3_conf

    return {
        "v2_class": v2_class,
        "v2_conf": v2_conf,
        "v2_probs": v2_probs,
        "v3_class": v3_class,
        "v3_conf": v3_conf,
        "v3_probs": v3_probs,
        "recommended_model": recommended_model,
        "selected_model": selected_model,
        "selected_index": selected_index,
        "selected_class": selected_class,
        "selected_conf": selected_conf
    }


# =========================================================
# GRAD-CAM FUNCTIONS
# =========================================================
def get_base_model(model):
    for layer in model.layers:
        if hasattr(layer, "layers"):
            for sub_layer in layer.layers:
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return layer
    return model


def get_last_conv_layer_name(base_model):
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise ValueError("No convolutional layer found for Grad-CAM.")


def apply_classifier_after_base(model, base_model, base_output):
    x = base_output
    se_input = base_output
    base_found = False

    for layer in model.layers:
        if layer.name == base_model.name:
            base_found = True
            continue

        if not base_found:
            continue

        if isinstance(layer, tf.keras.layers.Multiply):
            x = layer([se_input, x])
        elif isinstance(layer, tf.keras.layers.Add):
            x = layer([se_input, x])
        else:
            x = layer(x)

    return x


def make_gradcam_heatmap(img_array, model, pred_index):
    base_model = get_base_model(model)
    last_conv_layer_name = get_last_conv_layer_name(base_model)

    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    grad_feature_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_output = grad_feature_model(img_array)
        predictions = apply_classifier_after_base(model, base_model, base_output)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise ValueError("Grad-CAM failed because gradients could not be computed.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)

    if max_value <= 0:
        return np.zeros((224, 224), dtype=np.float32)

    heatmap = heatmap / max_value
    return heatmap.numpy()


def overlay_gradcam(image_path, heatmap, alpha=0.40):
    original = cv2.imread(image_path)

    if original is None:
        raise ValueError("Could not read image for Grad-CAM overlay.")

    original = cv2.resize(original, (360, 300))

    heatmap = cv2.resize(heatmap, (360, 300))
    heatmap = np.uint8(255 * heatmap)

    _, thresh = cv2.threshold(heatmap, 120, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxed_image = original.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(boxed_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    heat_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(boxed_image, 1 - alpha, heat_color, alpha, 0)

    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))


# =========================================================
# GUI IMAGE DISPLAY
# =========================================================
def show_image_on_label(image_path, label, width=360, height=300):
    img = Image.open(image_path).convert("RGB")

    img_ratio = img.width / img.height
    frame_ratio = width / height

    if img_ratio > frame_ratio:
        new_width = width
        new_height = int(width / img_ratio)
    else:
        new_height = height
        new_width = int(height * img_ratio)

    img = img.resize((new_width, new_height), Image.LANCZOS)

    background = Image.new("RGB", (width, height), (248, 250, 252))
    offset = ((width - new_width) // 2, (height - new_height) // 2)
    background.paste(img, offset)

    img_display = ImageTk.PhotoImage(background)
    label.config(image=img_display, text="")
    label.image = img_display


def show_pil_image_on_label(pil_img, label, width=360, height=300):
    img = pil_img.resize((width, height), Image.LANCZOS)
    img_display = ImageTk.PhotoImage(img)

    label.config(image=img_display, text="")
    label.image = img_display


# =========================================================
# GUI UPDATE FUNCTIONS
# =========================================================
def update_probability_box(text_widget, probs):
    text_widget.config(state="normal")
    text_widget.delete("1.0", tk.END)

    for i, prob in enumerate(probs):
        label = display_class_name(class_names[i])
        text_widget.insert(tk.END, f"{label:<18} {prob * 100:>6.2f}%\n")

    text_widget.config(state="disabled")


def load_and_predict(image_path):
    if not os.path.exists(image_path):
        messagebox.showerror("File Error", "Image file not found.")
        return

    try:
        status_label.config(text="Analyzing image...")
        root.update_idletasks()

        show_image_on_label(image_path, original_image_label)
        file_name_label.config(text=os.path.basename(image_path))

        results = compare_models(image_path)

        v2_prediction_value.config(text=display_class_name(results["v2_class"]))
        v2_confidence_value.config(text=f"{results['v2_conf']:.2f}%")

        v3_prediction_value.config(text=display_class_name(results["v3_class"]))
        v3_confidence_value.config(text=f"{results['v3_conf']:.2f}%")

        update_probability_box(v2_probs_box, results["v2_probs"])
        update_probability_box(v3_probs_box, results["v3_probs"])

        final_prediction_value.config(text=display_class_name(results["selected_class"]))
        final_confidence_value.config(text=f"{results['selected_conf']:.2f}%")
        selected_model_value.config(text=results["recommended_model"])

        img_array = preprocess_image(image_path)

        heatmap = make_gradcam_heatmap(
            img_array,
            results["selected_model"],
            results["selected_index"]
        )

        gradcam_img = overlay_gradcam(image_path, heatmap)
        show_pil_image_on_label(gradcam_img, gradcam_image_label)

        status_label.config(text="Analysis complete.")

    except Exception as e:
        status_label.config(text="Analysis failed.")
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
if DND_AVAILABLE:
    root = TkinterDnD.Tk()
else:
    root = tk.Tk()

root.title("LumberNet AI - Deployable Wood Defect Detection System")
root.geometry("1280x760")
root.configure(bg="#E5E7EB")
root.resizable(False, False)


# =========================================================
# HEADER
# =========================================================
header = tk.Frame(root, bg="#0F172A", height=82)
header.pack(fill="x")

title = tk.Label(
    header,
    text="LumberNet AI",
    bg="#0F172A",
    fg="white",
    font=("Segoe UI", 26, "bold")
)
title.pack(side="left", padx=28, pady=18)

subtitle = tk.Label(
    header,
    text="Deployable Wood Surface Defect Classification with Grad-CAM Bounding Box",
    bg="#0F172A",
    fg="#CBD5E1",
    font=("Segoe UI", 11)
)
subtitle.pack(side="left", pady=30)

status_label = tk.Label(
    header,
    text="Ready for analysis.",
    bg="#0F172A",
    fg="#38BDF8",
    font=("Segoe UI", 10, "bold")
)
status_label.pack(side="right", padx=28)


# =========================================================
# MAIN CONTAINER
# =========================================================
main = tk.Frame(root, bg="#E5E7EB")
main.pack(fill="both", expand=True, padx=22, pady=18)


# =========================================================
# LEFT PANEL
# =========================================================
left_panel = tk.Frame(main, bg="white", width=410, height=610)
left_panel.pack(side="left", fill="y", padx=(0, 14))
left_panel.pack_propagate(False)

left_title = tk.Label(
    left_panel,
    text="Image Input",
    bg="white",
    fg="#111827",
    font=("Segoe UI", 17, "bold")
)
left_title.pack(anchor="w", padx=20, pady=(18, 4))

left_desc = tk.Label(
    left_panel,
    text="Upload or drag a wood surface image for inspection.",
    bg="white",
    fg="#6B7280",
    font=("Segoe UI", 9)
)
left_desc.pack(anchor="w", padx=20)

drop_area = tk.Label(
    left_panel,
    text="Drop Image Here",
    bg="#F1F5F9",
    fg="#334155",
    relief="ridge",
    borderwidth=2,
    font=("Segoe UI", 13, "bold"),
    height=3
)
drop_area.pack(fill="x", padx=20, pady=14)

if DND_AVAILABLE:
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
    fg="#64748B",
    font=("Segoe UI", 9)
)
file_name_label.pack(pady=10)

preview_title = tk.Label(
    left_panel,
    text="Original Image",
    bg="white",
    fg="#111827",
    font=("Segoe UI", 12, "bold")
)
preview_title.pack(anchor="w", padx=20, pady=(6, 6))

original_frame = tk.Frame(left_panel, bg="#F8FAFC", width=360, height=300)
original_frame.pack(padx=20, pady=5)
original_frame.pack_propagate(False)

original_image_label = tk.Label(
    original_frame,
    text="Image preview",
    bg="#F8FAFC",
    fg="#94A3B8",
    font=("Segoe UI", 11)
)
original_image_label.pack(expand=True)


# =========================================================
# CENTER PANEL
# =========================================================
center_panel = tk.Frame(main, bg="white", width=410, height=610)
center_panel.pack(side="left", fill="y", padx=(0, 14))
center_panel.pack_propagate(False)

center_title = tk.Label(
    center_panel,
    text="Grad-CAM Bounding Box",
    bg="white",
    fg="#111827",
    font=("Segoe UI", 17, "bold")
)
center_title.pack(anchor="w", padx=20, pady=(18, 4))

center_desc = tk.Label(
    center_panel,
    text="Highlighted regions and boxes show where the model focused.",
    bg="white",
    fg="#6B7280",
    font=("Segoe UI", 9)
)
center_desc.pack(anchor="w", padx=20)

gradcam_frame = tk.Frame(center_panel, bg="#F8FAFC", width=360, height=300)
gradcam_frame.pack(padx=20, pady=(22, 10))
gradcam_frame.pack_propagate(False)

gradcam_image_label = tk.Label(
    gradcam_frame,
    text="Grad-CAM bounding box output",
    bg="#F8FAFC",
    fg="#94A3B8",
    font=("Segoe UI", 11)
)
gradcam_image_label.pack(expand=True)

final_card = tk.Frame(center_panel, bg="#EFF6FF", height=175)
final_card.pack(fill="x", padx=20, pady=15)
final_card.pack_propagate(False)

final_title = tk.Label(
    final_card,
    text="Final System Output",
    bg="#EFF6FF",
    fg="#1E3A8A",
    font=("Segoe UI", 14, "bold")
)
final_title.pack(anchor="w", padx=18, pady=(14, 6))

final_prediction_label = tk.Label(
    final_card,
    text="Prediction",
    bg="#EFF6FF",
    fg="#64748B",
    font=("Segoe UI", 9)
)
final_prediction_label.pack(anchor="w", padx=18)

final_prediction_value = tk.Label(
    final_card,
    text="-",
    bg="#EFF6FF",
    fg="#111827",
    font=("Segoe UI", 18, "bold")
)
final_prediction_value.pack(anchor="w", padx=18)

final_confidence_label = tk.Label(
    final_card,
    text="Confidence",
    bg="#EFF6FF",
    fg="#64748B",
    font=("Segoe UI", 9)
)
final_confidence_label.pack(anchor="w", padx=18, pady=(8, 0))

final_confidence_value = tk.Label(
    final_card,
    text="-",
    bg="#EFF6FF",
    fg="#111827",
    font=("Segoe UI", 14, "bold")
)
final_confidence_value.pack(anchor="w", padx=18)

selected_model_label = tk.Label(
    final_card,
    text="Selected Model",
    bg="#EFF6FF",
    fg="#64748B",
    font=("Segoe UI", 9)
)
selected_model_label.place(x=210, y=84)

selected_model_value = tk.Label(
    final_card,
    text="-",
    bg="#EFF6FF",
    fg="#111827",
    font=("Segoe UI", 14, "bold")
)
selected_model_value.place(x=210, y=105)


# =========================================================
# RIGHT PANEL
# =========================================================
right_panel = tk.Frame(main, bg="#E5E7EB")
right_panel.pack(side="right", fill="both", expand=True)

right_title = tk.Label(
    right_panel,
    text="Model Comparison",
    bg="#E5E7EB",
    fg="#111827",
    font=("Segoe UI", 17, "bold")
)
right_title.pack(anchor="w", pady=(0, 8))


def create_model_card(parent, model_name):
    card = tk.Frame(parent, bg="white", height=270)
    card.pack(fill="x", pady=8)
    card.pack_propagate(False)

    model_label = tk.Label(
        card,
        text=model_name,
        bg="white",
        fg="#111827",
        font=("Segoe UI", 15, "bold")
    )
    model_label.pack(anchor="w", padx=18, pady=(14, 8))

    row = tk.Frame(card, bg="white")
    row.pack(fill="x", padx=18)

    pred_box = tk.Frame(row, bg="#F8FAFC", width=145, height=72)
    pred_box.pack(side="left", padx=(0, 10))
    pred_box.pack_propagate(False)

    pred_label = tk.Label(
        pred_box,
        text="Prediction",
        bg="#F8FAFC",
        fg="#64748B",
        font=("Segoe UI", 9)
    )
    pred_label.pack(anchor="w", padx=10, pady=(8, 0))

    pred_value = tk.Label(
        pred_box,
        text="-",
        bg="#F8FAFC",
        fg="#111827",
        font=("Segoe UI", 12, "bold")
    )
    pred_value.pack(anchor="w", padx=10)

    conf_box = tk.Frame(row, bg="#F8FAFC", width=145, height=72)
    conf_box.pack(side="left")
    conf_box.pack_propagate(False)

    conf_label = tk.Label(
        conf_box,
        text="Confidence",
        bg="#F8FAFC",
        fg="#64748B",
        font=("Segoe UI", 9)
    )
    conf_label.pack(anchor="w", padx=10, pady=(8, 0))

    conf_value = tk.Label(
        conf_box,
        text="-",
        bg="#F8FAFC",
        fg="#111827",
        font=("Segoe UI", 12, "bold")
    )
    conf_value.pack(anchor="w", padx=10)

    probs_label = tk.Label(
        card,
        text="Class Probabilities",
        bg="white",
        fg="#374151",
        font=("Segoe UI", 10, "bold")
    )
    probs_label.pack(anchor="w", padx=18, pady=(12, 4))

    probs_box = tk.Text(
        card,
        height=6,
        bg="#F8FAFC",
        fg="#111827",
        relief="flat",
        font=("Consolas", 10)
    )
    probs_box.pack(fill="x", padx=18)
    probs_box.config(state="disabled")

    return pred_value, conf_value, probs_box


v2_prediction_value, v2_confidence_value, v2_probs_box = create_model_card(
    right_panel,
    "MobileNetV2"
)

v3_prediction_value, v3_confidence_value, v3_probs_box = create_model_card(
    right_panel,
    "MobileNetV3"
)


# =========================================================
# FOOTER
# =========================================================
footer = tk.Label(
    root,
    text="Input Size: 224×224 | Models: MobileNetV2 and MobileNetV3 | Explainability: Grad-CAM Bounding Box | Output Classes: Crack, Fuzz, Knots, Natural Patterns",
    bg="#E5E7EB",
    fg="#64748B",
    font=("Segoe UI", 9)
)
footer.pack(pady=(0, 8))


root.mainloop()