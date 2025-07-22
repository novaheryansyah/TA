# gui_predict.py

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load model YOLOv8 klasifikasi
model = YOLO("beras_clf/yolov8-cls/weights/best.pt")

# GUI App
root = tk.Tk()
root.title("Klasifikasi Varietas Beras")
root.geometry("400x500")

img_label = tk.Label(root)
img_label.pack(pady=10)

result_label = tk.Label(root, text="Pilih gambar untuk prediksi", font=("Arial", 14))
result_label.pack(pady=10)

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        img = Image.open(file_path).resize((224, 224))
        tk_img = ImageTk.PhotoImage(img)
        img_label.config(image=tk_img)
        img_label.image = tk_img

        # Prediksi
        result = model(file_path)[0]
        kelas = model.names[result.probs.top1]
        skor = result.probs.data[result.probs.top1].item()
        result_label.config(text=f"Prediksi: {kelas}\nConfidence: {skor:.2%}")

btn = tk.Button(root, text="Pilih Gambar", command=open_image, font=("Arial", 12))
btn.pack(pady=20)

root.mainloop()
