# model_evaluation.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ====== 1. Langsung baca hasil dari training (tanpa val ulang) ======
save_dir = "runs/classify/beras_clf/yolov8-cls"
results_path = os.path.join(save_dir, "results.csv")
confusion_path = os.path.join(save_dir, "confusion_matrix.png")

# ====== 2. Ambil Akurasi Terakhir dari CSV ======
if os.path.exists(results_path):
    df = pd.read_csv(results_path)
    acc_col = next((col for col in df.columns if "accuracy" in col), None)

    if acc_col:
        last_acc = df[acc_col].iloc[-1] * 100
        print(f"Top-1 Accuracy (final): {last_acc:.2f}%")
    else:
        print("❌ Kolom akurasi tidak ditemukan di results.csv")
else:
    print("❌ results.csv tidak ditemukan")

# ====== 3. Tampilkan Confusion Matrix Jika Ada ======
if os.path.exists(confusion_path):
    img = Image.open(confusion_path)
    img.show(title="Confusion Matrix")
else:
    print("❌ Confusion matrix tidak ditemukan.")

# ====== 4. Visualisasi Grafik Accuracy & Loss ======
if os.path.exists(results_path):
    df = pd.read_csv(results_path)
    plt.figure(figsize=(10, 5))

    acc_col = next((col for col in df.columns if "accuracy" in col), None)
    if acc_col:
        plt.subplot(1, 2, 1)
        plt.plot(df[acc_col], label="Top-1 Accuracy")
        plt.title("Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.legend()

    if '              train/cls_loss' in df.columns:
        plt.subplot(1, 2, 2)
        plt.plot(df['              train/cls_loss'], label="Train Loss")
        plt.title("Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()

    plt.tight_layout()
    plt.show()