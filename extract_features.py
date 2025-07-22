# extract_features.py
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops

DATASET_PATH = 'dataset_beras'
OUTPUT_CSV = 'fitur_beras.csv'

def extract_features_from_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas, widths, heights = [], [], []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 50:
            areas.append(w * h)
            widths.append(w)
            heights.append(h)

    avg_area = np.mean(areas) if areas else 0
    avg_w = np.mean(widths) if widths else 0
    avg_h = np.mean(heights) if heights else 0

    mean_color = cv2.mean(img)[:3]
    mean_r, mean_g, mean_b = mean_color[2], mean_color[1], mean_color[0]

    glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

    return {
        'avg_width': avg_w,
        'avg_height': avg_h,
        'avg_area': avg_area,
        'mean_R': mean_r,
        'mean_G': mean_g,
        'mean_B': mean_b,
        'texture_contrast': contrast,
        'texture_homogeneity': homogeneity,
        'texture_entropy': entropy
    }

data = []
for varietas in os.listdir(DATASET_PATH):
    path_var = os.path.join(DATASET_PATH, varietas)
    if not os.path.isdir(path_var):
        continue
    for img_name in tqdm(os.listdir(path_var), desc=varietas):
        img_path = os.path.join(path_var, img_name)
        f = extract_features_from_image(img_path)
        if f:
            f['varietas'] = varietas
            data.append(f)

df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✔ Fitur disimpan di: {OUTPUT_CSV}")
