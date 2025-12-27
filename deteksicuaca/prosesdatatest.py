import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# Fungsi untuk menghitung fitur RGB
def rgb_features(image):
    # Pisahkan komponen RGB
    red = image[:, :, 2]
    green = image[:, :, 1]
    blue = image[:, :, 0]
    
    # Menghitung statistik dasar untuk setiap komponen
    red_mean = red.mean()
    red_std = red.std()
    red_var = red.var()
    
    green_mean = green.mean()
    green_std = green.std()
    green_var = green.var()
    
    blue_mean = blue.mean()
    blue_std = blue.std()
    blue_var = blue.var()

    return [red_mean, red_std, red_var, green_mean, green_std, green_var, blue_mean, blue_std, blue_var]

# Fungsi untuk ekstraksi fitur Gray Level Co-occurrence Matrix (GLCM)
def glcm_features(image, distances=[1], angles=[0], symmetric=True, normed=True):
    # Menghitung GLCM untuk gambar grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, symmetric=symmetric, normed=normed)
    
    # Menghitung beberapa properti dari GLCM
    contrast = graycoprops(glcm, prop='contrast')
    dissimilarity = graycoprops(glcm, prop='dissimilarity')
    homogeneity = graycoprops(glcm, prop='homogeneity')
    energy = graycoprops(glcm, prop='energy')
    correlation = graycoprops(glcm, prop='correlation')
    
    # Rata-rata dari masing-masing properti GLCM
    glcm_features = [
        contrast.mean(),
        dissimilarity.mean(),
        homogeneity.mean(),
        energy.mean(),
        correlation.mean()
    ]
    
    return glcm_features

# Fungsi untuk menghitung fitur dari gambar grayscale
def grayscale_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_mean = gray_image.mean()
    gray_std = gray_image.std()
    gray_var = gray_image.var()
    return [gray_mean, gray_std, gray_var]

# Fungsi untuk ekstraksi fitur dari satu gambar
def extract_glcm_and_rgb_features(image_path):
    img_array = cv2.imread(image_path)
    if img_array is None:
        return None  # Jika gambar tidak valid, return None
    
    # Ekstraksi fitur GLCM
    glcm_feats = glcm_features(img_array)
    
    # Ekstraksi fitur RGB
    rgb_feats = rgb_features(img_array)
    
    # Ekstraksi fitur grayscale
    gray_feats = grayscale_features(img_array)
    
    return glcm_feats + rgb_feats + gray_feats  # Gabungkan fitur GLCM, RGB, dan grayscale


# Fungsi untuk membuat dataset testing
def create_testing_data(directory):
    data = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):  # Pastikan file gambar
                image_path = os.path.join(subdir, file)
                features = extract_glcm_and_rgb_features(image_path)
                if features is not None:
                    label = subdir.split(os.path.sep)[-1]  # Nama folder sebagai label
                    data.append([image_path, label] + list(features))

    # Kolom untuk GLCM, RGB, dan Grayscale
    columns = [
        "filepath", "label", 
        "contrast", "dissimilarity", "homogeneity", "energy", "correlation", 
        "red_mean", "red_std", "red_var", "green_mean", "green_std", "green_var", 
        "blue_mean", "blue_std", "blue_var", 
        "gray_mean", "gray_std", "gray_var"
    ]
    
    # Buat DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Simpan ke CSV
    df.to_csv("data_testing.csv", index=False)
    print("CSV data testing telah dibuat dengan berjumlah 500.")


# Jalankan fungsi
create_testing_data("C:/dataset/data_test")


# 3