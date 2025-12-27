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

# Fungsi untuk menghitung fitur Gray Level Co-occurrence Matrix (GLCM)
def glcm_features(image, distances=[1], angles=[0]):
    # Menghitung GLCM untuk gambar grayscale
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)
    
    # Menghitung beberapa properti GLCM
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    ASM = graycoprops(glcm, 'ASM')  # Angular Second Moment (ASM)
    
    # Rata-rata dari semua jarak dan sudut
    contrast_mean = contrast.mean()
    dissimilarity_mean = dissimilarity.mean()
    homogeneity_mean = homogeneity.mean()
    energy_mean = energy.mean()
    correlation_mean = correlation.mean()
    ASM_mean = ASM.mean()

    return [contrast_mean, dissimilarity_mean, homogeneity_mean, energy_mean, correlation_mean, ASM_mean]

# Fungsi untuk menghitung fitur dari gambar grayscale
def grayscale_features(image):
    gray_mean = image.mean()
    gray_std = image.std()
    gray_var = image.var()
    return [gray_mean, gray_std, gray_var]

# Path ke folder data training
data_train_path = r'C:\dataset\data_train'

# List untuk menyimpan data
data = []

# Ekstraksi fitur
for label in os.listdir(data_train_path):
    folder_path = os.path.join(data_train_path, label)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            img = cv2.imread(file_path)
            if img is not None:
                # Konversi ke grayscale dan ekstrak fitur
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Ekstraksi fitur GLCM
                glcm_features_list = glcm_features(gray_img)
                
                # Ekstraksi fitur RGB
                rgb_features_list = rgb_features(img)
                
                # Ekstraksi fitur Grayscale
                grayscale_features_list = grayscale_features(gray_img)
                
                # Tambahkan data ke list
                data.append(
                    [file_path, label] + glcm_features_list + rgb_features_list + grayscale_features_list
                )

# Kolom untuk GLCM features, RGB features, dan Grayscale features
glcm_columns = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
rgb_columns = ['red_mean', 'red_std', 'red_var', 'green_mean', 'green_std', 'green_var', 'blue_mean', 'blue_std', 'blue_var']
grayscale_columns = ['gray_mean', 'gray_std', 'gray_var']

# Buat DataFrame
columns = ['filepath', 'label'] + glcm_columns + rgb_columns + grayscale_columns
df = pd.DataFrame(data, columns=columns)

# Simpan DataFrame ke CSV
df.to_csv('train_features.csv', index=False)
print("CSV data training telah dibuat dengan berjumlah 2000")


# 3