import os
import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from skimage.feature import graycomatrix, graycoprops

# Fungsi untuk menghitung fitur grayscale
def grayscale_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_mean = gray.mean()
    gray_std = gray.std()
    gray_var = gray.var()
    return [gray_mean, gray_std, gray_var]

# Fungsi untuk menghitung fitur RGB
def rgb_features(image):
    # Pisahkan komponen Red, Green, dan Blue
    red = image[:, :, 2]  # BGR to RGB, red is at index 2
    green = image[:, :, 1]  # green is at index 1
    blue = image[:, :, 0]  # blue is at index 0
    
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


# Fungsi untuk ekstraksi fitur GLCM
def glcm_features(image, distances=[1], angles=[0], symmetric=True, normed=True):
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=symmetric, normed=normed)
    contrast = graycoprops(glcm, prop='contrast')
    dissimilarity = graycoprops(glcm, prop='dissimilarity')
    homogeneity = graycoprops(glcm, prop='homogeneity')
    energy = graycoprops(glcm, prop='energy')
    correlation = graycoprops(glcm, prop='correlation')
    return contrast.mean(), dissimilarity.mean(), homogeneity.mean(), energy.mean(), correlation.mean()

# Fungsi untuk ekstraksi fitur dari satu gambar (GLCM, RGB, dan grayscale)
def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return [0] * 17  # Jika gambar tidak valid, return fitur default (5 fitur GLCM + 9 fitur RGB + 3 fitur grayscale)

    # Konversi gambar ke grayscale untuk GLCM dan grayscale fitur
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ekstraksi GLCM
    glcm_feats = glcm_features(gray_img)
    
    # Ekstraksi RGB
    rgb_feats = rgb_features(img)
    
    # Ekstraksi grayscale
    gray_feats = grayscale_features(img)
    
    return list(glcm_feats) + list(rgb_feats) + list(gray_feats)  # Gabungkan fitur GLCM, RGB, dan grayscale


# Fungsi untuk membuat dataset pelatihan
def create_training_data(directory):
    data = []
    labels = []
    
    # Melalui setiap gambar dalam dataset
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):  # Pastikan hanya gambar
                image_path = os.path.join(subdir, file)
                label = subdir.split(os.path.sep)[-1]  # Nama folder sebagai label
                features = extract_features(image_path)
                data.append(features)
                labels.append(label)

    # Membuat DataFrame untuk dataset
    columns = ['glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation',
               'red_mean', 'red_std', 'red_var', 'green_mean', 'green_std', 'green_var',
               'blue_mean', 'blue_std', 'blue_var',
               'gray_mean', 'gray_std', 'gray_var']
    df = pd.DataFrame(data, columns=columns)
    df['label'] = labels
    
    # Simpan ke CSV
    df.to_csv('train_features.csv', index=False)
    print("Training data telah disimpan di 'train_features.csv'.")


# Fungsi untuk membuat dataset pengujian
def create_testing_data(directory):
    data = []
    labels = []
    
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):  # Pastikan hanya gambar
                image_path = os.path.join(subdir, file)
                label = subdir.split(os.path.sep)[-1]
                features = extract_features(image_path)
                data.append(features)
                labels.append(label)

    columns = ['glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation',
               'red_mean', 'red_std', 'red_var', 'green_mean', 'green_std', 'green_var',
               'blue_mean', 'blue_std', 'blue_var',
               'gray_mean', 'gray_std', 'gray_var']
    df = pd.DataFrame(data, columns=columns)
    df['label'] = labels
    
    # Simpan ke CSV
    df.to_csv('data_testing.csv', index=False)
    print("Testing data telah disimpan di 'data_testing.csv'.")


# Fungsi utama untuk pelatihan dan prediksi menggunakan KNN
def knn_classifier():
    # Muat data pelatihan
    train_data = pd.read_csv('train_features.csv')

    # Pisahkan fitur dan label
    X_train = train_data.drop(columns=['label'])
    y_train = train_data['label']

    # Encode label
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Muat data pengujian
    test_data = pd.read_csv('data_testing.csv')

    # Pisahkan fitur pengujian
    X_test = test_data.drop(columns=['label'])
    y_test = test_data['label']
    
    # KNN Model
    k = 3
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train_encoded)

    # Prediksi pada data pengujian
    predictions = knn_model.predict(X_test)

    # Decode hasil prediksi
    predicted_labels = label_encoder.inverse_transform(predictions)

    # Simpan hasil prediksi
    test_data['predicted_label'] = predicted_labels
    test_data.to_csv('test_results.csv', index=False)
    print("Prediksi selesai. Hasil disimpan ke 'test_results.csv'.")


# Jalankan proses
create_training_data('C:/dataset/data_train')
create_testing_data('C:/dataset/data_test')
knn_classifier()


# 3