import numpy as np
import cv2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(image_path):
    """
    Mengambil fitur GLCM dari gambar.
    Fitur yang diambil: contrast, correlation, energy, homogeneity.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Warning: Gambar tidak dapat dibaca di path {image_path}")
        return None  # Mengembalikan None jika gambar tidak valid

    # Resize gambar untuk konsistensi ukuran (opsional)
    img_resized = cv2.resize(img, (128, 128))
    
    # Ekstraksi fitur GLCM
    glcm = graycomatrix(img_resized, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    return [contrast, correlation, energy, homogeneity]

def extract_rgb_features(image_path):
    """
    Mengambil fitur RGB dari gambar.
    Fitur yang diambil: rata-rata nilai Red, Green, Blue, dan Grayscale.
    """
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Warning: Gambar tidak dapat dibaca di path {image_path}")
        return None
    
    # Resize gambar untuk konsistensi ukuran (opsional)
    img_resized = cv2.resize(img, (128, 128))
    
    # Menghitung rata-rata nilai Red, Green, Blue
    r_mean = np.mean(img_resized[:, :, 2])  # Red
    g_mean = np.mean(img_resized[:, :, 1])  # Green
    b_mean = np.mean(img_resized[:, :, 0])  # Blue

    # Menghitung rata-rata nilai Grayscale
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray_mean = np.mean(gray_img)
    
    return [r_mean, g_mean, b_mean, gray_mean]


def load_data_from_directory(base_dir):
    """
    Memuat gambar dan label dari setiap kategori dalam folder dan mengekstrak fitur GLCM, RGB, dan Grayscale.
    """
    X = []
    y = []
    
    categories = ['Langit Bersih', 'Berawan', 'Berawan Gelap']  # Daftar kategori cuaca
    
    for category in categories:
        category_path = os.path.join(base_dir, category)
        
        if not os.path.exists(category_path):
            print(f"Folder '{category_path}' tidak ditemukan!")
            continue
        
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            glcm_features = extract_glcm_features(img_path)
            rgb_features = extract_rgb_features(img_path)
            
            if glcm_features is None or rgb_features is None:
                continue
            
            # Gabungkan fitur GLCM, RGB, dan Grayscale
            features = glcm_features + rgb_features
            X.append(features)
            y.append(category)
    
    X = np.array(X)
    y = np.array(y)
    return X, y


# Path ke direktori dataset
base_dir = r'C:/dataset/data_train'

# Memuat data gambar dan fitur GLCM dan RGB
X, y = load_data_from_directory(base_dir)

# Encode label
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Membagi data menjadi training dan validation set
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Cross-validation untuk mencari nilai k terbaik
k_values = range(1, 21)
best_k = None
best_score = 0



for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, X_train, y_train, cv=5, scoring='accuracy')
    mean_score = scores.mean()
    print(f"Nilai k={k}, Skor Rata-rata={mean_score}")
    
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"\nNilai k terbaik: {best_k} dengan skor {best_score}")

# Normalisasi data fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Print hasil normalisasi
print("Contoh hasil normalisasi data training (5 data pertama):")
print(X_train_scaled[:5])  # Menampilkan 5 baris pertama dari data training yang telah dinormalisasi

print("\nContoh hasil normalisasi data validation (5 data pertama):")
print(X_val_scaled[:5])  # Menampilkan 5 baris pertama dari data validation yang telah dinormalisasi

# Melatih model KNN dengan k terbaik
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)

# Menyimpan model, encoder, dan scaler ke file .pkl
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model (RGB + Grayscale + GLCM), encoder, dan scaler telah disimpan.")



# 3