from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from skimage.filters import gabor
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.color import rgb2hsv



# Memuat model, encoder, dan scaler
knn_model = joblib.load('knn_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl') 

def predict_view(request):
    try:
        if request.method == 'POST':
            image = request.FILES.get('image')
            # Proses prediksi dengan model Anda
            prediksi = knn_model.predict(input_scaled) # type: ignore
            prediksi_label = label_encoder.inverse_transform(prediksi)[0]  # Ambil prediksi pertama
            
            return JsonResponse({'prediction': prediksi_label})  # Kembalikan JSON dengan prediksi
        else:
            return JsonResponse({'error': 'Hanya menerima metode POST'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Kesalahan Saat Prediksi: {str(e)}'}, status=500)

def find_best_k(X, y):
    """
    Mencari nilai k terbaik untuk KNN menggunakan cross-validation.
    """
    try:
        # Daftar kandidat k yang akan diuji
        k_values = range(1, 21)  # Mencoba nilai k dari 1 hingga 20
        best_k = 1
        best_score = 0

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
            scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')  # 5-fold cross-validation
            avg_score = scores.mean()
            
            if avg_score > best_score:
                best_k = k
                best_score = avg_score

        print(f"Nilai k terbaik: {best_k} dengan akurasi rata-rata: {best_score}")
        return best_k
    except Exception as e:
        print(f"Error in find_best_k: {str(e)}")
        return 5  # Default nilai k jika terjadi error


def extract_features(image):
    """
    Ekstraksi fitur dari gambar menggunakan Gray Level Co-occurrence Matrix (GLCM), RGB, dan Grayscale.
    """
    try:
        # Konversi gambar ke array numpy
        img_array = np.array(image)
        
        # --- Pastikan gambar dikonversi ke grayscale ---
        img_gray = rgb2gray(img_array)
        
        # --- Grayscale Average ---
        grayscale_mean = np.mean(img_gray)  # Rata-rata piksel grayscale
        
        # --- GLCM (Gray Level Co-occurrence Matrix) ---
        glcm = graycomatrix((img_gray * 255).astype('uint8'), distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        glcm_features = np.concatenate([contrast, correlation, energy, homogeneity])
        
        # --- RGB (Red, Green, Blue) ---
        # Extract RGB features (mean of each channel)
        rgb_red_mean = np.mean(img_array[:,:,0])  # Red channel
        rgb_green_mean = np.mean(img_array[:,:,1])  # Green channel
        rgb_blue_mean = np.mean(img_array[:,:,2])  # Blue channel
        rgb_features = np.array([rgb_red_mean, rgb_green_mean, rgb_blue_mean])
        
        # Gabungkan fitur GLCM, RGB, dan Grayscale
        all_features = np.concatenate([glcm_features, rgb_features, [grayscale_mean]])
        
        return all_features
    except Exception as e:
        print(f"Kesalahan dalam ekstraksi fitur: {str(e)}")
        return None



from sklearn.preprocessing import StandardScaler
import joblib

def train_model():
    """
    Melatih model KNN dengan data training menggunakan fitur GLCM dan HSV.
    """
    try:
        # Tentukan direktori file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Baca data pelatihan
        train_features = pd.read_csv(os.path.join(base_dir, 'train_features.csv'))
        test_results = pd.read_csv(os.path.join(base_dir, 'test_results.csv'))
        
        # Persiapkan fitur dan label
        X = train_features.values  # Semua kolom digunakan sebagai fitur
        y = test_results['weather_condition'].values
        
        # Filter label kosong
        valid_indices = ~pd.isna(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Map label ke kategori cuaca
        weather_categories = {
            'berawan gelap': 'Berawan Gelap',
            'berawan': 'Berawan',
            'langit bersih': 'Langit Bersih'
            
        }
        y = np.array([weather_categories.get(str(label).lower(), 'Tidak Dapat Ditentukan') for label in y])
        
        # Normalisasi data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Inisialisasi dan latih model KNN
        knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        knn_model.fit(X_scaled, y)
        
        # Inisialisasi dan latih label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(y)
        
        # Simpan model, scaler, dan encoder
        joblib.dump(knn_model, os.path.join(base_dir, 'knn_model.pkl'))
        joblib.dump(label_encoder, os.path.join(base_dir, 'label_encoder.pkl'))
        joblib.dump(scaler, os.path.join(base_dir, 'scaler.pkl'))
        
        print("Model, encoder, dan scaler telah berhasil disimpan.")
        return knn_model, label_encoder, scaler
    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        return None, None, None



def load_model():
    """
    Memuat model KNN, scaler, dan label encoder yang telah di-training.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'knn_model.pkl')
        encoder_path = os.path.join(base_dir, 'label_encoder.pkl')
        scaler_path = os.path.join(base_dir, 'scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(encoder_path) and os.path.exists(scaler_path):
            knn_model = joblib.load(model_path)
            label_encoder = joblib.load(encoder_path)
            scaler = joblib.load(scaler_path)
            return knn_model, label_encoder, scaler
        else:
            print("Model atau scaler belum tersedia. Melatih ulang model.")
            return train_model()
    except Exception as e:
        print(f"Error in load_model: {str(e)}")
        return train_model()



@ensure_csrf_cookie
def index(request):
    return render(request, 'cuaca/cuaca.html')

@csrf_exempt
def predict(request):
    """
    Endpoint untuk melakukan prediksi cuaca menggunakan fitur dari Gabor Filter, GLCM, Perimeter, Area, dan Hu Moments.
    """
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Baca gambar yang diunggah
            image_file = request.FILES['image']
            image = Image.open(image_file).convert('RGB')
            
            # Ekstrak fitur dari gambar
            features = extract_features(image)
            if features is None:
                return JsonResponse({
                    'success': False,
                    'prediction': 'Tidak Dapat Mendeteksi Fitur Gambar'
                })
            
            # Ubah fitur ke array 2D
            features = np.array(features).reshape(1, -1)
            
            # Muat model, encoder, dan scaler
            knn_model, label_encoder, scaler = load_model()
            if knn_model is None or label_encoder is None or scaler is None:
                return JsonResponse({
                    'success': False,
                    'prediction': 'Kesalahan Pada Model'
                })
            
            # Normalisasi fitur
            features_scaled = scaler.transform(features)
            
            # Prediksi cuaca
            prediction = knn_model.predict(features_scaled)
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            
            return JsonResponse({
                'success': True,
                'prediction': str(predicted_label)
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'prediction': 'Kesalahan: ' + str(e)
            }, status=500)
            
    return JsonResponse({
        'success': False,
        'prediction': 'Request Tidak Valid'
    }, status=400)



# 3