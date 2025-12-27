from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load hasil prediksi
test_results = pd.read_csv('C:/deteksicuaca/deteksicuaca/test_results.csv')

# Label asli dan prediksi
y_true = test_results['label']
y_pred = test_results['predicted_label']

# 1. Hitung akurasi
accuracy = accuracy_score(y_true, y_pred)
print(f"Akurasi: {accuracy:.2f}")

# 2. Tampilkan classification report
report = classification_report(y_true, y_pred)
print("\nClassification Report:\n", report)

# 3. Confusion Matrix
labels = y_true.unique()  # Get the unique labels from the actual labels
conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
