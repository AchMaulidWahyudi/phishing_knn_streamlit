import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif

# ====================== SETUP HALAMAN ======================
st.set_page_config(page_title="Aplikasi Deteksi Website Phishing", layout="wide")

st.title('🔐 Aplikasi Deteksi Website Phishing')
st.write("### Klasifikasi website phishing menggunakan algoritma K-Nearest Neighbors (KNN)")

# ====================== LOAD DATA ======================
df = pd.read_csv("phishing_website_dataset.csv")
st.header("1. Tentang Dataset")
st.write("Dataset ini digunakan untuk mengklasifikasikan website apakah phishing atau bukan.")
st.write("Jumlah data:", df.shape)
st.dataframe(df.head())

# Tentukan kolom target
label_col = "Result" if "Result" in df.columns else df.columns[-1]

# Pisahkan fitur dan label
X = df.drop(columns=[label_col])
y = df[label_col]

# Pastikan data numerik
X = X.fillna(0)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Seleksi fitur terbaik (top 30)
mi = mutual_info_classif(X, y)
top_features = pd.Series(mi, index=X.columns).sort_values(ascending=False).head(30).index
X_selected = X[top_features]

# ====================== SIDEBAR - PARAMETER ======================
st.sidebar.header("⚙️ Pengaturan KNN")
k = st.sidebar.slider("Nilai k (tetangga terdekat)", 1, 15, 3)
metric = st.sidebar.selectbox("Jarak", ["euclidean", "manhattan", "chebyshev"])
eval_method = st.sidebar.radio("Metode Evaluasi", ["Split Test", "K-Fold"])
n_splits = st.sidebar.slider("Jumlah Fold (K-Fold)", 2, 10, 5)

# ====================== EVALUASI MODEL ======================
st.header("2. Evaluasi Model")

model = KNeighborsClassifier(n_neighbors=k, metric=metric)

if eval_method == "Split Test":
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("📋 Hasil Evaluasi (Split Test)")
    st.write(f"- Akurasi: **{acc:.4f}**")
    st.write(f"- Presisi: **{prec:.4f}**")
    st.write(f"- Recall: **{rec:.4f}**")
    st.write(f"- F1-Score: **{f1:.4f}**")
    st.write("Confusion Matrix:")
    st.dataframe(pd.DataFrame(cm))

else:
    st.subheader("📋 Hasil Evaluasi (K-Fold Cross Validation)")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in kf.split(X_selected):
        X_train_k, X_test_k = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train_k, y_test_k = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train_k, y_train_k)
        y_pred_k = model.predict(X_test_k)
        scores.append(accuracy_score(y_test_k, y_pred_k))

    st.write("Akurasi Tiap Fold:", scores)
    st.success(f"Rata-rata Akurasi: **{np.mean(scores):.4f}**")

    # Visualisasi bar chart
    fig = plt.figure()
    plt.bar(range(1, len(scores) + 1), scores, color='skyblue')
    plt.xlabel("Fold ke-")
    plt.ylabel("Akurasi")
    plt.title("Akurasi Tiap Fold")
    st.pyplot(fig)

# ====================== PREDIKSI DATA BARU ======================
st.header("3. Prediksi Data Baru")

st.write("Masukkan nilai fitur untuk melakukan prediksi apakah website termasuk phishing atau bukan.")

user_input = {}
for feature in top_features:
    user_input[feature] = st.sidebar.number_input(f"{feature}", value=0)

if st.sidebar.button("Prediksi"):
    input_array = np.array([list(user_input.values())])
    model.fit(X_selected, y)
    prediction = model.predict(input_array)[0]

    if prediction == 1:
        st.sidebar.write("### ⚠️ Website ini diprediksi sebagai **Phishing**")
    else:
        st.sidebar.write("### ✅ Website ini diprediksi **Aman**")
