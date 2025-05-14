
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif

st.set_page_config(page_title="KNN Phishing Classifier", layout="wide")

st.markdown("""
    <style>
        .stApp {
            background-color: #d4f4dd;
        }
        .css-1v3fvcr, .css-18e3th9 {
            background-color: #d4f4dd !important;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .blue-button button {
            background-color: #0074cc;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("phishing_website_dataset.csv")

df = load_data()

tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 Hasil Pengujian", "🧪 Coba Datamu"])

with tab1:
    st.header("Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini digunakan untuk melakukan **klasifikasi phishing website** menggunakan algoritma **K-Nearest Neighbors (KNN)**.

    Anda dapat melakukan evaluasi model dengan menggunakan metode:
    - **K-Fold Cross Validation**
    - **Split Test**

    Setelah menemukan model terbaik, Anda dapat mencobanya terhadap data baru.
    ---
    """)

with tab2:
    st.header("Evaluasi Model")

    eval_method = st.selectbox("Pilih Metode Evaluasi", ["K-Fold", "Split Test"])

    label_col = st.selectbox("Pilih Kolom Target", df.columns, index=len(df.columns)-1)
    n_features = st.selectbox("Jumlah Fitur Teratas (Information Gain)", [30, 25, 20])
    k_val = st.slider("Nilai k untuk KNN", 1, 15, 3)
    metric = st.selectbox("Metric Jarak", ["euclidean", "manhattan", "chebyshev"])
    k_fold_val = st.slider("Jumlah Fold (Jika pakai K-Fold)", 2, 10, 5)

    X = df.drop(columns=[label_col])
    y = df[label_col]

    
# Pastikan semua data numerik dan tidak ada NaN
X = X.fillna(0)
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

# Jalankan mutual information
mi = mutual_info_classif(X, y)

    top_features = pd.Series(mi, index=X.columns).sort_values(ascending=False).head(n_features).index
    X_selected = X[top_features]

    model = KNeighborsClassifier(n_neighbors=k_val, metric=metric)

    if eval_method == "Split Test":
        st.subheader("🔍 Hasil Evaluasi (Split Test)")
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Akurasi (Split Test): {acc:.4f}")

    else:
        st.subheader("🔁 Hasil Evaluasi (K-Fold)")
        kf = KFold(n_splits=k_fold_val, shuffle=True, random_state=42)
        scores = []
        for fold, (train_index, test_index) in enumerate(kf.split(X_selected), 1):
            X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)
            st.write(f"Fold {fold}: Akurasi = {acc:.4f}")
        st.success(f"Rata-rata Akurasi (K-Fold): {np.mean(scores):.4f}")

with tab3:
    st.header("Prediksi Data Baru")

    eval_mode = st.selectbox("Gunakan Model dari:", ["K-Fold", "Split Test"])
    uploaded_file = st.file_uploader("Upload file CSV data baru", type=["csv"])

    if uploaded_file:
        data_new = pd.read_csv(uploaded_file)
        st.write("📄 Data yang diupload:")
        st.dataframe(data_new.head())

        X_upload = data_new[top_features]

        if eval_mode == "K-Fold":
            model = KNeighborsClassifier(n_neighbors=k_val, metric=metric)
            model.fit(X_selected, y)
        else:
            X_train, _, y_train, _ = train_test_split(X_selected, y, test_size=0.2, random_state=42)
            model = KNeighborsClassifier(n_neighbors=k_val, metric=metric)
            model.fit(X_train, y_train)

        pred = model.predict(X_upload)
        data_new['Prediksi'] = pred
        st.success("✅ Prediksi selesai!")
        st.dataframe(data_new)

        csv = data_new.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil", csv, file_name="hasil_prediksi.csv", mime='text/csv')
