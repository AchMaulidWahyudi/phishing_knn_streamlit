
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
            background-color: #73946B;
        }
        .css-1v3fvcr, .css-18e3th9 {
            background-color: #73946B !important;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .blue-button button {
            background-color: #73946B;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        data = pd.read_csv("phishing_website_dataset.csv")
        if data.isnull().any().any():
            st.warning("Warning: Dataset contains missing values. Rows with missing values will be dropped.")
            data = data.dropna()
        return data
    except FileNotFoundError:
        st.error("Error: Dataset file 'phishing_website_dataset.csv' not found in the working directory.")
        return pd.DataFrame()  # Return empty DataFrame to avoid further errors
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

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

    if df.empty:
        st.error("Dataset tidak tersedia atau kosong. Silakan periksa file dataset.")
    else:
        eval_method = st.selectbox("Pilih Metode Evaluasi", ["K-Fold", "Split Test"])

        label_col = st.selectbox("Pilih Kolom Target", df.columns, index=len(df.columns)-1)
        n_features = st.selectbox("Jumlah Fitur Teratas (Information Gain)", [30, 25, 20])
        k_val = st.slider("Nilai k untuk KNN", 1, 15, 3)
        metric = st.selectbox("Metric Jarak", ["euclidean", "manhattan", "chebyshev"])
        k_fold_val = st.slider("Jumlah Fold (Jika pakai K-Fold)", 2, 10, 5)

        X = df.drop(columns=[label_col])
        y = df[label_col]

        try:
            mi = mutual_info_classif(X, y)
            top_features = pd.Series(mi, index=X.columns).sort_values(ascending=False).head(n_features).index
            X_selected = X[top_features]
        except Exception as e:
            st.error(f"Error during feature selection: {e}")
            top_features = []
            X_selected = pd.DataFrame()

        model = KNeighborsClassifier(n_neighbors=k_val, metric=metric)

        if eval_method == "Split Test":
            st.subheader("🔍 Hasil Evaluasi (Split Test)")
            try:
                X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"Akurasi (Split Test): {acc:.4f}")
            except Exception as e:
                st.error(f"Error during Split Test evaluation: {e}")

        else:
            st.subheader("🔁 Hasil Evaluasi (K-Fold)")
            try:
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
            except Exception as e:
                st.error(f"Error during K-Fold evaluation: {e}")

with tab3:
    st.header("Prediksi Data Baru")

    if df.empty:
        st.error("Dataset tidak tersedia atau kosong. Tidak dapat melakukan prediksi.")
    else:
        eval_mode = st.selectbox("Gunakan Model dari:", ["K-Fold", "Split Test"])
        uploaded_file = st.file_uploader("Upload file CSV data baru", type=["csv"])

        # Initialize top_features to empty list to avoid undefined error
        top_features = []

        # We need to define top_features and model parameters consistently with tab2 selections
        # To do this, we replicate the selections with default values or use session state if needed
        # For simplicity, we use default values here
        label_col = df.columns[-1]
        n_features = 30
        k_val = 3
        metric = "euclidean"

        try:
            X = df.drop(columns=[label_col])
            y = df[label_col]
            mi = mutual_info_classif(X, y)
            top_features = pd.Series(mi, index=X.columns).sort_values(ascending=False).head(n_features).index
            X_selected = X[top_features]
        except Exception as e:
            st.error(f"Error during feature selection: {e}")
            top_features = []
            X_selected = pd.DataFrame()

        if uploaded_file:
            try:
                data_new = pd.read_csv(uploaded_file)
                st.write("📄 Data yang diupload:")
                st.dataframe(data_new.head())

                missing_features = [feat for feat in top_features if feat not in data_new.columns]
                if missing_features:
                    st.error(f"Data yang diupload tidak memiliki kolom fitur yang diperlukan: {missing_features}")
                else:
                    X_upload = data_new[top_features]

                    model = KNeighborsClassifier(n_neighbors=k_val, metric=metric)
                    if eval_mode == "K-Fold":
                        model.fit(X_selected, y)
                    else:
                        X_train, _, y_train, _ = train_test_split(X_selected, y, test_size=0.2, random_state=42)
                        model.fit(X_train, y_train)

                    pred = model.predict(X_upload)
                    data_new['Prediksi'] = pred
                    st.success("✅ Prediksi selesai!")
                    st.dataframe(data_new)

                    csv = data_new.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Hasil", csv, file_name="hasil_prediksi.csv", mime='text/csv')
            except Exception as e:
                st.error(f"Error saat memproses file yang diupload atau melakukan prediksi: {e}")
