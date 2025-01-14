import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import pickle

# Fungsi untuk memuat model dan scaler yang telah disimpan
@st.cache_resource
def load_model_and_scaler():
    # Pastikan file model dan scaler tersimpan di folder yang sama
    with open('DM-A11.2022.14253-UAS/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('DM-A11.2022.14253-UAS/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# Muat model dan scaler
gnb, scaler = load_model_and_scaler()

# Judul aplikasi
st.title("Prediksi Infeksi Dengue dengan Naive Bayes")
st.write("Masukkan nilai untuk setiap fitur di bawah ini, lalu klik tombol 'Prediksi' untuk mendapatkan hasil.")

# Input data dari pengguna
temperature = st.number_input("Temperature (Fahrenheit)", min_value=90.0, max_value=110.0, step=0.1)
platelet_count = st.number_input("Platelet Count", min_value=10000.0, max_value=500000.0, step=100.0)
wbc_count = st.number_input("White Blood Cell Count", min_value=3000.0, max_value=20000.0, step=100.0)
body_pain = st.selectbox("Body Pain (Nyeri Badan)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
rash = st.selectbox("Rash (Ruam Kulit)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
gender = st.selectbox("Gender (Jenis Kelamin)", options=["Male", "Female"], format_func=lambda x: x)

# Konversi input menjadi DataFrame
input_data = pd.DataFrame([{
    'Temperature': temperature,
    'Platelet_Count': platelet_count,
    'White_Blood_Cell_Count': wbc_count,
    'Body_Pain': body_pain,
    'Rash': rash,
    'Gender': 0 if gender == "Female" else 1
}])

# Tombol untuk prediksi
if st.button("Prediksi"):
    # Preprocessing: Scaling data numerik
    input_data_scaled = scaler.transform(input_data)
    
    # Prediksi dengan model
    prediction = gnb.predict(input_data_scaled)
    
    # Tampilkan hasil prediksi
    st.write("Hasil Prediksi:")
    if prediction[0] == 1:
        st.success("Positif Terinfeksi Dengue!")
    else:
        st.error("Negatif, tidak terinfeksi Dengue.")

