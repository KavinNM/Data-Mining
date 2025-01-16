import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
import pickle

# Fungsi untuk memuat model, scaler, dan encoder yang telah disimpan
@st.cache_resource
def load_model_scaler_encoder():
    # Memuat model
    with open('DM-A11.2022.14253-UAS/model1.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    # Memuat scaler
    with open('DM-A11.2022.14253-UAS/scaler1.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    # Memuat encoder (untuk Body_Pain)
    with open('DM-A11.2022.14253-UAS/encoder1.pkl', 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)
    return model, scaler, encoder

# Muat model, scaler, dan encoder
gnb, scaler, encoder = load_model_scaler_encoder()

# Judul aplikasi
st.title("Prediksi Infeksi Dengue dengan Naive Bayes")
st.write("Masukkan nilai untuk setiap fitur di bawah ini, lalu klik tombol 'Prediksi' untuk mendapatkan hasil.")

# Input data dari pengguna
temperature = st.number_input("Temperature (Fahrenheit)", min_value=90.0, max_value=110.0, step=0.1)
platelet_count = st.number_input("Platelet Count", min_value=10000.0, max_value=500000.0, step=100.0)
wbc_count = st.number_input("White Blood Cell Count", min_value=3000.0, max_value=20000.0, step=100.0)
body_pain = st.selectbox("Body Pain (Nyeri Badan)", options=[0, 1, 2])
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
    # One-Hot Encoding untuk Body_Pain
    body_pain_encoded = encoder.transform(input_data[['Body_Pain']])
    encoded_columns = encoder.get_feature_names_out(['Body_Pain'])
    body_pain_df = pd.DataFrame(body_pain_encoded, columns=encoded_columns)

    # Gabungkan hasil encoding dengan data input
    input_data = pd.concat([input_data.drop('Body_Pain', axis=1), body_pain_df], axis=1)

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
