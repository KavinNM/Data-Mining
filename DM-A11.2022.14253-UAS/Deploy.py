import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
import pickle

@st.cache_resource
def load_model_scaler_encoder():
    with open('DM-A11.2022.14253-UAS/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('DM-A11.2022.14253-UAS/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('DM-A11.2022.14253-UAS/encoder.pkl', 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)
    return model, scaler, encoder

gnb, scaler, encoder = load_model_scaler_encoder()

st.title("Prediksi Infeksi Dengue dengan Naive Bayes")
st.write("Masukkan nilai untuk setiap fitur di bawah ini, lalu klik tombol 'Prediksi' untuk mendapatkan hasil.")

temperature = st.slider("Temperature tubuh (Fahrenheit)", 94, 115, 97)
st.write("Temperature tubuh dari Pasien = ", temperature, "Fahrenheit")

temperature = st.slider("Jumlah Platelet", 10000, 500000, 11000)


platelet_count = st.slider("Jumlah Platelet", 10000, 500000, 11000)


wbc_count = st.slider("Jumlah Sel Darah Putih", 3000, 20000, 5000)

body_pain = st.selectbox("Body Pain (Nyeri Badan)", options=[0, 1, 2])
st.write("0 = Tidak ada")
st.write("1 = Nyeri Ringan")
st.write("2 = Nyeri Berat")


rash = st.selectbox("Rash (Ruam Kulit)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
st.write("0 = Tidak ada")
st.write("1 =  Ada")

gender = st.selectbox("Gender (Jenis Kelamin)", options=["Male", "Female"], format_func=lambda x: x)

input_data = pd.DataFrame([{
    'Temperature': temperature,
    'Platelet_Count': platelet_count,
    'White_Blood_Cell_Count': wbc_count,
    'Body_Pain': body_pain,
    'Rash': rash,
    'Gender': 0 if gender == "Female" else 1
}])

if st.button("Prediksi"):
    body_pain_encoded = encoder.transform(input_data[['Body_Pain']])
    encoded_columns = encoder.get_feature_names_out(['Body_Pain'])
    body_pain_df = pd.DataFrame(body_pain_encoded, columns=encoded_columns)

    input_data = pd.concat([input_data.drop('Body_Pain', axis=1), body_pain_df], axis=1)

    input_data_scaled = scaler.transform(input_data)

    prediction = gnb.predict(input_data_scaled)

    st.write("Hasil Prediksi:")
    if prediction[0] == 1:
        st.success("Positif Terinfeksi Dengue")
    else:
        st.error("Negatif, tidak terinfeksi Dengue.")
