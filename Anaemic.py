import pickle
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Membaca model
Anaemic_model = pickle.load(open('Anaemic_prediksi.sav', 'rb'))

# Judul web
st.title('Prediksi Anemia')

# Input data dengan validasi
st.sidebar.header('Input Data')
def user_input_features():
    Redpixel = st.sidebar.text_input('Redpixel', '0.0')
    Greenpixel = st.sidebar.text_input('Greenpixel', '0.0')
    Bluepixel = st.sidebar.text_input('Bluepixel', '0.0')
    Hb = st.sidebar.text_input('Hb', '0.0')
    data = {
        'Redpixel': float(Redpixel),
        'Greenpixel': float(Greenpixel),
        'Bluepixel': float(Bluepixel),
        'Hb': float(Hb)
    }
    return data

data = user_input_features()

# Menampilkan data input
st.subheader('Input Data')
st.write(data)

if st.button('Prediksi'):
    try:
        # Konversi input menjadi numerik
        inputs = np.array([[data['Redpixel'], data['Greenpixel'], data['Bluepixel'], data['Hb']]])
        
        # Lakukan prediksi
        pokemon_prediksi = Anaemic_model.predict(inputs)
        probabilities = Anaemic_model.predict_proba(inputs)  # Mendapatkan probabilitas
        
        # Menampilkan hasil prediksi
        st.subheader('Hasil Prediksi')
        if pokemon_prediksi[0] == 0:
            st.write('**Pasien tidak terkena anemia**')
        else:
            st.write('**Pasien terkena anemia**')
        
        
        # Visualisasi probabilitas
        fig, ax = plt.subplots()
        classes = ['Tidak Terkena Anemia', 'Terkena Anemia']
        ax.bar(classes, probabilities[0])
        ax.set_ylabel('Probabilitas')
        ax.set_title('Probabilitas Prediksi')
        st.pyplot(fig)
    
    except ValueError:
        st.error("Pastikan semua input diisi dengan angka yang valid.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
