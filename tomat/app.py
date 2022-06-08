import tensorflow as tf  
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=['Dashboard','Uji Kematangan Buah','Tentang Kami'],
        icons=['house','image','people-fill']
    )
if selected == 'Dashboard':
    img = Image.open("logo.png")
    st.image(img, width=300)
    st.write("""# TOMATTASTIC""")
    st.write("""Ketahui Kematangan Buah Tomat Bersama Tomattastic!!""")
    st.write("""### OUR TIM""")
    st.image('indra.PNG')
    st.write("""I GEDE INDRA ARYASA""")
    st.image('nurul.PNG')
    st.write("""NURUL MUJAHIDAH""")
    st.image('fito.PNG')
    st.write("""YOHANES FITO""")
    st.image('phino.PNG')
    st.write("""KRISPHINO SAPUTRA NURBIDIN""")
    st.image('dwii.PNG')
    st.write("""DWI SAHRUL SETIAWAN""")

if selected == 'Uji Kematangan Buah':
        st.write('''
        # Identifikasi Kematangan Buah Tomat''')

        file = st.file_uploader("Silahkan Masukkan Gambar Buah Tomat", type=['jpg','png'])

        def predict_stage(image_data,model):
            size = (150, 150)
            image = ImageOps.fit(image_data,size, Image.ANTIALIAS)
            image_array = np.array(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
            data[0] = normalized_image_array
            preds = ""
            prediction = model.predict(data)
            if np.argmax(prediction)==0:
                st.write(f'fully-ripe')
            elif np.argmax(prediction)==1:
                st.write(f'semi-ripe')
            else :
                st.write(f'unripe')
        
            return prediction
   
        if file is None:
            st.text("Silahkan Masukkan File Gambar")
        else:
            image = Image.open(file)
            st.image(image, use_column_width=True)
            model = tf.keras.models.load_model('ripeness tomato.h5')
            Generate_pred = st.button("Memprediksi Tingkat Kematangan...")
            if Generate_pred:
                prediction = predict_stage(image, model)
                st.text("Probabilitas (0: fully-ripe, 1: semi-ripe, 2: unripe)")
                st.write(prediction)

if selected == 'Tentang Kami':
    if(st.button("Tentang Kami")):
        st.write("Tomattastic dibuat untuk membantu masyarakat dan tentunya untuk industri dalam proses monitoring dan mengidentifikasi tingkat kematangan buah tomat. Website ini merupakan project tugas akhir kami selama belajar di Orbit Future Academy dalam program Foundation of AI and Life Skills for Gen-Z. Kelompok kami terdiri dari 5 orang yang berasal dari perguruan tinggi yang berbeda. Dengan adanya website ini diharapkan dapat mempermudah masyarakat dan industri untuk mendapatkan hasil identifikasi buah tomat secara akurat.  ")
        st.balloons()


    