import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

model_ciberbully = pickle.load(open('model_ciberbully.sav','rb'))


tfidf = TfidfVectorizer
loaded_vec = TfidfVectorizer(decode_error="replace",vocabulary=set(pickle.load(open("new_selected_feature_tf-idf.sav","rb"))))

st.title('Prediksi Komentar Cyberbully')
clean_teks = st.text_input('Masukan Text Komentar')

ciberbully_detection = ''

if st.button('Hasil Deteksi'):
    predict_ciberbully = model_ciberbully.predict(loaded_vec.fit_transform([clean_teks]))

    if (predict_ciberbully == 0):
        ciberbully_detection = 'Komentar bersifat Bullying'
    else:
        ciberbully_detection = 'Komentar bersifat Non Bullying'

st.success(ciberbully_detection)