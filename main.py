import streamlit as st
import joblib
import pandas as pd

@st.cache(allow_output_mutation=True)
def load(model_path):
    df = pd.read_pickle(model_path)
    return df

def inference(df, name):
    similar_to_movie = df.corrwith(df[name])
    similar_to_movie = pd.DataFrame(similar_to_movie,columns=['Correlation'])
    similar_to_movie = similar_to_movie.sort_values(by = 'Correlation', ascending = False)
    return similar_to_movie

st.title('Anime Recommendation system')
st.write('Sistem rekomendasi Anime')
dataframe = load('models/df.zip')

option = st.selectbox('Pilih anime favorit anda', (dataframe.columns))

st.write('Rekomendasi anime mirip :', option)

if (st.button('Cari Rekomendasi')):
    # dataframe = load('../models/df.pkl')
    result = inference(dataframe,option)
    st.write(result.head(20))

