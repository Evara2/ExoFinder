import io
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv('API_URL', 'http://localhost:8000').rstrip('/')
st.set_page_config(page_title='ExoFinder', page_icon='🪐', layout='wide')
st.title('🪐 ExoFinder — Stellar Transit Candidate Detection')
st.caption('Research demonstration; predictions are not confirmed exoplanet discoveries.')
try:
    info = requests.get(f'{API_URL}/model-info', timeout=15).json()
    st.write(f"Model: {info['architecture']} | Classification threshold: {info['threshold']:.2f}")
except requests.RequestException as exc:
    st.error(f'API unavailable: {exc}')
    st.stop()

uploaded = st.file_uploader('Upload Kepler exoTest/exoTrain format CSV', type='csv')
if uploaded:
    contents = uploaded.getvalue()
    try:
        df = pd.read_csv(io.BytesIO(contents))
        if len(df) == 0 or len(df) > 1000:
            st.error('CSV must contain 1–1000 rows.')
            st.stop()
        st.write(f'{len(df)} light curves uploaded')
        row = st.number_input('Light curve to visualize (zero-based index)', min_value=0, max_value=len(df)-1, value=0)
        flux = df.filter(regex=r'^FLUX\.\d+$').iloc[int(row)]
        st.line_chart(flux.reset_index(drop=True))
        if st.button('Analyze light curves', type='primary'):
            with st.spinner('Running CNN–Transformer inference...'):
                response = requests.post(f'{API_URL}/predict-csv', files={'file': ('curves.csv', contents, 'text/csv')}, timeout=180)
                response.raise_for_status()
                predictions = response.json()['predictions']
            result = pd.DataFrame(predictions)
            st.dataframe(result, use_container_width=True)
            st.metric('Candidates flagged', int(result['candidate'].sum()))
            st.download_button('Download predictions', result.to_csv(index=False), 'exofinder_predictions.csv', 'text/csv')
    except (ValueError, requests.RequestException) as exc:
        st.error(str(exc))
