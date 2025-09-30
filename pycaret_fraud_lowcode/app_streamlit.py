import streamlit as st, pandas as pd
from pycaret.classification import load_model, predict_model
st.set_page_config(page_title='Fraud Detection (PyCaret)', layout='wide')
st.title('Credit Card Fraud Detection — PyCaret')
model_path = st.text_input('Model base path', value='models/pycaret_fraud_v1')
file = st.file_uploader('Upload CSV', type=['csv'])
if file:
    df = pd.read_csv(file); st.write('Input sample:', df.head())
    model = load_model(model_path); preds = predict_model(model, data=df)
    st.write('Predictions:', preds.head()); st.download_button('Download predictions', preds.to_csv(index=False), 'predictions.csv')
