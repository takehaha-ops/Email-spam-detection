import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
import streamlit as st
import joblib

df = pd.read_csv("Email.xls", encoding='latin')
df = df[["Message", "Category"]]
df.loc[:, 'Category'] = df['Category'].replace({"0.0": 0, "1.0": 1})
df.loc[:, 'Category'] = df['Category'].replace({"0": 0, "1": 1})
df = df[df['Category'].isin([0, 1])]
print(df)
st.subheader("examples")
st.write(df.head())


def process(text):
    sentence = [char for char in text if char not in string.punctuation]
    sentence = ''.join(sentence)
    clean = [word for word in sentence.split() if word.lower() not in stopwords.words('english')]
    return clean


with open("rf_model.pkl", "rb") as file, open("vectorizer.pkl", "rb") as vectorizer_file:
    rf_model = joblib.load(file)
    vectorizer = joblib.load(vectorizer_file)

st.subheader("input text")
email_text = st.text_area("content", "")

if st.button("detection"):
    if email_text:
        email_text = vectorizer.transform([email_text])
        email_text = email_text.toarray()
        result = rf_model.predict(email_text)
        # st.write(f"result:{result[0]}")
        if result[0] == 0:
            st.write(f"prediction result: Ham")
        else:
            st.write(f"prediction result: Spam")
    else:
        st.write("input")
