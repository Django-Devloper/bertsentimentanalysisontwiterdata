import streamlit as st
from transformers import  pipeline

# Streamlit app layout
st.title("Sentiment Analysis with BERT")
st.write("Enter text below to analyze its sentiment.")

# Input text from the user
user_input = st.text_area("Input Text", placeholder="Type your text here...")

# Predict sentiment
if st.button("Analyze Sentiment"):
    if user_input.strip():
        classifier = pipeline('text-classification', model='bert-base-case-uncased-sentiment-model')
        prediction = classifier(user_input)
        st.write(f"Sentiment: {prediction[0]['label']}")
        st.write(f"Confidence: {prediction[0]['score']* 100:.2f}%")
    else:
        st.warning("Please enter some text to analyze.")
