import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load Model and Vectorizer
model = joblib.load('news_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Map category numbers to names
category_map = {
    1: 'World',
    2: 'Sports',
    3: 'Business',
    4: 'Sci/Tech'
}

# Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("🗞️ News Category Predictor")
st.write("Paste your news article below and predict its category!")

user_input = st.text_area("Enter News Text Here:", height=200)

if st.button("Predict"):
    if user_input.strip() != "":
        processed_text = preprocess(user_input)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]
        category = category_map.get(int(prediction), "Unknown")
        st.success(f"The news belongs to **{category}** category.")
    else:
        st.warning("Please enter some text to predict.")

st.markdown("---")
st.markdown("Created with ❤️ using Streamlit")
