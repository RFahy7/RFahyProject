import streamlit as st
import numpy as np
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

# Download any required NLTK data
download('stopwords')
download('vader_lexicon')
download('wordnet')

# Load the model and vectorizer
GradientBoostingRegressor = joblib.load('gmb.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Define text preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(headline):
    headline = ''.join([c for c in headline if c not in ('!', '.', ':', ',', '?', '(', ')', '"')])
    headline = ' '.join([lemmatizer.lemmatize(word) for word in headline.split() if word not in stop_words])
    return headline


# Define the Streamlit app
def main():
    st.title("Headline Sentiment Prediction")

    st.header("Enter a News Headline")
    user_input = st.text_input("Headline", "Type your headline here then click PREDICT SENTIMENT...")

    if st.button("PREDICT SENTIMENT"):
        # Preprocess the input headline
        processed_input = preprocess_text(user_input)

        # Transform the input text to the TF-IDF vector
        input_vector = vectorizer.transform([processed_input])

        # Make prediction using the linear regression model
        sentiment_prediction = GradientBoostingRegressor.predict(input_vector)

        st.write(f"Predicted Sentiment Score: {sentiment_prediction[0]:.2f}")


if __name__ == "__main__":
    main()
