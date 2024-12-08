import streamlit as st
import numpy as np
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor  # Import the model if needed for context

# Download any required NLTK data
download('stopwords')
download('vader_lexicon')
download('wordnet')

# Load the Gradient Boosting model and TF-IDF vectorizer
linear_regressor = joblib.load('linead_regressor.joblib')
gradient_booster = joblib.load('gmb.joblib')
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
    user_input = st.text_input("Headline", "Type your headline here then press Predict Sentiment")

    if st.button("Predict Sentiment"):
        # Preprocess the input headline
        processed_input = preprocess_text(user_input)

        # Transform the input text to the TF-IDF vector
        input_vector = vectorizer.transform([processed_input])

        # Make predictions
        sentiment_prediction1 = linear_regressor.predict(input_vector)
        st.write(f"Using Linear Regressor...Predicted Sentiment Score: {sentiment_prediction1[0]:.2f}")
        sentiment_prediction2 = gradient_booster.predict(input_vector)
        st.write(f"Using Graident Booster...Predicted Sentiment Score: {sentiment_prediction2[0]:.2f}")
        
        
if __name__ == "__main__":
    main()
