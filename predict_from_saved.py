import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Prepare text preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Predict sentiment for user input
while True:
    print("\nType a review (or type 'exit' to stop):")
    user_input = input(">>> ").strip()

    if user_input.lower() == 'exit':
        break

    if not user_input:
        print("⚠️ Please enter a valid sentence.")
        continue

    cleaned_input = preprocess_text(user_input)
    vectorized_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vectorized_input)[0]

    if prediction == 1:
        print("🟢 Sentiment: POSITIVE")
    else:
        print("🔴 Sentiment: NEGATIVE")
