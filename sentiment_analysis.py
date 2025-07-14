# Step 1: Import Libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')



# Download NLTK data
nltk.download('stopwords')

# Step 2: Create Dataset
data = {
    'text': [
        "I absolutely loved the movie! It was fantastic.",
        "What a waste of time. The movie was terrible.",
        "The food was delicious and the service was excellent.",
        "I am never buying from this store again. Awful experience.",
        "Great product, works exactly as expected!",
        "Very disappointing. It broke on the first day.",
        "Such a heartwarming story, I cried at the end!",
        "The phone is too slow and freezes all the time.",
        "Amazing performance by the lead actor!",
        "Worst customer service ever. Rude and unhelpful."
        "Good,Mine,Positive."
    ],
    'label': [
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
        'negative'
    ]
}

# Step 3: Convert to DataFrame
df = pd.DataFrame(data)

# Step 4: Text Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation and numbers
    tokens = text.split()  # simple split instead of word_tokenize
    filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Step 5: Main Execution
if __name__ == "__main__":
    # Apply preprocessing
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # Print original and cleaned data
    print("Original & Cleaned Dataset:\n")
    print(df[['text', 'cleaned_text', 'label']])

    # Step 6: TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label'].map({'positive': 1, 'negative': 0})  # encode labels

    # Step 7: Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 8: Train Model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Step 9: Predictions
    y_pred = model.predict(X_test)

    # Step 10: Evaluation
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    # Step 11: Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
       # Step 12: Predict on custom input
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
                # Step 13: Save Model and Vectorizer
    joblib.dump(model, "sentiment_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("\n✅ Model and vectorizer saved successfully.")

