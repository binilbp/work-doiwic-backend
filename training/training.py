import os
import kagglehub
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Download and locate the real dataset
print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("praveengovi/emotions-dataset-for-nlp")
print(f"Dataset downloaded to: {path}")

# This specific Kaggle dataset provides pre-split text files using a semicolon separator
train_file = os.path.join(path, "train.txt")
test_file = os.path.join(path, "test.txt")

print("\n1. Loading real dataset from text files...")
df_train = pd.read_csv(train_file, sep=';', names=['text', 'emotion'])
df_test = pd.read_csv(test_file, sep=';', names=['text', 'emotion'])

print("2. Cleaning and mapping the data to DOIWIC states...")
# Map Kaggle emotions to 0, 1, 2 DOIWIC logic
emotion_mapping = {
    'anger': 0, 'sadness': 0, 'fear': 0,  # 0 = Frustrated/Overwhelmed
    'surprise': 1,                        # 1 = Neutral/Baseline (Dataset lacks a true 'neutral', surprise is closest baseline)
    'joy': 2, 'love': 2                   # 2 = Highly Motivated
}

df_train['motivation_target'] = df_train['emotion'].map(emotion_mapping)
df_test['motivation_target'] = df_test['emotion'].map(emotion_mapping)

# Drop any unmapped or empty rows
df_train = df_train.dropna(subset=['motivation_target'])
df_test = df_test.dropna(subset=['motivation_target'])

X_train = df_train['text']
y_train = df_train['motivation_target'].astype(int)

X_test = df_test['text']
y_test = df_test['motivation_target'].astype(int)

print(f"Training on {len(X_train)} rows. Testing on {len(X_test)} rows.")

print("\n3. Building the ML Pipeline...")
# The pipeline binds the text translator and the AI model into a single object
# max_df/min_df ignores typos and extremely common filler words
# ngram_range=(1,2) allows the model to understand 2-word phrases like "not good"
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, ngram_range=(1,2))),
    ('classifier', MultinomialNB(alpha=0.1))
])

print("4. Training the unified model...")
pipeline.fit(X_train, y_train)

print("5. Evaluating on unseen test data...")
predictions = pipeline.predict(X_test)

# Calculate and print metrics
accuracy = accuracy_score(y_test, predictions)
print("\n" + "="*50)
print(f"MODEL ACCURACY: {accuracy * 100:.2f}%")
print("="*50)
print("CLASSIFICATION REPORT:")
print(classification_report(
    y_test, 
    predictions, 
    target_names=["0: Frustrated", "1: Baseline", "2: Motivated"]
))

print("\n6. Saving the production pipeline...")
# Because we used a Pipeline, we only save ONE file
joblib.dump(pipeline, "motivation_pipeline.pkl")

print("Success! 'motivation_pipeline.pkl' is ready for your FastAPI backend.")
