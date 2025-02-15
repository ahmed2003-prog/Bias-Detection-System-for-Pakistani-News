from textblob import TextBlob
import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import re
import string

# Load dataset
def load_dataset(path):
    return pd.read_csv(path)

# Preprocess text: Lowercase, remove punctuation, extra spaces
def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Use VADER for sentiment analysis
def get_vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(str(text))
    return sentiment['compound']  # Between -1 and 1

# Compute sentiment and bias
def calculate_sentiment_and_bias(df, column_name):
    df[column_name] = df[column_name].apply(clean_text)
    df['Sentiment'] = df[column_name].apply(get_vader_sentiment)

    # Normalize Bias Score (0-100 scaling)
    df['Bias_Score'] = (abs(df['Sentiment']) - df['Sentiment'].min()) / (df['Sentiment'].max() - df['Sentiment'].min()) * 100

    # Label as Biased or Neutral
    df['Bias_Label'] = df['Sentiment'].apply(lambda x: 'Biased' if abs(x) > 0.05 else 'Neutral')

    return df

# Named Entity Recognition
def extract_named_entities(text, nlp):
    doc = nlp(text)
    entities = {'PERSON': [], 'ORG': [], 'GPE': [], 'OTHER': []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
        else:
            entities['OTHER'].append(ent.text)
    return entities

def perform_ner(df, column_name):
    nlp = spacy.load("en_core_web_sm")
    df['Entities'] = df[column_name].apply(lambda text: extract_named_entities(text, nlp))

    df['Persons'] = df['Entities'].apply(lambda x: ', '.join(set(x['PERSON'])))
    df['Organizations'] = df['Entities'].apply(lambda x: ', '.join(set(x['ORG'])))
    df['Locations'] = df['Entities'].apply(lambda x: ', '.join(set(x['GPE'])))
    df['Other Entities'] = df['Entities'].apply(lambda x: ', '.join(set(x['OTHER'])))

    df.drop(columns=['Entities'], inplace=True)
    return df

# TF-IDF Vectorization
def vectorize_text(df, column_name):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df[column_name])
    return X, vectorizer

# Train classifier
def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    
    print("\n--- Logistic Regression ---")
    print(classification_report(y_test, y_pred))
    print("Cross-validation Accuracy:", cross_val_score(logreg, X, y, cv=5).mean())

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("\n--- Random Forest ---")
    print(classification_report(y_test, y_pred_rf))
    print("Cross-validation Accuracy:", cross_val_score(rf, X, y, cv=5).mean())

def main():
    dataset_path = 'cleaned_news_data.csv'
    df = load_dataset(dataset_path)

    df = calculate_sentiment_and_bias(df, 'Story Excerpt')
    df = perform_ner(df, 'Story Excerpt')

    # TF-IDF + ML Classifier
    X, vectorizer = vectorize_text(df, 'Story Excerpt')
    y = df['Bias_Label']
    train_classifier(X, y)

    df.to_csv('news_with_bias_and_ner.csv', index=False)

if __name__ == "__main__":
    main()
