from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from utils import load_dataset

def perform_tfidf_vectorization(df, column_name, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(df[column_name])

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    return tfidf_df, vectorizer

# Load the dataset
dataset_path = 'labeled_news_data.csv'
df = load_dataset(dataset_path)

# Perform TF-IDF Vectorization on the 'Story Excerpt' column
tfidf_df, vectorizer = perform_tfidf_vectorization(df, 'Story Excerpt')

# Save the result to a new CSV file
tfidf_df.to_csv('tfidf_news_data.csv', index=False)
