from textblob import TextBlob
import pandas as pd
import spacy
from utils import load_dataset

# Function to calculate sentiment and bias score
def calculate_sentiment_and_bias(df, column_name):
    # Calculate sentiment using TextBlob
    df['Sentiment'] = df[column_name].apply(lambda text: TextBlob(str(text)).sentiment.polarity)

    # Calculate Bias Score (scale from 0 to 100)
    df['Bias_Score'] = df['Sentiment'].apply(lambda x: abs(x) * 100)  # Absolute sentiment value scaled to 100

    # Bias label based on threshold of sentiment (you can adjust this threshold)
    df['Bias_Label'] = df['Sentiment'].apply(lambda x: 'Biased' if abs(x) > 0.1 else 'Neutral')  

    return df

# Function to extract named entities using spaCy
def extract_named_entities(text, nlp):
    doc = nlp(text)
    entities = {'PERSON': [], 'ORG': [], 'GPE': [], 'OTHER': []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
        else:
            entities['OTHER'].append(ent.text)
    return entities

# Perform NER on the DataFrame
def perform_ner(df, column_name):
    nlp = spacy.load("en_core_web_sm")  # Load small English model
    df['Entities'] = df[column_name].apply(lambda text: extract_named_entities(text, nlp))

    # Expand dictionary into separate columns
    df['Persons'] = df['Entities'].apply(lambda x: ', '.join(set(x['PERSON'])))
    df['Organizations'] = df['Entities'].apply(lambda x: ', '.join(set(x['ORG'])))
    df['Locations'] = df['Entities'].apply(lambda x: ', '.join(set(x['GPE'])))
    df['Other Entities'] = df['Entities'].apply(lambda x: ', '.join(set(x['OTHER'])))

    # Drop the raw dictionary column
    df.drop(columns=['Entities'], inplace=True)

    return df

def main():
    dataset_path = 'cleaned_news_data.csv'  # Path to your dataset
    df = load_dataset(dataset_path)

    # Perform Sentiment Analysis and calculate Bias Score
    df = calculate_sentiment_and_bias(df, 'Story Excerpt')
    df.to_csv('labeled_news_data.csv', index=False)

    # Perform Named Entity Recognition (NER)
    df = perform_ner(df, 'Story Excerpt')
    df.to_csv('news_with_ner.csv', index=False)

if __name__ == "__main__":
    main()
