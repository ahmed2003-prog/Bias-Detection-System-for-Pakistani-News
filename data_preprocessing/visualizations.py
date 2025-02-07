import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob

# Function to calculate sentiment for each article using TextBlob
def calculate_sentiment(df):
    # Create a new column 'Sentiment' that stores the polarity from TextBlob
    df['Sentiment'] = df['Story Excerpt'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    # Add a column for bias score on a scale of 0-100
    df['Bias_Score'] = df['Sentiment'].apply(lambda x: (abs(x) * 100) if abs(x) > 0.1 else 0)  # Example: Abs sentiment score scaled to 100
    df['Bias_Label'] = df['Sentiment'].apply(lambda x: 'Biased' if abs(x) > 0.1 else 'Neutral')  # Label as Biased or Neutral
    
    return df

# Function to visualize sentiment distribution by source
def sentiment_distribution_by_source(df):
    source_sentiment = df.groupby('Source')['Sentiment'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    plt.bar(source_sentiment['Source'], source_sentiment['Sentiment'], color='skyblue')
    plt.xlabel('News Source')
    plt.ylabel('Average Sentiment')
    plt.title('Sentiment Distribution by Source')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Function to visualize sentiment distribution by section
def sentiment_distribution_by_section(df):
    section_sentiment = df.groupby('Section')['Sentiment'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    plt.bar(section_sentiment['Section'], section_sentiment['Sentiment'], color='salmon')
    plt.xlabel('News Section')
    plt.ylabel('Average Sentiment')
    plt.title('Sentiment Distribution by Section')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Function for correlation analysis between sentiment and source/section
def sentiment_correlation_analysis(df):
    # Check for missing data and drop rows with NaN values
    df_clean = df.dropna(subset=['Sentiment', 'Source', 'Section'])

    # Convert 'Source' and 'Section' to categorical codes to calculate correlation
    df_clean['Source_Code'] = df_clean['Source'].astype('category').cat.codes
    df_clean['Section_Code'] = df_clean['Section'].astype('category').cat.codes

    # Calculate correlation between sentiment and source/section
    source_corr = df_clean['Sentiment'].corr(df_clean['Source_Code'])
    section_corr = df_clean['Sentiment'].corr(df_clean['Section_Code'])
    
    print(f"Correlation between sentiment and source: {source_corr}")
    print(f"Correlation between sentiment and section: {section_corr}")

# Function for topic modeling with LDA
def topic_modeling(df):
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(df['Story Excerpt'])
    
    # LDA Model
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(tfidf)

    # Display topics
    print("Top words per topic:")
    terms = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda.components_):
        print(f"Topic {idx+1}:")
        print([terms[i] for i in topic.argsort()[-50:]])  # Top 10 words per topic
        print()

# Function for sentiment analysis over time
def sentiment_over_time(df):
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Group by date and calculate the average sentiment
    sentiment_over_time = df.groupby(df['Date'].dt.date)['Sentiment'].mean()

    # Plot sentiment over time
    sentiment_over_time.plot(figsize=(10, 6), color='blue', linestyle='-', marker='o')
    plt.title('Sentiment Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment')
    plt.tight_layout()
    plt.show()

def main():
    dataset_path = 'news_with_sentiment.csv'
    df = pd.read_csv(dataset_path)

    # Calculate sentiment and add bias score/label
    df = calculate_sentiment(df)

    # Visualizations
    sentiment_distribution_by_source(df)
    sentiment_distribution_by_section(df)

    # Sentiment correlation analysis
    sentiment_correlation_analysis(df)

    # Topic modeling
    topic_modeling(df)

    # Sentiment over time analysis
    sentiment_over_time(df)

    # Display the dataset with Bias Score and Bias Label
    print(df[['Source', 'Section', 'Sentiment', 'Bias_Score', 'Bias_Label']].head(200))

    # Save the labeled dataset to a new CSV file
    df.to_csv('labeled_dataset.csv', index=False)

if __name__ == "__main__":
    main()
