"""
This module cleans and normalizes a news dataset.

Features:
- Removes special characters, stopwords, and duplicates.
- Converts text to lowercase and lemmatizes words.
- Ensures consistent date formatting (YYYY-MM-DD).
- Logs important steps and errors.

Usage:
    python clean_news_data.py
"""

import re
import pandas as pd
import logging
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from enum import Enum
from logger_setup import scraper_logger

# Download necessary resources (only run this once)
# import nltk
# nltk.download('all')

class LogLevel(Enum):
    """Enum for defining log levels."""
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    DEBUG = logging.DEBUG


def configure_logger():
    """Configures logging settings."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=LogLevel.INFO.value
    )


def clean_text(text: str) -> str:
    """
    Cleans and normalizes text by:
    - Removing special characters.
    - Converting to lowercase.
    - Removing stopwords.
    - Performing lemmatization.

    Args:
        text (str): Input text to be cleaned.

    Returns:
        str: Cleaned and processed text.
    """
    if pd.isnull(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and punctuation
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Tokenize words
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    # Apply lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


def clean_dataset(file_path: str) -> pd.DataFrame:
    """
    Cleans the dataset by:
    - Removing missing values.
    - Normalizing text fields.
    - Formatting date fields.
    - Removing duplicates.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        scraper_logger.info(f"Dataset loaded: {file_path}")

        # Drop rows with missing Story Heading or Story Excerpt
        df.dropna(subset=["Story Heading", "Story Excerpt"], inplace=True)

        # Fill missing Section values
        df["Section"] = df["Section"].fillna("Unknown")

        # Format date field
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

        # Remove rows where date parsing failed
        df.dropna(subset=["Date"], inplace=True)

        # Convert Date to string format
        df["Date"] = df["Date"].astype(str)

        # Remove duplicates
        df.drop_duplicates(inplace=True)

        # Clean text fields
        df["Story Heading"] = df["Story Heading"].apply(clean_text)
        df["Story Excerpt"] = df["Story Excerpt"].apply(clean_text)

        scraper_logger.info("Dataset cleaning completed successfully.")
        return df

    except Exception as e:
        scraper_logger.error(f"Error during data cleaning: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


if __name__ == "__main__":
    configure_logger()

    input_file = "news_dataset.csv"
    output_file = "cleaned_news_data.csv"

    df_cleaned = clean_dataset(input_file)

    if not df_cleaned.empty:
        df_cleaned.to_csv(output_file, index=False, encoding="utf-8-sig")
        scraper_logger.info(f"Cleaned dataset saved to {output_file}")
    else:
        scraper_logger.error("Data cleaning failed. No output file generated.")
