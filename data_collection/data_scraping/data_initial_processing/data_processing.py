import pandas as pd
import logging
from datetime import datetime
from enum import Enum
from websites_scrapper.logger_setup import setup_logger

# Initialize logger
logger = setup_logger()

class SourceType(Enum):
    """Enum for different news sources."""
    ALL = "all"
    DAILY = "daily"
    DAWN = "dawn"

class LogLevel(Enum):
    """Enum for logging levels."""
    INFO = logging.INFO
    ERROR = logging.ERROR
    DEBUG = logging.DEBUG


def process_csv(file_path: str, source: SourceType, date_format: str) -> pd.DataFrame:
    """Processes a CSV file by cleaning and formatting it according to source-specific date format.

    Args:
        file_path (str): Path to the CSV file.
        source (SourceType): Enum representing the source type.
        date_format (str): Date format to be parsed.

    Returns:
        pd.DataFrame: Cleaned and formatted DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        df = df[df['Section'].notna() & (df['Section'] != 'World')]

        df['Story Heading'] = df['Story Heading'].str.replace(',', '', regex=False)
        df['Story Excerpt'] = df['Story Excerpt'].str.replace(',', '', regex=False)

        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format=date_format, errors='coerce')
        df = df[df['Timestamp'].notna()]
        df['Date'] = df['Timestamp'].dt.date
        df.drop(columns=['Timestamp'], inplace=True)

        df['Date'] = df['Date'].apply(lambda x: f"{x.day}-{x.month}-{x.year}" if pd.notnull(x) else '')
        df['Source'] = source.value

        logger.log(LogLevel.INFO.value, f"Successfully processed {file_path}")
        return df
    except Exception as e:
        logger.log(LogLevel.ERROR.value, f"Error processing {file_path}: {e}")
        return pd.DataFrame()


def combine_csv(*file_paths: str) -> None:
    """Combines multiple processed CSV files by ensuring all sources appear together for the same date.

    Args:
        *file_paths (str): Paths to the processed CSV files.
    """
    try:
        dataframes = [pd.read_csv(fp) for fp in file_paths]
        for df in dataframes:
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        
        combined_df = pd.concat(dataframes)
        combined_df.sort_values(by=['Date', 'Source'], ascending=[True, True], inplace=True)
        combined_df.to_csv('combined_data.csv', index=False, encoding='utf-8-sig')
        
        logger.log(LogLevel.INFO.value, "Successfully combined CSV files into 'combined_data.csv'")
    except Exception as e:
        logger.log(LogLevel.ERROR.value, f"Error combining CSV files: {e}")


if __name__ == "__main__":
    df_2020_2021_all = process_csv("2020-2021_all.csv", SourceType.ALL, "%d %b, %Y %I:%M%p")
    df_2021_2025_daily = process_csv("2021-2025_daily.csv", SourceType.DAILY, "%B %d, %Y")
    df_2021_2025_dawn = process_csv("2021-2025_dawn.csv", SourceType.DAWN, "%d %b, %Y %I:%M%p")

    df_2020_2021_all.to_csv("processed_2020_2021_all.csv", index=False, encoding="utf-8-sig")
    df_2021_2025_daily.to_csv("processed_2021_2025_daily.csv", index=False, encoding="utf-8-sig")
    df_2021_2025_dawn.to_csv("processed_2021_2025_dawn.csv", index=False, encoding="utf-8-sig")

    combine_csv("processed_2020_2021_all.csv", "processed_2021_2025_daily.csv", "processed_2021_2025_dawn.csv")