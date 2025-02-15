""" from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time
import calendar
from logger_setup import scraper_logger
from utils import WEB_DRIVER_EXECUTABLE_PATH
from edge_driver_setup import configure_edge_webdriver
from datetime import datetime

# Initialize WebDriver
driver = configure_edge_webdriver(WEB_DRIVER_EXECUTABLE_PATH)
scraper_logger.info("WebDriver initialized")

# Target URL (base URL for front page)
base_url = "https://www.dawn.com/newspaper/front-page/"

# Prepare CSV
columns = ['Story Heading', 'Story Excerpt', 'Timestamp', 'Section']
data = []

# Get the current year and month for scraping until the present date
current_year = datetime.now().year
current_month = datetime.now().month
current_day = datetime.now().day

# Scrape data from 2020 to the present day
for year in range(2020, current_year + 1):
    for month in range(1, 13):
        # Handle the current month so it doesn't go beyond today's date
        if year == current_year and month > current_month:
            break
        num_days = calendar.monthrange(year, month)[1]

        for day in range(1, num_days + 1):
            date_obj = datetime(year, month, day)
            date_str = date_obj.strftime('%Y-%m-%d')  # Format as "2024-01-01"
            url = f"{base_url}{date_str}"
            scraper_logger.info(f"Starting to scrape data for {date_str}...")

            try:
                # Open the URL for the specific date
                driver.get(url)
                scraper_logger.info(f"Opened URL: {url}")

                # Wait for the page to load and ensure articles are present
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'article.story'))
                    )
                except TimeoutException:
                    scraper_logger.warning(f"Articles not found for {date_str}, skipping...")
                    continue  # Skip to the next date if articles aren't loaded

                # Scrape the article data
                articles = driver.find_elements(By.CSS_SELECTOR, 'article.story')
                scraper_logger.info(f"Found {len(articles)} articles for {date_str}.")

                for index, article in enumerate(articles, start=1):
                    try:
                        heading = article.find_element(By.CSS_SELECTOR, 'h2.story__title a').text if article.find_elements(By.CSS_SELECTOR, 'h2.story__title a') else "N/A"
                        excerpt = article.find_element(By.CSS_SELECTOR, 'div.story__excerpt').text if article.find_elements(By.CSS_SELECTOR, 'div.story__excerpt') else "N/A"
                        timestamp = article.find_element(By.CLASS_NAME, 'timestamp--time').get_attribute('title') if article.find_elements(By.CLASS_NAME, 'timestamp--time') else "N/A"
                        section = 'Pakistan'
                        data.append([heading, excerpt, timestamp, section])

                        # Log successful extraction
                        scraper_logger.info(f"Scraped article {index}/{len(articles)} on {date_str}: {heading[:50]}...")

                    except NoSuchElementException as e:
                        scraper_logger.error(f"Error scraping article {index} on {date_str}: {e}")

            except Exception as e:
                scraper_logger.error(f"Error on {date_str}: {e}")

            scraper_logger.info(f"Finished scraping data for {date_str}.\n")

# Save to CSV
scraper_logger.info("Saving scraped data to dawn_news_dataset.csv...")
df = pd.DataFrame(data, columns=columns)
df.to_csv('dawn_news_dataset.csv', index=False, encoding='utf-8-sig')
scraper_logger.info("Scraping completed. Data saved to dawn_news_dataset.csv")

# Close WebDriver
driver.quit()
scraper_logger.info("WebDriver closed")
 """