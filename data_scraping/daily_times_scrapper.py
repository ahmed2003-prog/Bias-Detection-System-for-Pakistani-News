""" from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time
from logger_setup import scraper_logger
from utils import WEB_DRIVER_EXECUTABLE_PATH
from edge_driver_setup import configure_edge_webdriver

# Initialize WebDriver
driver = configure_edge_webdriver(WEB_DRIVER_EXECUTABLE_PATH)
scraper_logger.info("WebDriver initialized")

# Base URL for Pakistan news section
base_url = "https://dailytimes.com.pk/pakistan/"

# Prepare CSV
columns = ['Story Heading', 'Story Excerpt', 'Timestamp', 'Section']
data = []

page_number = 1  # Start from the first page
max_articles_per_page = 10  # Limit to 10 articles per page
max_pages = 5000  # Limit to 5000 pages

while page_number <= max_pages:
    url = f"{base_url}page/{page_number}/" if page_number > 1 else base_url
    scraper_logger.info(f"Scraping page {page_number}: {url}")

    try:
        # Open the page
        driver.get(url)

        # Wait until articles are loaded
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'article.post'))
            )
        except TimeoutException:
            scraper_logger.warning(f"No articles found on page {page_number}, stopping pagination.")
            break  # Stop scraping if no articles are found

        # Scrape the articles
        articles = driver.find_elements(By.CSS_SELECTOR, 'article.post')
        scraper_logger.info(f"Found {len(articles)} articles on page {page_number}.")

        # Check if we got more than expected, limit to 10
        if len(articles) > max_articles_per_page:
            scraper_logger.warning(f"Found more than {max_articles_per_page} articles. Limiting to {max_articles_per_page}.")
            articles = articles[:max_articles_per_page]

        for index, article in enumerate(articles, start=1):
            try:
                # Extract Story Heading & URL
                heading_element = article.find_element(By.CSS_SELECTOR, 'h2.entry-title a')
                heading = heading_element.text.strip()
                article_url = heading_element.get_attribute('href')

                # Extract Story Excerpt
                excerpt_element = article.find_elements(By.CSS_SELECTOR, 'div.entry-content p')
                excerpt = excerpt_element[0].text.strip() if excerpt_element else "N/A"

                # Extract Timestamp (Date)
                timestamp_element = article.find_elements(By.CSS_SELECTOR, 'p.entry-meta time.entry-time')
                timestamp = timestamp_element[0].text.strip() if timestamp_element else "N/A"

                # Section Name (Fixed as 'Pakistan' since we are scraping that section)
                section = 'Pakistan'

                # Store Data
                data.append([heading, excerpt, timestamp, section])
                scraper_logger.info(f"Scraped article {index}/{len(articles)}: {heading[:50]}...")

            except NoSuchElementException as e:
                scraper_logger.error(f"Error scraping article {index} on page {page_number}: {e}")

        # Check for the "Next Page" link and move to the next page
        try:
            next_page = driver.find_element(By.CSS_SELECTOR, 'div.pagination-next a')
            next_page_url = next_page.get_attribute('href')
            scraper_logger.info(f"Moving to the next page: {next_page_url}")
            driver.get(next_page_url)  # Navigate to the next page directly
            page_number += 1  # Increment the page number
        except NoSuchElementException:
            scraper_logger.info("No more pages to scrape.")
            break  # Stop when "Next Page" link is not found

    except Exception as e:
        scraper_logger.error(f"Error on page {page_number}: {e}")
        break  # Stop execution if a critical error occurs

# Save to CSV
scraper_logger.info("Saving scraped data to dailytimes_news.csv...")
df = pd.DataFrame(data, columns=columns)
df.to_csv('dailytimes_news.csv', index=False, encoding='utf-8-sig')
scraper_logger.info("Scraping completed. Data saved to dailytimes_news.csv")

# Close WebDriver
driver.quit()
scraper_logger.info("WebDriver closed") """