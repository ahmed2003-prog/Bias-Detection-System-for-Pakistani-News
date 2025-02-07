import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from datetime import datetime
import time
import calendar
import logging
from enum import Enum
from logger_setup import scraper_logger
from utils import WEB_DRIVER_EXECUTABLE_PATH
from edge_driver_setup import configure_edge_webdriver


class LogLevel(Enum):
    """Enum for defining log levels."""
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


class NewsScraper:
    """A web scraper for extracting news articles from multiple sources."""

    def __init__(self, base_url: str, driver_path: str, max_pages: int = 5000, max_articles_per_page: int = 10):
        """
        Initializes the NewsScraper instance.

        :param base_url: The base URL of the news site.
        :param driver_path: Path to the WebDriver executable.
        :param max_pages: Maximum number of pages to scrape.
        :param max_articles_per_page: Maximum articles to scrape per page.
        """
        self.base_url = base_url
        self.driver = configure_edge_webdriver(driver_path)
        self.max_pages = max_pages
        self.max_articles_per_page = max_articles_per_page
        self.data = []
        scraper_logger.info("NewsScraper initialized.")

    def scrape_daily_times(self) -> None:
        """Scrapes news articles from the Daily Times website."""
        driver = self.driver
        base_url = "https://dailytimes.com.pk/pakistan/"
        columns = ['Story Heading', 'Story Excerpt', 'Timestamp', 'Section']
        data = []
        page_number = 1

        while page_number <= self.max_pages:
            url = f"{base_url}page/{page_number}/" if page_number > 1 else base_url
            scraper_logger.info(f"Scraping page {page_number}: {url}")

            try:
                driver.get(url)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'article.post'))
                )
                articles = driver.find_elements(By.CSS_SELECTOR, 'article.post')[:self.max_articles_per_page]

                for article in articles:
                    try:
                        heading_element = article.find_element(By.CSS_SELECTOR, 'h2.entry-title a')
                        heading = heading_element.text.strip()
                        excerpt = article.find_elements(By.CSS_SELECTOR, 'div.entry-content p')[0].text.strip()
                        timestamp = article.find_elements(By.CSS_SELECTOR, 'p.entry-meta time.entry-time')[0].text.strip()
                        section = 'Pakistan'
                        data.append([heading, excerpt, timestamp, section])
                    except NoSuchElementException:
                        scraper_logger.error("Error extracting article details.")

                page_number += 1
            except Exception as e:
                scraper_logger.error(f"Error on page {page_number}: {e}")
                break

        df = pd.DataFrame(data, columns=columns)
        df.to_csv('dailytimes_news.csv', index=False, encoding='utf-8-sig')
        scraper_logger.info("Daily Times scraping completed.")
        driver.quit()

    def scrape_dawn(self) -> None:
        """Scrapes news articles from Dawn."""
        driver = self.driver
        base_url = "https://www.dawn.com/newspaper/front-page/"
        columns = ['Story Heading', 'Story Excerpt', 'Timestamp', 'Section']
        data = []

        for year in range(2020, datetime.now().year + 1):
            for month in range(1, 13):
                if year == datetime.now().year and month > datetime.now().month:
                    break
                num_days = calendar.monthrange(year, month)[1]

                for day in range(1, num_days + 1):
                    date_str = f"{year}-{month:02d}-{day:02d}"
                    url = f"{base_url}{date_str}"
                    scraper_logger.info(f"Scraping Dawn for {date_str}.")
                    try:
                        driver.get(url)
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'article.story'))
                        )
                        articles = driver.find_elements(By.CSS_SELECTOR, 'article.story')

                        for article in articles:
                            try:
                                heading = article.find_element(By.CSS_SELECTOR, 'h2.story__title a').text.strip()
                                excerpt = article.find_element(By.CSS_SELECTOR, 'div.story__excerpt').text.strip()
                                timestamp = article.find_element(By.CLASS_NAME, 'timestamp--time').get_attribute('title')
                                data.append([heading, excerpt, timestamp, 'Pakistan'])
                            except NoSuchElementException:
                                scraper_logger.error("Error extracting article details.")
                    except Exception as e:
                        scraper_logger.error(f"Error on {date_str}: {e}")

        df = pd.DataFrame(data, columns=columns)
        df.to_csv('dawn_news_dataset.csv', index=False, encoding='utf-8-sig')
        scraper_logger.info("Dawn scraping completed.")
        driver.quit()

    def scrape_pakistan_today(self) -> None:
        """Scrapes news articles from Pakistan Today."""
        driver = self.driver
        url = "https://www.pakistantoday.com.pk/category/national"
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.td_module_10.td_module_wrap"))
        )

        articles = driver.find_elements(By.CSS_SELECTOR, "div.td_module_10")
        data = []

        for article in articles:
            try:
                title = article.find_element(By.CSS_SELECTOR, "h3.entry-title a").text.strip()
                excerpt = article.find_element(By.CSS_SELECTOR, "div.td-excerpt").text.strip()
                date = article.find_element(By.CSS_SELECTOR, "span.td-post-date time").text.strip()
                data.append([title, excerpt, date, "Pakistan"])
            except NoSuchElementException:
                scraper_logger.error("Error extracting article details.")

        df = pd.DataFrame(data, columns=["Story Heading", "Story Excerpt", "Timestamp", "Section"])
        df.to_csv("pakistan_today_articles.csv", index=False, encoding="utf-8-sig")
        scraper_logger.info("Pakistan Today scraping completed.")
        driver.quit()


def main() -> None:
    """Main function to execute all scrapers."""
    scraper = NewsScraper(base_url="", driver_path=WEB_DRIVER_EXECUTABLE_PATH)
    scraper.scrape_daily_times()
    scraper.scrape_dawn()
    scraper.scrape_pakistan_today()


if __name__ == "__main__":
    main()
