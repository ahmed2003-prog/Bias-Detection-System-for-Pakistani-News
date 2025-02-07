from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from enum import Enum

class LogLevel(Enum):
    """Enum for defining log level settings in Selenium."""
    OFF = "0"
    ERROR = "1"
    WARNING = "2"
    INFO = "3"

def configure_edge_webdriver(webdriver_executable_path: str) -> webdriver.Edge:
    """
    Configures and returns a Selenium WebDriver for Microsoft Edge with optimized settings.

    Args:
        webdriver_executable_path (str): Path to the Microsoft Edge WebDriver executable.

    Returns:
        webdriver.Edge: Configured Microsoft Edge WebDriver instance.
    """
    edge_browser_options = Options()
    edge_browser_options.add_argument("--headless")  # Run in headless mode (no GUI)
    edge_browser_options.add_argument("--disable-extensions")  # Disable browser extensions
    edge_browser_options.add_argument("--start-maximized")  # Open browser in maximized mode

    # Set logging preferences to suppress warnings
    edge_browser_options.add_argument("--log-level=3")  # INFO or above
    edge_browser_options.add_argument("--disable-gpu")  # Additional potential stability improvement in headless mode

    edge_service = Service(executable_path=webdriver_executable_path)
    edge_driver = webdriver.Edge(service=edge_service, options=edge_browser_options)

    return edge_driver
