# 📚 BookScraper Analytics Dashboard

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.com)

A comprehensive data pipeline for scraping, analyzing, and visualizing book market data from [Books to Scrape](http://books.toscrape.com/).

## 🚀 Features

- **Web Scraping Engine**
  - Multi-page crawler with BeautifulSoup
  - Auto-retry for failed requests
  - Data validation checks

- **Statistical Analysis**
  ```python
  # Example test output
  Shapiro-Wilk Test (Price):
  W = 0.9532, p < .001***  # Non-normal distribution
  
  T-test (In Stock vs Out of Stock):
  t(98) = 1.024, p = .306  # No significant difference
