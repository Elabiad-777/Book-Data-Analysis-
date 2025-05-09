# 📚 Book Data Analysis Dashboard

An interactive data analysis dashboard built using **Streamlit** to explore book data scraped from [Books to Scrape](http://books.toscrape.com).

---

## 🚀 Project Features

- Filter books by **price**, **rating**, and **availability**
- View interactive charts:
  - Price Distribution
  - Ratings Pie Chart
  - Price vs Rating Scatter
  - Boxplot of Price by Rating and Availability
- Descriptive statistics (KPIs)
- Text analysis for most common words in book titles
- Download filtered data

---

## 🛠️ Tech Stack

- Python
- Pandas
- Streamlit
- Plotly & Seaborn
- Matplotlib

---

## 📁 Project Structure

├── app.py # Streamlit app for interactive dashboard
├── scraper.py # Script to scrape book data from the website
├── clean_data.py # Data cleaning and preprocessing
├── analysis.py # Statistical tests and visual analysis
├── books_data.csv # Raw scraped data
├── clean_books_data.csv # Cleaned and processed data
├── correlation_heatmap.png# Correlation matrix heatmap
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## 📊 Statistical Tests Included

- **Normality Test**: Shapiro-Wilk test on price
- **ANOVA Test**: Compare mean price across availability statuses
- **Tukey's HSD Test**: Post-hoc test for pairwise comparison
- **Chi-Square Test**: Test independence between price category and availability

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/book-data-dashboard.git
   cd book-data-dashboard




