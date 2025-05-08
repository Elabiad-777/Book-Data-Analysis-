import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def scrape_books():
    base_url = "http://books.toscrape.com/catalogue/page-{}.html"
    all_books = []
    
    for page in range(1, 51):  # يجمع البيانات من 50 صفحة
        try:
            response = requests.get(base_url.format(page))
            soup = BeautifulSoup(response.content, 'html.parser')
            
            books = soup.find_all('article', class_='product_pod')
            for book in books:
                title = book.h3.a['title']
                price = float(re.sub(r'[^\d.]', '', book.find('p', class_='price_color').text))
                rating = book.p['class'][1]
                availability = 'In stock' if 'In stock' in book.find('p', class_='instock availability').text else 'Out of stock'
                
                all_books.append({
                    'title': title,
                    'price': price,
                    'rating': rating,
                    'availability': availability
                })
        except Exception as e:
            print(f"Error scraping page {page}: {str(e)}")
    
    return pd.DataFrame(all_books)

if __name__ == "__main__":
    df = scrape_books()
    df.to_csv('books_data.csv', index=False)
    print("Scraping completed! Data saved to books_data.csv")
    