import pandas as pd
import numpy as np

def clean_data():
    df = pd.read_csv('books_data.csv')
    
    # تحويل التقييم من نص إلى رقم
    rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
    df['rating'] = df['rating'].map(rating_map)
    
    # معالجة القيم الناقصة
    df['price'] = df['price'].fillna(df['price'].median())
    
    # إضافة عمود جديد للتصنيف السعري
    df['price_category'] = pd.cut(df['price'], 
                                bins=[0, 20, 40, 60, 100, np.inf],
                                labels=['Cheap', 'Affordable', 'Expensive', 'Premium', 'Luxury'])
    
    df.to_csv('clean_books_data.csv', index=False)
    return df

if __name__ == "__main__":
    clean_data()
    print("Data cleaning completed! Clean data saved to clean_books_data.csv")

    