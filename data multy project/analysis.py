import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

def analyze_data():
    df = pd.read_csv('clean_books_data.csv')
    
    # 1. الإحصائيات الوصفية
    stats = {
        'mean': df.mean(),
        'median': df.median(),
        'std': df.std(),
        'correlation': df.corr()
    }
    
    # 2. اختبار التوزيع الطبيعي
    normality = {}
    for col in ['price', 'rating']:
        stat, p = stats.shapiro(df[col])
        normality[col] = {'statistic': stat, 'p-value': p}
    
    # 3. اختبار الفرضيات
    available = df[df['availability'] == 'In stock']['price']
    not_available = df[df['availability'] == 'Out of stock']['price']
    t_test = stats.ttest_ind(available, not_available)
    
    # 4. الرسوم البيانية
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.savefig('correlation_heatmap.png')
    
    return {
        'descriptive_stats': stats,
        'normality_tests': normality,
        'hypothesis_test': t_test
    }

if __name__ == "__main__":
    results = analyze_data()
    print("Analysis completed! Results:")
    print(results)

    