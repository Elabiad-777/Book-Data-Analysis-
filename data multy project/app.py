import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats 
from collections import Counter
import re
import numpy as np
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(
    page_title="Book Data Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    df = pd.read_csv('clean_books_data.csv')
    df = df.dropna(subset=['price', 'rating', 'availability'])
    df = df[(df['price'] > 0) & (df['rating'].between(1, 5))]
    return df

df = load_data()

st.markdown("""
<style>
    .st-emotion-cache-1y4p8pa { padding: 2rem; }
    h1 { color: #2a9d8f; text-align: center; }
    h2 { color: #264653; border-bottom: 2px solid #2a9d8f; padding-bottom: 0.3rem; }
    .stMetric { background-color: #f0f2f6; padding: 1rem; border-radius: 10px; }
    .stPlotlyChart { border: 1px solid #e1e4e8; border-radius: 10px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Filters")
    search_title = st.text_input("🔍 Search by Title")
    price_range = st.slider("💰 Price Range (£)", float(df['price'].min()), float(df['price'].max()), (10.0, 50.0), step=0.5)
    rating_options = st.multiselect("⭐ Rating (1 to 5)", options=sorted(df['rating'].unique()), default=sorted(df['rating'].unique()))
    availability_options = st.multiselect("📦 Availability", options=df['availability'].unique(), default=df['availability'].unique())
    top_rated = st.checkbox("📌 Top Rated Books Only (4-5 Stars)")
    st.download_button("💾 Download Filtered Data", df.to_csv(index=False), "filtered_books.csv", "text/csv")

df_filtered = df[
    (df['price'] >= price_range[0]) & 
    (df['price'] <= price_range[1]) &
    (df['rating'].isin(rating_options)) &
    (df['availability'].isin(availability_options))
]

if search_title:
    df_filtered = df_filtered[df_filtered['title'].str.contains(search_title, case=False)]

if top_rated:
    df_filtered = df_filtered[df_filtered['rating'] >= 4]

st.title("📊 Book Data Analysis Dashboard")

st.markdown("""
    <style>
    .kpi-box {
        background-color: #e0f7fa;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        color: #264653;
    }
    .kpi-box h1 {
        font-size: 2.2rem;
        margin-bottom: 5px;
    }
    .kpi-box p {
        font-size: 1.2rem;
        margin: 0;
    }
    </style>
""", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class='kpi-box'>
        <p>📚 Total Books</p>
        <h1>{len(df_filtered)}</h1>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class='kpi-box'>
        <p>💰 Average Price</p>
        <h1>£{df_filtered['price'].mean():.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class='kpi-box'>
        <p>⭐ Average Rating</p>
        <h1>{df_filtered['rating'].mean():.1f}/5</h1>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class='kpi-box'>
        <p>🏆 Highest Price</p>
        <h1>£{df_filtered['price'].max()}</h1>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 General Analytics", "📊 Variable Relationships", "📋 Data View", "🧪 Statistical Tests", "📚 Descriptive Stats"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Price Distribution")
        fig1 = px.histogram(df_filtered, x='price', nbins=20, color='availability', template='plotly_white')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.subheader("Rating Distribution")
        fig2 = px.pie(df_filtered, names='rating', color_discrete_sequence=px.colors.sequential.Viridis, hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Price vs Rating")
    fig3 = px.scatter(df_filtered, x='price', y='rating', color='availability', hover_name='title', size='price', trendline="lowess")
    st.plotly_chart(fig3, use_container_width=True)
    st.subheader("Availability by Rating")
    fig4 = px.box(df_filtered, x='rating', y='price', color='availability', points="all")
    st.plotly_chart(fig4, use_container_width=True)
    st.subheader("Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df_filtered[['price', 'rating']].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with tab3:
    st.subheader("Filtered Data View")
    st.dataframe(df_filtered, use_container_width=True)
    if st.checkbox("🔍 Textual Analysis of Titles"):
        words = ' '.join(df_filtered['title']).lower()
        words = re.findall(r'\w{3,}', words)
        word_counts = Counter(words).most_common(10)
        st.table(pd.DataFrame(word_counts, columns=['Word', 'Count']))

with tab4:
    st.header("🧪 Comprehensive Statistical Tests between Price and Rating")
    st.markdown("### ℹ️ All tests assume α = 0.05")
    
    df_test = df_filtered[['price', 'rating']].dropna().copy()
    df_test['price_z'] = stats.zscore(df_test['price'])
    df_test['rating_z'] = stats.zscore(df_test['rating'])

    # تصنيف المجموعات حسب rating الأصلي وليس الموحد
    df_test['rating_group'] = pd.cut(df_filtered['rating'], bins=3, labels=['Low', 'Medium', 'High'])

    # --- Shapiro-Wilk ---
    st.subheader("1️⃣ Shapiro-Wilk Test for Normality")
    st.write("**H₀**: Data is normally distributed  \n**H₁**: Data is not normally distributed")
    normality_results = pd.DataFrame(columns=['Variable', 'W Statistic', 'p-value', 'Decision', 'Conclusion'])
    for col in ['price_z', 'rating_z']:
        stat, p = stats.shapiro(df_test[col])
        decision = "✅ Normal" if p > 0.05 else "❌ Not Normal"
        conclusion = "Meets normality" if p > 0.05 else "Violates normality"
        normality_results.loc[len(normality_results)] = [col.replace('_z', ''), f"{stat:.4f}", f"{p:.4f}", decision, conclusion]
    st.table(normality_results)

    # --- Levene's Test ---
    st.subheader("2️⃣ Levene's Test for Homogeneity of Variance")
    st.write("**H₀**: Equal variances between rating groups  \n**H₁**: Unequal variances between rating groups")
    groups = [df_test[df_test['rating_group'] == g]['price_z'] for g in df_test['rating_group'].unique()]
    stat, p = stats.levene(*groups)
    levene_result = pd.DataFrame({
        'Tested Variable': ['Price across Rating Groups'],
        'Levene Statistic': [f"{stat:.4f}"],
        'p-value': [f"{p:.4f}"],
        'Decision': ["✅ Equal variances" if p > 0.05 else "❌ Unequal variances"],
        'Conclusion': ["Variance assumption met" if p > 0.05 else "Variance assumption violated"]
    })
    st.table(levene_result)

    # --- Correlation Test ---
    st.subheader("3️⃣ Pearson Correlation between Price and Rating")
    st.write("**H₀**: No correlation between price and rating  \n**H₁**: There is a correlation between price and rating")
    corr_coef, p_value = stats.pearsonr(df_test['price_z'], df_test['rating_z'])
    corr_result = pd.DataFrame({
        'Correlation Coefficient': [f"{corr_coef:.4f}"],
        'p-value': [f"{p_value:.4f}"],
        'Decision': ["✅ Significant" if p_value < 0.05 else "❌ Not significant"],
        'Conclusion': ["There is a relationship" if p_value < 0.05 else "No clear relationship"]
    })
    st.table(corr_result)

    # --- MANOVA ---
    st.subheader("4️⃣ MANOVA: Effect of Rating Group on Price and Rating")
    st.write("**H₀**: Rating groups have no effect on price and rating  \n**H₁**: Rating groups affect price and rating jointly")
    try:
        manova = MANOVA.from_formula('price + rating ~ rating_group', data=df_test)
        result = manova.mv_test()
        wilks = result.results['rating_group']['stat']
        stat_val = wilks.loc["Wilks' lambda", "Value"]
        f_val = wilks.loc["Wilks' lambda", "F Value"]
        p_val = wilks.loc["Wilks' lambda", "Pr > F"]
        manova_result = pd.DataFrame({
            "Wilks' Lambda": [f"{stat_val:.4f}"],
            'F-value': [f"{f_val:.4f}"],
            'p-value': [f"{p_val:.4f}"],
            'Decision': ["✅ Significant effect" if p_val < 0.05 else "❌ No significant effect"],
            'Conclusion': ["Groups influence jointly" if p_val < 0.05 else "No joint effect"]
        })
        st.table(manova_result)
    except Exception as e:
        st.warning(f"MANOVA could not be performed: {str(e)}")





with tab5:
    st.subheader("Descriptive Statistics")
    st.write("#### Summary Statistics:")
    st.dataframe(df_filtered[['price', 'rating']].describe())
    st.write("#### Covariance Matrix:")
    st.dataframe(df_filtered[['price', 'rating']].cov())
    st.write("#### Correlation Matrix:")
    st.dataframe(df_filtered[['price', 'rating']].corr())

st.divider()
with st.expander("ℹ️ Project Info"):
    st.write("""
    **📚 Book Data Analysis Project**
    - Data collected from [Books to Scrape](http://books.toscrape.com/)
    - Tools used: Python, Pandas, Streamlit, Plotly
    - Focus: Prices, Ratings, Availability
    """)
