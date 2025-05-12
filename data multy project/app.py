import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, chi2_contingency, f_oneway, ttest_ind
from collections import Counter
import re
import numpy as np

# ------------------------- إعداد الصفحة -------------------------
st.set_page_config(
    page_title="Book Data Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- تحميل البيانات -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('clean_books_data.csv')
    # ضمان وجود قيم صالحة للاختبارات الإحصائية
    df = df.dropna(subset=['price', 'rating', 'availability'])
    df = df[(df['price'] > 0) & (df['rating'].between(1, 5))]
    return df

df = load_data()

# ------------------------- تحسين التنسيق -------------------------
st.markdown("""
<style>
    .st-emotion-cache-1y4p8pa { padding: 2rem; }
    h1 { color: #2a9d8f; text-align: center; }
    h2 { color: #264653; border-bottom: 2px solid #2a9d8f; padding-bottom: 0.3rem; }
    .stMetric { background-color: #f0f2f6; padding: 1rem; border-radius: 10px; }
    .stPlotlyChart { border: 1px solid #e1e4e8; border-radius: 10px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------- شريط الفلاتر الجانبي -------------------------
with st.sidebar:
    st.header("⚙️ فلاتر البيانات")
    search_title = st.text_input("🔍 بحث بالعنوان")
    price_range = st.slider("💰 نطاق السعر (£)", float(df['price'].min()), float(df['price'].max()), (10.0, 50.0), step=0.5)
    rating_options = st.multiselect("⭐ التقييم (من 1 إلى 5)", options=sorted(df['rating'].unique()), default=sorted(df['rating'].unique()))
    availability_options = st.multiselect("📦 حالة التوفر", options=df['availability'].unique(), default=df['availability'].unique())
    top_rated = st.checkbox("📌 الكتب الأعلى تقييمًا فقط (4-5 نجوم)")
    st.download_button("💾 تحميل البيانات المصفاة", df.to_csv(index=False), "filtered_books.csv", "text/csv")

# ------------------------- تطبيق الفلاتر -------------------------
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

# ------------------------- مؤشرات الأداء الرئيسية -------------------------
st.title("📊 لوحة تحليل بيانات الكتب")

# تنسيق خاص لعناصر KPIs
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
        <p>📚 عدد الكتب</p>
        <h1>{len(df_filtered)}</h1>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class='kpi-box'>
        <p>💰 متوسط السعر</p>
        <h1>£{df_filtered['price'].mean():.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class='kpi-box'>
        <p>⭐ متوسط التقييم</p>
        <h1>{df_filtered['rating'].mean():.1f}/5</h1>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class='kpi-box'>
        <p>🏆 أعلى سعر</p>
        <h1>£{df_filtered['price'].max()}</h1>
    </div>
    """, unsafe_allow_html=True)

# ------------------------- التابات -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 التحليلات العامة", "📊 العلاقات بين المتغيرات", "📋 عرض البيانات", "🧪 اختبارات إحصائية", "📚 وصف إحصائي"])

# --- التحليلات العامة ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("توزيع الأسعار")
        fig1 = px.histogram(df_filtered, x='price', nbins=20, color='availability', template='plotly_white')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.subheader("توزيع التقييمات")
        fig2 = px.pie(df_filtered, names='rating', color_discrete_sequence=px.colors.sequential.Viridis, hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

# --- العلاقات ---
with tab2:
    st.subheader("العلاقة بين السعر والتقييم")
    fig3 = px.scatter(df_filtered, x='price', y='rating', color='availability', hover_name='title', size='price', trendline="lowess")
    st.plotly_chart(fig3, use_container_width=True)
    st.subheader("مقارنة التوفر حسب التقييم")
    fig4 = px.box(df_filtered, x='rating', y='price', color='availability', points="all")
    st.plotly_chart(fig4, use_container_width=True)
    st.subheader("Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df_filtered[['price', 'rating']].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --- عرض البيانات ---
with tab3:
    st.subheader("عرض البيانات المصفاة")
    st.dataframe(df_filtered, use_container_width=True)
    if st.checkbox("🔍 عرض تحليل نصي للعناوين"):
        words = ' '.join(df_filtered['title']).lower()
        words = re.findall(r'\w{3,}', words)
        word_counts = Counter(words).most_common(10)
        st.table(pd.DataFrame(word_counts, columns=['كلمة', 'تكرار']))

# --- اختبارات إحصائية ---# ------------------ Tab 4: الاختبارات الإحصائية ------------------
with tab4:
    st.header("تحليل إحصائي حسب السمات المختلفة")

    # المقارنة حسب النوع (T-test)
    st.subheader("1. مقارنة الأسعار حسب النوع (T-test)")
    groups = df['Category'].unique()
    if len(groups) == 2:
        group1 = df[df['Category'] == groups[0]]['Price']
        group2 = df[df['Category'] == groups[1]]['Price']
        t_stat, p_val = stats.ttest_ind(group1, group2)
        st.markdown(f"- T-statistic = {t_stat:.4f}")
        st.markdown(f"- p-value = {p_val:.4f}")
        if p_val < 0.05:
            st.success("يوجد فرق دال إحصائيًا بين النوعين")
        else:
            st.info("لا يوجد فرق دال إحصائيًا بين النوعين")
    else:
        st.warning("البيانات لا تحتوي على مجموعتين فقط للمقارنة.")

    # المقارنة حسب التقييم (ANOVA)
    st.subheader("2. مقارنة الأسعار حسب التقييم (اختبار ANOVA)")
    grouped_prices = [group['Price'].values for name, group in df.groupby('Rating')]
    f_stat, p_val = stats.f_oneway(*grouped_prices)
    st.markdown(f"- F-statistic = {f_stat:.4f}")
    st.markdown(f"- p-value = {p_val:.4f}")
    if p_val < 0.05:
        st.success("يوجد فرق دال إحصائيًا بين مستويات التقييم المختلفة")
    else:
        st.info("لا يوجد فرق دال إحصائيًا بين مستويات التقييم المختلفة")

    # جدول لمتوسط السعر حسب التقييم
    st.write("متوسط السعر حسب التقييم:")
    rating_price_summary = df.groupby('Rating').agg(
        متوسط_السعر=('Price', lambda x: f"£{x.mean():.2f}"),
        عدد_الكتب=('Price', 'count')
    ).reset_index().rename(columns={"Rating": "مستوى التقييم"})
    st.dataframe(rating_price_summary)

    # المقارنة حسب توفر المنتج (T-test)
    st.subheader("3. مقارنة الأسعار حسب التوفر (T-test)")
    if 'Availability' in df.columns:
        available = df[df['Availability'] == 'In stock']['Price']
        unavailable = df[df['Availability'] != 'In stock']['Price']
        t_stat, p_val = stats.ttest_ind(available, unavailable)
        st.markdown(f"- T-statistic = {t_stat:.4f}")
        st.markdown(f"- p-value = {p_val:.4f}")
        if p_val < 0.05:
            st.success("يوجد فرق دال إحصائيًا حسب التوفر")
        else:
            st.info("لا يوجد فرق دال إحصائيًا حسب التوفر")
    else:
        st.warning("لا يوجد عمود 'Availability' في البيانات.")
# --- الوصف الإحصائي ---
with tab5:
    st.subheader("إحصاءات وصفية")
    st.write("#### الإحصاءات الأساسية:")
    st.dataframe(df_filtered[['price', 'rating']].describe())
    st.write("#### مصفوفة التباين-التغاير:")
    st.dataframe(df_filtered[['price', 'rating']].cov())
    st.write("#### مصفوفة الارتباط:")
    st.dataframe(df_filtered[['price', 'rating']].corr())

# ------------------------- معلومات إضافية -------------------------
st.divider()
with st.expander("ℹ️ معلومات عن المشروع"):
    st.write("""
    **📚 مشروع تحليل بيانات الكتب**
    - تم جمع البيانات من موقع [Books to Scrape](http://books.toscrape.com/)
    - الأدوات المستخدمة: Python, Pandas, Streamlit, Plotly
    - تحليل الأسعار، التقييمات، حالة التوفر
    """)