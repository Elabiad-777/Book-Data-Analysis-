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

# --- اختبارات إحصائية ---
with tab4:
    st.subheader("🧪 الاختبارات الإحصائية المتقدمة")
    
    # 1. اختبار التوزيع الطبيعي (Shapiro-Wilk)
    st.markdown("#### 1. اختبار التوزيع الطبيعي (Shapiro-Wilk)")
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            stat, p = shapiro(df['price'])
            st.write("**اختبار توزيع السعر:**")
            st.write(f"- الإحصائية = {stat:.4f}")
            st.write(f"- القيمة p = {p:.4e}")
            if p > 0.05:
                st.success("البيانات تتبع توزيعًا طبيعيًا", icon="✅")
            else:
                st.warning("البيانات لا تتبع توزيعًا طبيعيًا", icon="⚠️")
        except Exception as e:
            st.error(f"خطأ في اختبار السعر: {str(e)}")
    
    with col2:
        try:
            stat, p = shapiro(df['rating'])
            st.write("**اختبار توزيع التقييم:**")
            st.write(f"- الإحصائية = {stat:.4f}")
            st.write(f"- القيمة p = {p:.4e}")
            if p > 0.05:
                st.success("البيانات تتبع توزيعًا طبيعيًا", icon="✅")
            else:
                st.warning("البيانات لا تتبع توزيعًا طبيعيًا", icon="⚠️")
        except Exception as e:
            st.error(f"خطأ في اختبار التقييم: {str(e)}")
    
    st.divider()
    
    # 2. اختبار Levene لتساوي التباين (بين مستويات التقييم)
    st.markdown("#### 2. اختبار تساوي التباين (Levene)")
    st.write("**مقارنة تباينات الأسعار بين مستويات التقييم المختلفة:**")
    
    try:
        rating_groups = [df[df['rating'] == r]['price'] for r in sorted(df['rating'].unique()) if len(df[df['rating'] == r]) >= 2]
        
        if len(rating_groups) >= 2:
            stat, p = levene(*rating_groups)
            st.write(f"- الإحصائية = {stat:.4f}")
            st.write(f"- القيمة p = {p:.4e}")
            
            if p > 0.05:
                st.success("يمكن افتراض تساوي التباينات بين مستويات التقييم", icon="✅")
            else:
                st.warning("لا يمكن افتراض تساوي التباينات بين مستويات التقييم", icon="⚠️")
            
            # عرض متوسط التباين لكل مجموعة
            st.write("\n**مقارنة التباينات:**")
            var_comparison = []
            for rating in sorted(df['rating'].unique()):
                group_data = df[df['rating'] == rating]['price']
                if len(group_data) >= 2:
                    var_comparison.append({
                        'مستوى التقييم': rating,
                        'عدد الكتب': len(group_data),
                        'متوسط السعر': f"£{group_data.mean():.2f}",
                        'التباين': f"{group_data.var():.2f}"
                    })
            st.table(pd.DataFrame(var_comparison))
        else:
            st.warning("يحتاج إلى مستويين تقييم على الأقل مع بيانات كافية")
    except Exception as e:
        st.error(f"خطأ في إجراء اختبار Levene: {str(e)}")
    
    st.divider()
    
    # 3. اختبار T-test لمقارنة الأسعار حسب التوفر
    st.markdown("#### 3. اختبار T-test لمقارنة الأسعار حسب التوفر")
    st.write("**مقارنة متوسطات الأسعار بين حالات التوفر المختلفة:**")
    
    try:
        availability_types = df['availability'].unique()
        comparisons = []
        
        for i in range(len(availability_types)):
            for j in range(i+1, len(availability_types)):
                group1 = availability_types[i]
                group2 = availability_types[j]
                data1 = df[df['availability'] == group1]['price']
                data2 = df[df['availability'] == group2]['price']
                
                if len(data1) >= 2 and len(data2) >= 2:
                    t_stat, p_value = ttest_ind(data1, data2, equal_var=False)
                    comparisons.append({
                        'المجموعة الأولى': group1,
                        'المجموعة الثانية': group2,
                        'متوسط المجموعة الأولى': f"£{data1.mean():.2f}",
                        'متوسط المجموعة الثانية': f"£{data2.mean():.2f}",
                        't-statistic': f"{t_stat:.4f}",
                        'p-value': f"{p_value:.4e}",
                        'النتيجة': "يوجد فرق معنوي" if p_value <= 0.05 else "لا يوجد فرق معنوي"
                    })
        
        if comparisons:
            st.table(pd.DataFrame(comparisons))
            
            # عرض تفسير النتائج
            st.write("\n**تفسير النتائج:**")
            for comp in comparisons:
                if float(comp['p-value']) <= 0.05:
                    st.info(f"يوجد فرق ذو دلالة إحصائية بين {comp['المجموعة الأولى']} و {comp['المجموعة الثانية']} (p = {comp['p-value']})")
        else:
            st.warning("لا توجد مقارنات صالحة يمكن إجراؤها (تحتاج كل مجموعة إلى كتابين على الأقل)")
    except Exception as e:
        st.error(f"خطأ في إجراء اختبار t-test: {str(e)}")
    
    st.divider()
    
    # 4. اختبار ANOVA لمقارنة الأسعار حسب التقييم
    st.markdown("#### 4. اختبار ANOVA لمقارنة الأسعار حسب التقييم")
    st.write("**مقارنة متوسطات الأسعار بين مستويات التقييم المختلفة:**")
    
    try:
        rating_groups = [df[df['rating'] == r]['price'] for r in sorted(df['rating'].unique()) if len(df[df['rating'] == r]) >= 2]
        
        if len(rating_groups) >= 2:
            f_stat, p_value = f_oneway(*rating_groups)
            st.write(f"- F-statistic = {f_stat:.4f}")
            st.write(f"- p-value = {p_value:.4e}")
            
            if p_value > 0.05:
                st.success("لا يوجد فرق ذو دلالة إحصائية بين المجموعات", icon="✅")
            else:
                st.warning("يوجد فرق ذو دلالة إحصائية بين المجموعات", icon="⚠️")
            
            # عرض متوسط السعر لكل مستوى تقييم
            st.write("\n**متوسط السعر حسب التقييم:**")
            avg_prices = []
            for rating in sorted(df['rating'].unique()):
                group_data = df[df['rating'] == rating]['price']
                if len(group_data) >= 1:
                    avg_prices.append({
                        'مستوى التقييم': rating,
                        'عدد الكتب': len(group_data),
                        'متوسط السعر': f"£{group_data.mean():.2f}",
                        'الانحراف المعياري': f"£{group_data.std():.2f}"
                    })
            st.table(pd.DataFrame(avg_prices))
        else:
            st.warning("يحتاج إلى مستويين تقييم على الأقل مع بيانات كافية")
    except Exception as e:
        st.error(f"خطأ في إجراء اختبار ANOVA: {str(e)}")
    
    st.divider()
    
    # 5. اختبار Chi-square للاستقلال بين التوفر والتقييم
    st.markdown("#### 5. اختبار استقلالية التوفر والتقييم (Chi-square)")
    
    try:
        contingency_table = pd.crosstab(df['availability'], df['rating'])
        
        if contingency_table.size > 0 and not (contingency_table == 0).all().all():
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            st.write(f"- Chi-square statistic = {chi2:.4f}")
            st.write(f"- p-value = {p:.4e}")
            st.write(f"- Degrees of freedom = {dof}")
            
            if p > 0.05:
                st.success("لا يوجد ارتباط بين حالة التوفر ومستوى التقييم", icon="✅")
            else:
                st.warning("يوجد ارتباط بين حالة التوفر ومستوى التقييم", icon="⚠️")
            
            # عرض جدول التكرارات
            st.write("\n**جدول التكرارات الملاحظة:**")
            st.table(contingency_table)
            
            st.write("\n**جدول التكرارات المتوقعة:**")
            st.table(pd.DataFrame(expected, 
                                index=contingency_table.index, 
                                columns=contingency_table.columns).round(2))
        else:
            st.warning("لا توجد بيانات كافية لبناء جدول التكرارات")
    except Exception as e:
        st.error(f"خطأ في إجراء اختبار Chi-square: {str(e)}")

        
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