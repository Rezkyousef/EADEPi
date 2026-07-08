import streamlit as st
import pandas as pd
import joblib

# 1. تحميل الملفات الأساسية (المودل، المقياس، وترتيب الأعمدة)
# استخدام cache_resource يمنع تحميل المودل مع كل ضغطة ويسرع التطبيق جداً
@st.cache_resource 
def load_assets():
    model = joblib.load("attrition_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_assets()

# 2. دالة التنبؤ الذكية
def predict_attrition(age, monthly_income, overtime, total_working_years, years_at_company, job_satisfaction, env_satisfaction):
    
    # إنشاء قاموس يحتوي على الـ 34 عموداً وكلها قيمتها صفر مبدئياً
    input_row = {col: 0 for col in feature_columns}
    
    # تحديث القاموس بالقيم الحقيقية التي أدخلها المستخدم
    input_row["Age"] = age
    input_row["MonthlyIncome"] = monthly_income
    input_row["OverTime"] = 1 if overtime == "Yes" else 0
    input_row["TotalWorkingYears"] = total_working_years
    input_row["YearsAtCompany"] = years_at_company
    input_row["JobSatisfaction"] = job_satisfaction
    input_row["EnvironmentSatisfaction"] = env_satisfaction
    
    # حساب الميزة الهندسية التي صنعتها أنت في النوت بوك
    input_row["SalaryExperienceRatio"] = monthly_income / (total_working_years + 1)
    
    # تحويل القاموس إلى DataFrame بنفس ترتيب أعمدة التدريب بالضبط لتجنب أي خطأ
    input_df = pd.DataFrame([input_row])[feature_columns]
    
    # تطبيق توحيد المقاييس (Scaling)
    scaled_data = scaler.transform(input_df)
    
    # التنبؤ
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1] * 100 
    
    if prediction == 1:
        return True, f"⚠️ تحذير: الموظف معرض بنسبة {probability:.1f}% لمغادرة الشركة!"
    else:
        return False, f"✅ أمان: الموظف مستقر (احتمالية المغادرة {probability:.1f}% فقط)."

# 3. تصميم واجهة المستخدم
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="🏢")
st.title("🏢 Employee Attrition Predictor (XGBoost)")
st.write("قم بإدخال أهم بيانات الموظف (أقوى المؤثرات حسب التحليل)، وسيقوم نموذج الذكاء الاصطناعي بحساب احتمالية استقالته.")

# تقسيم الشاشة لعمودين لشكل احترافي
col1, col2 = st.columns(2)

with col1:
    age = st.slider("العمر (Age)", min_value=18, max_value=60, value=30, step=1)
    monthly_income = st.number_input("الدخل الشهري (Monthly Income)", value=5000)
    overtime = st.radio("هل يعمل ساعات إضافية؟ (OverTime)", options=["Yes", "No"], index=1)

with col2:
    total_working_years = st.slider("إجمالي سنوات الخبرة (Total Working Years)", min_value=0, max_value=40, value=5, step=1)
    years_at_company = st.slider("سنوات العمل في الشركة (Years at Company)", min_value=0, max_value=40, value=3, step=1)
    job_satisfaction = st.slider("الرضا الوظيفي (Job Satisfaction: 1 ضعيف - 4 ممتاز)", min_value=1, max_value=4, value=3, step=1)
    env_satisfaction = st.slider("الرضا عن بيئة العمل (Environment Satisfaction: 1-4)", min_value=1, max_value=4, value=3, step=1)

st.markdown("---")

# زر الحساب وتفعيل الدالة
if st.button("احسب الاحتمالية", use_container_width=True):
    is_leaving, message = predict_attrition(
        age, monthly_income, overtime, total_working_years, 
        years_at_company, job_satisfaction, env_satisfaction
    )
    
    # عرض النتيجة بشكل مرئي (أحمر للتحذير، وأخضر للأمان)
    if is_leaving:
        st.error(message)
    else:
        st.success(message)
