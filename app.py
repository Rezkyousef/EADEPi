import gradio as gr
import pandas as pd
import joblib

# 1. تحميل الملفات الأساسية (المودل، المقياس، وترتيب الأعمدة)
model = joblib.load("attrition_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

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
        return f"⚠️ تحذير: الموظف معرض بنسبة {probability:.1f}% لمغادرة الشركة!"
    else:
        return f"✅ أمان: الموظف مستقر (احتمالية المغادرة {probability:.1f}% فقط)."

# 3. تصميم واجهة المستخدم
inputs = [
    gr.Slider(minimum=18, maximum=60, step=1, value=30, label="العمر (Age)"),
    gr.Number(value=5000, label="الدخل الشهري (Monthly Income)"),
    gr.Radio(choices=["Yes", "No"], value="No", label="هل يعمل ساعات إضافية؟ (OverTime)"),
    gr.Slider(minimum=0, maximum=40, step=1, value=5, label="إجمالي سنوات الخبرة (Total Working Years)"),
    gr.Slider(minimum=0, maximum=40, step=1, value=3, label="سنوات العمل في الشركة (Years at Company)"),
    gr.Slider(minimum=1, maximum=4, step=1, value=3, label="الرضا الوظيفي (Job Satisfaction: 1 ضعيف - 4 ممتاز)"),
    gr.Slider(minimum=1, maximum=4, step=1, value=3, label="الرضا عن بيئة العمل (Environment Satisfaction: 1-4)")
]

outputs = gr.Textbox(label="نتيجة التنبؤ (Prediction)")

# تشغيل التطبيق
app = gr.Interface(
    fn=predict_attrition, 
    inputs=inputs, 
    outputs=outputs,
    title="🏢 Employee Attrition Predictor (XGBoost)",
    description="قم بإدخال أهم بيانات الموظف (أقوى المؤثرات حسب التحليل)، وسيقوم نموذج الذكاء الاصطناعي بحساب احتمالية استقالته."
)

if __name__ == "__main__":
    app.launch()
