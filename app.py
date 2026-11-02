from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "breast_cancer_rf.joblib"
SAMPLE_PATH = BASE_DIR / "sample_input.csv"

st.set_page_config(
    page_title="طبقه‌بندی توده پستان | پروژه آموزشی",
    page_icon="🧬",
    layout="wide",
)

st.title("سامانه آموزشی طبقه‌بندی توده پستان")
st.caption("مدل یادگیری ماشین Random Forest | پروژه آموزشی")
st.warning(
    "این برنامه صرفاً برای نمایش آموزشی است و ابزار تشخیص پزشکی نیست. "
    "نتیجه آن نباید مبنای تصمیم درمانی قرار گیرد. لطفاً اطلاعات هویتی یا پرونده واقعی بیمار را بارگذاری نکنید."
)

if not MODEL_PATH.exists():
    st.error(
        "فایل مدل پیدا نشد. فایل breast_cancer_rf.joblib را در کنار app.py "
        "و requirements.txt در مخزن GitHub قرار دهید."
    )
    st.stop()

@st.cache_resource
def load_model_package(path: str):
    # مدل ذخیره‌شده را یک بار بارگذاری می‌کند و برای درخواست‌های بعدی نگه می‌دارد.
    return joblib.load(path)

try:
    package = load_model_package(str(MODEL_PATH))
    model = package["model"]
    feature_names = package["feature_names"]
    classes = list(package.get("classes", model.classes_))
except Exception as exc:
    st.error(f"بارگذاری مدل با خطا مواجه شد: {exc}")
    st.stop()

st.subheader("۱. دریافت فایل ورودی")
st.write(
    "یک فایل CSV بارگذاری کنید که هر ردیف آن یک نمونه و ستون‌های آن شامل "
    "۳۰ ویژگی عددی دیتاست باشد. نام و ترتیب ستون‌ها باید مطابق نمونه باشد."
)

if SAMPLE_PATH.exists():
    sample_bytes = SAMPLE_PATH.read_bytes()
    st.download_button(
        label="دانلود فایل CSV نمونه",
        data=sample_bytes,
        file_name="sample_input.csv",
        mime="text/csv",
    )

uploaded_file = st.file_uploader("فایل CSV را انتخاب کنید", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"خواندن فایل CSV ناموفق بود: {exc}")
        st.stop()

    missing_columns = [name for name in feature_names if name not in input_df.columns]
    if missing_columns:
        st.error("این ستون‌ها در فایل شما وجود ندارند: " + ", ".join(missing_columns))
        st.info("از فایل نمونه همین صفحه استفاده کنید تا نام دقیق ستون‌ها را ببینید.")
        st.stop()

    # ستون‌های اضافه را کنار می‌گذارد و ترتیب ویژگی‌ها را دقیقاً مطابق زمان آموزش می‌چیند.
    features_df = input_df.loc[:, feature_names].copy()

    # همه ورودی‌ها باید عددی باشند؛ مقادیر غیرعددی به مقدار خالی تبدیل می‌شوند.
    for column in feature_names:
        features_df[column] = pd.to_numeric(features_df[column], errors="coerce")

    invalid_columns = features_df.columns[features_df.isna().any()].tolist()
    if invalid_columns:
        st.error(
            "در بعضی ستون‌ها مقدار خالی یا غیرعددی وجود دارد: "
            + ", ".join(invalid_columns)
        )
        st.stop()

    if features_df.empty:
        st.error("فایل ورودی هیچ ردیفی ندارد.")
        st.stop()

    st.subheader("۲. پیش‌نمایش داده‌های ورودی")
    st.dataframe(features_df.head(10), use_container_width=True)
    st.caption(f"تعداد نمونه‌های دریافت‌شده: {len(features_df)}")

    if st.button("اجرای پیش‌بینی", type="primary"):
        try:
            predictions = model.predict(features_df)
            results = input_df.copy()
            results["Prediction"] = predictions

            if hasattr(model, "predict_proba") and "Malignant" in classes:
                probabilities = model.predict_proba(features_df)
                malignant_index = classes.index("Malignant")
                results["Malignant probability (%)"] = (
                    probabilities[:, malignant_index] * 100
                ).round(2)

            st.subheader("۳. نتیجه پیش‌بینی مدل")
            st.dataframe(results, use_container_width=True)

            malignant_count = int((pd.Series(predictions) == "Malignant").sum())
            benign_count = int((pd.Series(predictions) == "Benign").sum())
            col1, col2, col3 = st.columns(3)
            col1.metric("تعداد کل نمونه‌ها", len(predictions))
            col2.metric("پیش‌بینی بدخیم", malignant_count)
            col3.metric("پیش‌بینی خوش‌خیم", benign_count)

            csv_output = results.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "دانلود نتایج پیش‌بینی (CSV)",
                data=csv_output,
                file_name="prediction_results.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"اجرای مدل با خطا مواجه شد: {exc}")

st.divider()
st.subheader("مشخصات مدل آموزشی")
col1, col2, col3 = st.columns(3)
col1.metric("الگوریتم", "Random Forest")
col2.metric("تعداد درخت‌ها", getattr(model, "n_estimators", "نامشخص"))
col3.metric("تعداد ویژگی‌ها", len(feature_names))
st.caption(
    "معیارهای ارزیابی به‌دست‌آمده روی مجموعه تست پروژه: "
    "Accuracy = 97.37% | Recall کلاس Malignant = 92.86% | "
    "F1 Score کلاس Malignant = 96.30%. این مقادیر مربوط به اجرای آموزشی ثبت‌شده هستند."
)
