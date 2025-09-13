# ----------------------------
# super_store_app.py
# ----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import os

# ----------------------------
# Load pipeline from Hugging Face
# ----------------------------
model_url = "https://huggingface.co/sonic222/model/resolve/main/super_store_pipeline_new.pkl"
model_path = "super_store_pipeline_new.pkl"

if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        r = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(r.content)

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="Superstore Profit Predictor", page_icon="💰", layout="wide")
st.sidebar.title("📂 Data Options")
data_source = st.sidebar.radio("Choose input method:", ["Upload CSV", "Manual Input"])

# ----------------------------
# Safe preprocessing function
# ----------------------------
def safe_preprocess_dates(df):
    df = df.copy()
    required_cols = ["Order Date", "Ship Date", "Sales", "Quantity", "Discount", "Category", "Sub-Category", 
                     "Country", "Region", "Ship Mode", "Segment", "State", "City", "Product Name"]
    
    for col in required_cols:
        if col not in df.columns:
            st.error(f"⚠️ Missing required column: '{col}'")
            st.stop()
    
    # Convert to datetime safely
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors='coerce')
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors='coerce')
    
    if df["Order Date"].isna().all() or df["Ship Date"].isna().all():
        st.error("⚠️ Could not convert 'Order Date' or 'Ship Date' to datetime!")
        st.stop()
    
    # Extract date features
    df["Order_Year"] = df["Order Date"].dt.year
    df["Order_Month"] = df["Order Date"].dt.month
    df["Order_Day"] = df["Order Date"].dt.day
    df["Order_DayOfWeek"] = df["Order Date"].dt.dayofweek
    df["Order_Quarter"] = df["Order Date"].dt.quarter
    df["Shipping_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Month_sin"] = np.sin(2 * np.pi * df["Order_Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Order_Month"] / 12)
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["Order_DayOfWeek"] / 7)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["Order_DayOfWeek"] / 7)
    
    # Safe week extraction
    df["Order_Week"] = df["Order Date"].dt.isocalendar().week
    return df

# ----------------------------
# Load data
# ----------------------------
if data_source == "Upload CSV":
    file = st.sidebar.file_uploader("Upload your Superstore CSV", type=["csv"])
    if file:
        data = pd.read_csv(file, encoding="latin1")
        st.success("✅ Data uploaded successfully")
        data = safe_preprocess_dates(data)
    else:
        st.warning("⚠️ Please upload a file")
        st.stop()
else:
    st.sidebar.subheader("Enter Order Details")
    order_date = st.sidebar.date_input("Order Date")
    ship_date = st.sidebar.date_input("Ship Date")
    row = {
        "Order Date": pd.to_datetime(order_date),
        "Ship Date": pd.to_datetime(ship_date),
        "Sales": st.sidebar.number_input("Sales", min_value=0.0, step=10.0),
        "Quantity": st.sidebar.number_input("Quantity", min_value=1, step=1),
        "Discount": st.sidebar.slider("Discount", 0.0, 1.0, 0.0, 0.05),
        "Category": st.sidebar.selectbox("Category", ["Furniture", "Office Supplies", "Technology"]),
        "Sub-Category": st.sidebar.text_input("Sub-Category", "Chairs"),
        "Country": "United States",
        "Region": st.sidebar.selectbox("Region", ["East", "West", "Central", "South"]),
        "Ship Mode": st.sidebar.selectbox("Ship Mode", ["Second Class", "Standard Class", "First Class", "Same Day"]),
        "Segment": st.sidebar.selectbox("Segment", ["Consumer", "Corporate", "Home Office"]),
        "State": st.sidebar.text_input("State", "California"),
        "City": st.sidebar.text_input("City", "Los Angeles"),
        "Product Name": st.sidebar.text_input("Product Name", "Staples Chair")
    }
    data = pd.DataFrame([row])
    data = safe_preprocess_dates(data)

# ----------------------------
# Tabs: Prediction & Analysis
# ----------------------------
tab1, tab2 = st.tabs(["🤖 Predictions", "📊 Performance Analysis"])

# ----------------------------
# Tab 1: Predictions
# ----------------------------
with tab1:
    st.header("Profit Prediction")
    try:
        data["Predicted_Profit"] = model.predict(data)
        st.success("✅ Prediction completed")
        st.dataframe(data.head(10))
    except Exception as e:
        st.error(f"Prediction error: {e}")

    # Time-based aggregation
    weekly_summary = data.groupby(["Order_Year","Order_Week"])["Predicted_Profit"].sum().reset_index()
    monthly_summary = data.groupby(["Order_Year","Order_Month"])["Predicted_Profit"].sum().reset_index()
    yearly_summary = data.groupby(["Order_Year"])["Predicted_Profit"].sum().reset_index()

    st.subheader("Weekly Predicted Profit")
    fig_week = px.line(weekly_summary, x="Order_Week", y="Predicted_Profit", color="Order_Year")
    st.plotly_chart(fig_week, use_container_width=True)

    st.subheader("Monthly Predicted Profit")
    fig_month = px.line(monthly_summary, x="Order_Month", y="Predicted_Profit", color="Order_Year")
    st.plotly_chart(fig_month, use_container_width=True)

    st.subheader("Yearly Predicted Profit")
    fig_year = px.bar(yearly_summary, x="Order_Year", y="Predicted_Profit")
    st.plotly_chart(fig_year, use_container_width=True)

    # Download predictions
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv", key="download-csv")

# ----------------------------
# Tab 2: Performance Analysis
# ----------------------------
with tab2:
    st.header("📊 Business Performance Analysis")

    # Top & Bottom Products
    top5 = data.groupby("Product Name")["Predicted_Profit"].sum().sort_values(ascending=False).head(5)
    bottom5 = data.groupby("Product Name")["Predicted_Profit"].sum().sort_values(ascending=True).head(5)
    st.subheader(" Top 5 Products")
    st.table(top5.reset_index().rename(columns={"Predicted_Profit":"Total Profit"}))
    st.subheader(" Bottom 5 Products")
    st.table(bottom5.reset_index().rename(columns={"Predicted_Profit":"Total Profit"}))

    # KPI Cards
    st.subheader("🌍 Regional KPIs")
    regions = data["Region"].unique()
    cols = st.columns(len(regions))
    for i, region in enumerate(regions):
        region_data = data[data["Region"] == region]
        total_sales_region = region_data["Sales"].sum()
        total_profit_region = region_data["Predicted_Profit"].sum()
        cols[i].metric(label=f"{region} Sales", value=f"${total_sales_region:,.2f}")
        cols[i].metric(label=f"{region} Predicted Profit", value=f"${total_profit_region:,.2f}")

    st.subheader(" Segment KPIs")
    segments = data["Segment"].unique()
    cols = st.columns(len(segments))
    for i, segment in enumerate(segments):
        seg_data = data[data["Segment"] == segment]
        total_sales_seg = seg_data["Sales"].sum()
        total_profit_seg = seg_data["Predicted_Profit"].sum()
        cols[i].metric(label=f"{segment} Sales", value=f"${total_sales_seg:,.2f}")
        cols[i].metric(label=f"{segment} Predicted Profit", value=f"${total_profit_seg:,.2f}")

    st.subheader("📂 Category KPIs")
    categories = data["Category"].unique()
    cols = st.columns(len(categories))
    for i, cat in enumerate(categories):
        cat_data = data[data["Category"] == cat]
        total_sales_cat = cat_data["Sales"].sum()
        total_profit_cat = cat_data["Predicted_Profit"].sum()
        cols[i].metric(label=f"{cat} Sales", value=f"${total_sales_cat:,.2f}")
        cols[i].metric(label=f"{cat} Predicted Profit", value=f"${total_profit_cat:,.2f}")

    # Performance Charts
    st.subheader("🌎 Regional Sales Performance")
    region_perf = data.groupby("Region")["Predicted_Profit"].sum().sort_values(ascending=False)
    st.bar_chart(region_perf)

    st.subheader(" Segment Performance")
    segment_perf = data.groupby("Segment")["Predicted_Profit"].sum().sort_values(ascending=False)
    fig_segment = px.bar(segment_perf.reset_index(), x="Segment", y="Predicted_Profit", color="Segment", title="Profit by Segment")
    st.plotly_chart(fig_segment, use_container_width=True)

    st.subheader("📂 Category Performance")
    category_perf = data.groupby("Category")["Predicted_Profit"].sum().sort_values(ascending=False)
    fig_category = px.bar(category_perf.reset_index(), x="Category", y="Predicted_Profit", color="Category", title="Profit by Category")
    st.plotly_chart(fig_category, use_container_width=True)

    st.subheader("🔹 Sub-Category Performance")
    subcat_perf = data.groupby("Sub-Category")["Predicted_Profit"].sum().sort_values(ascending=False)
    fig_subcat = px.bar(subcat_perf.reset_index(), x="Sub-Category", y="Predicted_Profit", color="Sub-Category", title="Profit by Sub-Category")
    st.plotly_chart(fig_subcat, use_container_width=True)
















