import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler , OneHotEncoder 
from sklearn.compose import ColumnTransformer
import seaborn as sns
# ---------------------------
# Load data
# ---------------------------
data = pd.read_csv(r"Sample - Superstore.csv", encoding='latin1')
# ---------------------------
# Data Preprocessing
# ---------------------------
# Remove outliers
data = data[(data['Profit'] > data['Profit'].quantile(0.01)) &
            (data['Profit'] < data['Profit'].quantile(0.99))]

# Convert to datetime
data["Order Date"] = pd.to_datetime(data["Order Date"])
data["Ship Date"] = pd.to_datetime(data["Ship Date"])

# Extract features
data["Order_Year"] = data["Order Date"].dt.year
data["Order_Month"] = data["Order Date"].dt.month
data["Order_Day"] = data["Order Date"].dt.day
data["Order_DayOfWeek"] = data["Order Date"].dt.dayofweek
data["Order_Quarter"] = data["Order Date"].dt.quarter
data["Shipping_Days"] = (data["Ship Date"] - data["Order Date"]).dt.days

# Cyclical encoding
data["Month_sin"] = np.sin(2 * np.pi * data["Order_Month"] / 12)
data["Month_cos"] = np.cos(2 * np.pi * data["Order_Month"] / 12)
data["DayOfWeek_sin"] = np.sin(2 * np.pi * data["Order_DayOfWeek"] / 7)
data["DayOfWeek_cos"] = np.cos(2 * np.pi * data["Order_DayOfWeek"] / 7)

# Drop unused columns
data = data.drop(['Row ID', 'Order ID', 'Customer ID', 'Customer Name',
                  'Postal Code', 'Product ID'], axis=1)


# ---------------------------
# Features & Target
# ---------------------------
y = data["Profit"]
X = data.drop(["Profit", "Order Date", "Ship Date"], axis=1)

# ---------------------------
# Define numeric and categorical features
# ---------------------------
numeric_features = ["Sales","Quantity","Discount",
                    "Order_Year","Order_Month","Order_Day",
                    "Order_DayOfWeek","Order_Quarter","Shipping_Days",
                    "Month_sin","Month_cos","DayOfWeek_sin","DayOfWeek_cos"]

categorical_features = ["Category","Sub-Category","Country","Region",
                        "Ship Mode","Segment","State","City","Product Name"]

# ---------------------------
# Preprocessing Pipeline
# ---------------------------
numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ---------------------------
# Safe MLflow setup
# ---------------------------
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:///tmp/mlruns")  # مسار ثابت وآمن للRunner
mlflow.set_experiment("superstore-regression")

# ---------------------------
# Models to test
# ---------------------------
models = {
    "RandomForestRegressor": RandomForestRegressor(random_state=40 , max_depth=20),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=45),
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=45),
    "KNeighborsRegressor": KNeighborsRegressor()
}

results = []

# ---------------------------
# Train + Evaluate
# ---------------------------
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline(steps=[("preprocessor", preprocessor),
                               ("model", model)])
        pipe.fit(x_train, y_train)

        y_pred_train = pipe.predict(x_train)
        y_pred_test = pipe.predict(x_test)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        # Log metrics
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # Log model with signature
        signature = infer_signature(x_train, pipe.predict(x_train))
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            signature=signature,
            input_example=x_train.head(3)
        )

        results.append({
            "Model": name,
            "Train R2": train_r2,
            "Test R2": test_r2,
            "MAE": mae,
            "RMSE": rmse
        })

        print(f"✅ Logged {name} | Test R2: {test_r2:.4f}")

# ---------------------------
# Results
# ---------------------------
results_data = pd.DataFrame(results)
print(results_data.sort_values(by="Test R2", ascending=False))

# ---------------------------
# Save Best Model
# ---------------------------
best_model_name = results_data.sort_values(by="Test R2", ascending=False).iloc[0]["Model"]
best_model = models[best_model_name]

final_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                 ("model", best_model)])
final_pipeline.fit(X, y)

joblib.dump(final_pipeline, "super_store_pipeline_new.pkl")
mlflow.sklearn.log_model(final_pipeline, artifact_path="best_model",
                         signature=infer_signature(X, final_pipeline.predict(X)),
                         input_example=X.head(3))

print(f"🏆 Best model saved: {best_model_name}")

import os

# ---------------------------
# Create artifacts folder
# ---------------------------
os.makedirs("artifacts", exist_ok=True)

# ---------------------------
# Visualizations
# ---------------------------
plt.figure(figsize=(10,6))
sns.barplot(x="Model", y="Test R2", data=results_data)
plt.title("📊 Test R2 Comparison Between Models")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("artifacts/r2_comparison.png")

plt.figure(figsize=(10,6))
sns.barplot(x="Model", y="RMSE", data=results_data)
plt.title("📊 RMSE Comparison Between Models")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("artifacts/rmse_comparison.png")

# ---------------------------
# Save Report
# ---------------------------
with open("artifacts/report.md", "w", encoding="utf-8") as f:
    f.write("## 📊 Model Training Report\n\n")
    f.write(results_data.to_markdown(index=False))
    f.write("\n\n### R2 Comparison\n")
    f.write("![](./r2_comparison.png)\n\n")
    f.write("### RMSE Comparison\n")
    f.write("![](./rmse_comparison.png)\n")

