import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

def test_pipeline_fit():
    # بيانات وهمية صغيرة للاختبار
    df = pd.DataFrame({
        "Sales": [100, 200, 150],
        "Quantity": [2, 3, 1],
        "Discount": [0.1, 0.2, 0.0],
        "Category": ["Furniture", "Technology", "Office Supplies"],
        "Sub-Category": ["Bookcases", "Phones", "Binders"],
        "Region": ["East", "West", "Central"],
        "Profit": [20, 50, 30]
    })

    X = df.drop("Profit", axis=1)
    y = df["Profit"]

    numeric_features = ["Sales", "Quantity", "Discount"]
    categorical_features = ["Category", "Sub-Category", "Region"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

    pipe.fit(X, y)

    preds = pipe.predict(X)
    assert len(preds) == len(y)
