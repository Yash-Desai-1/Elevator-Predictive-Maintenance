import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Elevator Predictive Maintenance",
                   layout="wide")

st.title("🏢 Elevator Predictive Maintenance")
st.write("Predict future elevator vibration using the trained XGBoost model.")

model = joblib.load(os.getcwd() + "/models/xgb_model.pkl")

uploaded_file = st.file_uploader(
    "Upload Elevator Sensor CSV",
    type="csv"
)

if uploaded_file is not None:

    # -----------------------------
    # Read Dataset
    # -----------------------------
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(df.head())

    # -----------------------------
    # Check Required Columns
    # -----------------------------
    required_columns = [
        "revolutions",
        "humidity",
        "vibration",
        "x1",
        "x2",
        "x3",
        "x4",
        "x5"
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if len(missing) > 0:

        st.error(f"Missing Columns : {missing}")

    else:

        # =====================================
        # Feature Engineering
        # =====================================

        # Lag Features

        df["vib_lag1"] = df["vibration"].shift(1)
        df["vib_lag4"] = df["vibration"].shift(4)
        df["vib_lag16"] = df["vibration"].shift(16)
        df["vib_lag32"] = df["vibration"].shift(32)

        df["rev_lag1"] = df["revolutions"].shift(1)
        df["rev_lag4"] = df["revolutions"].shift(4)

        # Rolling Features

        df["roll_mean_16"] = df["vibration"].rolling(16).mean()
        df["roll_std_16"] = df["vibration"].rolling(16).std()

        df["roll_mean_64"] = df["vibration"].rolling(64).mean()
        df["roll_std_64"] = df["vibration"].rolling(64).std()

        df["roll_max_16"] = df["vibration"].rolling(16).max()

        # Regime Feature

        median_rev = df["revolutions"].median()

        df["is_high_rev"] = (
            df["revolutions"] > median_rev
        ).astype(int)

        # Remove rows having NaN

        df = df.dropna().reset_index(drop=True)

        st.success("Feature Engineering Completed")

        st.subheader("Engineered Dataset")
        st.dataframe(df.head())

        # =====================================
        # Feature List
        # =====================================

        features = [

            "revolutions",
            "humidity",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",

            "vib_lag1",
            "vib_lag4",
            "vib_lag16",
            "vib_lag32",

            "rev_lag1",
            "rev_lag4",

            "roll_mean_16",
            "roll_std_16",
            "roll_mean_64",
            "roll_std_64",
            "roll_max_16",

            "is_high_rev"
        ]

        X = df[features]

        # =====================================
        # Prediction
        # =====================================

        prediction = model.predict(X)

        df["Predicted_Vibration"] = prediction

        # =====================================
        # Health Status
        # =====================================

        def health_status(actual, predicted):

            difference = abs(actual - predicted)

            if difference < 2:
                return "Healthy"

            elif difference < 5:
                return "Warning"

            else:
                return "Maintenance Required"

        df["Status"] = df.apply(
            lambda row: health_status(
                row["vibration"],
                row["Predicted_Vibration"]
            ),
            axis=1
        )

        # =====================================
        # Metrics
        # =====================================

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Records",
            len(df)
        )

        col2.metric(
            "Average Actual",
            round(df["vibration"].mean(),2)
        )

        col3.metric(
            "Average Predicted",
            round(df["Predicted_Vibration"].mean(),2)
        )

        # =====================================
        # Prediction Table
        # =====================================

        st.subheader("Prediction Results")

        st.dataframe(
            df[
                [
                    "vibration",
                    "Predicted_Vibration",
                    "Status"
                ]
            ]
        )

        # =====================================
        # Line Chart
        # =====================================

        st.subheader("Actual vs Predicted")

        chart = df[
            [
                "vibration",
                "Predicted_Vibration"
            ]
        ]

        st.line_chart(chart)

        st.subheader("Maintenance Summary")

        st.bar_chart(df["Status"].value_counts())

        csv = df.to_csv(index=False)

        st.download_button(
            "Download Predictions",
            csv,
            file_name="prediction.csv",
            mime="text/csv"
        )