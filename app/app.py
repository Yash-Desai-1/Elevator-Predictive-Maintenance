import streamlit as st   
import pandas as pd     
import numpy as np  
import joblib     
import os              

st.set_page_config(
    page_title = "Elevator Predictive Maintenance",
    layout     = "wide"
)

st.sidebar.title("What Do The Statuses Mean?")

# Green
st.sidebar.success("""
### HEALTHY 🟢

The predicted vibration is within the normal operating range observed in the training data.

- Elevator door is operating normally.
- Bearing condition appears healthy.
- No maintenance action is required.
- Continue routine monitoring.
""")
st.sidebar.info("""
### WATCH 🔵

The predicted vibration is slightly higher than normal.

- Early signs of increased vibration.
- Not an immediate problem.
- Monitor the elevator more frequently.
- Schedule an inspection if this status persists.
""")

st.sidebar.warning("""
### WARNING 🟡

The predicted vibration is significantly higher than normal.

- Bearing wear is likely increasing.
- Performance may begin to degrade.
- Maintenance should be scheduled as soon as possible.
- Ignoring this stage may lead to failure.
""")

st.sidebar.error("""
### CRITICAL 🔴

The predicted vibration is among the highest levels observed relative to the training data.

- High probability of severe bearing damage.
- Elevator operation may become unsafe.
- Stop the elevator if possible.
- Contact maintenance immediately.
""")

st.sidebar.divider()

st.sidebar.markdown("### How Are These Statuses Determined?")
st.sidebar.markdown("""
The model does **not** use fixed vibration values.

Instead, the thresholds are calculated automatically from the
training dataset using vibration percentiles:

• 🟢 HEALTHY → Below the 68th percentile
                    
• 🔵 WATCH → Between the 68th and 95th 
                    
• 🟡 WARNING → Between the 95th and 99th percentile
                    
• 🔴 CRITICAL → Above the 99th percentile

This makes the health assessment adaptive to the
characteristics of the training data.
""")

st.sidebar.divider()

st.sidebar.markdown("### Prediction Horizon")
st.sidebar.markdown("""
Bearing failure is a SLOW process.
It takes hours to days, not seconds.
So even though our model predicts one row at a time (every 0.25 sec),
we get **1 to 48 hours of advance warning** because the WARNING stage lasts a long time before CRITICAL is reached.
""")
# main page

st.title("🏢 Elevator Predictive Maintenance")
st.write("Upload elevator sensor data. The model predicts vibration and tells you if maintenance is needed.")

# st.info() shows a blue information box
st.info("""
**How this works (in simple words):**
1. You upload a CSV with sensor readings (revolutions, humidity, vibration)
2. The app adds extra time-based features (lag features, rolling averages)
3. Our trained XGBoost model predicts the vibration level
4. Based on predicted vibration, the app gives a health status
5. You can download the results
""")

#load the model
model_path = os.path.join(os.getcwd(), "models", "xgb_model.pkl")

data_path = os.path.join(os.getcwd(), "data", "processed",
                        "elevator_with_lag_rollstats_regime.csv")
full_df = pd.read_csv(data_path)

# try/except = attempt something, and if it fails, handle the error gracefully
try:
    model = joblib.load(model_path)    # load the model into memory
    st.sidebar.success(" Model loaded successfully")

except FileNotFoundError:
    # This runs only if the model file does not exist
    st.error(f" Model file not found at: {model_path}")
    st.error("Make sure xgb_model.pkl is inside a folder called 'models'")
    st.stop()    # stop the app here — nothing works without the model

except Exception as e:
    # This runs for any other error while loading
    st.error(f" Error loading model: {e}")
    st.stop()

# take a input file from user in csv format
st.divider()   # draws a horizontal line

uploaded_file = st.file_uploader(
    label = "📁 Upload Elevator Sensor CSV File",
    type  = ["csv"],
    help  = "Upload a CSV with at least these columns: revolutions, humidity, vibration"
)

# check if file is uploaded and then process it

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Uploaded Data (first 5 rows)")
    st.dataframe(df.head())    # st.dataframe() shows a scrollable table

    rows, cols = df.shape
    st.write(f"Dataset size: **{rows:,} rows** and **{cols} columns**")

    # checking if user has uploaded the data with required columns
    required = ["revolutions", "humidity", "vibration"]

    missing_cols = [col for col in required if col not in df.columns]

    if len(missing_cols) > 0:
        st.error(f" These required columns are missing: {missing_cols}")
        st.warning("Your CSV must have at minimum: revolutions, humidity, vibration")
        st.stop()   # stop here — cannot continue without required columns
    else:
        st.success(" All required columns found")

    # In the original dataset, x1-x5 come pre-computed.
    # But if someone uploads a new CSV without them, we can compute
    # them ourselves because we know the exact formulas in eda :
    #   x1 = revolutions + humidity
    #   x2 = revolutions - humidity
    #   x3 = revolutions / humidity
    #   x4 = revolutions squared
    #   x5 = humidity squared

    # Checking if x1 is missing (if x1 is missing, all x1-x5 are probably missing)
    if "x1" not in df.columns:
        st.info("ℹ️ Columns x1 to x5 not found in your file. Computing them automatically...")

        df["x1"] = df["revolutions"] + df["humidity"]       # sum
        df["x2"] = df["revolutions"] - df["humidity"]       # difference
        df["x3"] = df["revolutions"] / df["humidity"]       # ratio
        df["x4"] = df["revolutions"] ** 2                   # revolutions squared
        df["x5"] = df["humidity"] ** 2                      # humidity squared

        st.success("x1 to x5 computed from revolutions and humidity")
    else:
        st.success("x1 to x5 columns found in uploaded file")

    # handing missing values in test data
    missing_before = df["vibration"].isnull().sum()   # count NaN values
    df["vibration"] = df["vibration"].ffill().bfill()
    missing_after  = df["vibration"].isnull().sum()

    if missing_before > 0:
        st.info(f"ℹFixed {missing_before} missing vibration values using forward-fill")

    # feature engineering for test data
    st.subheader("Feature Engineering")
    st.write("Adding lag features and rolling statistics...")

    # LAG FEATURES
    # .shift(N) moves the column DOWN by N rows
    # This means: row 10's vib_lag4 = what vibration was at row 6 (4 rows ago)
    # In time terms: 4 rows × 0.25 seconds = 1 second ago

    df["vib_lag1"]  = df["vibration"].shift(1)    # vibration 0.25 seconds ago
    df["vib_lag4"]  = df["vibration"].shift(4)    # vibration 1.0 second ago
    df["vib_lag16"] = df["vibration"].shift(16)   # vibration 4.0 seconds ago
    df["vib_lag32"] = df["vibration"].shift(32)   # vibration 8.0 seconds ago

    df["rev_lag1"] = df["revolutions"].shift(1)   # motor speed 0.25 sec ago
    df["rev_lag4"] = df["revolutions"].shift(4)   # motor speed 1.0 sec ago

    # ROLLING FEATURES 
    # .rolling(N).mean() slides a window of N rows and computes average
    # At each row, it looks at the last N rows and takes the mean/std/max

    df["roll_mean_16"] = df["vibration"].rolling(16).mean()  # avg over last 4 sec
    df["roll_std_16"]  = df["vibration"].rolling(16).std()   # variability over last 4 sec
    df["roll_mean_64"] = df["vibration"].rolling(64).mean()  # avg over last 16 sec
    df["roll_std_64"]  = df["vibration"].rolling(64).std()   # variability over last 16 sec
    df["roll_max_16"]  = df["vibration"].rolling(16).max()   # peak value over last 4 sec

    # REGIME FEATURE 
    # 1 = door is opening (high revolutions)
    # 0 = door is closing (low revolutions)
    # .median() returns the middle value
    # .astype(int) converts True/False to 1/0

    df["is_high_rev"] = (df["revolutions"] > full_df["revolutions"].median()).astype(int)

    # remove NaN rows after feature engineering
    # Lag and rolling features create NaN in the first N rows
    # (because there is no "1 row before row 0")
    # .dropna() removes all rows that have any NaN
    # .reset_index(drop=True) renumbers the rows from 0 again

    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_after  = len(df)

    st.success(f" Feature engineering complete")
    st.write(f"Rows before cleanup: {rows_before:,} | Rows after: {rows_after:,} | Removed: {rows_before - rows_after}")

    # Check we have enough rows to proceed
    if len(df) < 10:
        st.error("Too few rows after feature engineering. Please upload at least 100+ rows.")
        st.stop()

    # extracting features for prediction

    features = [
        "revolutions", "humidity", "x1", "x2", "x3", "x4", "x5",
        "vib_lag1", "vib_lag4", "vib_lag16", "vib_lag32",
        "rev_lag1", "rev_lag4",
        "roll_mean_16", "roll_std_16", "roll_mean_64", "roll_std_64",
        "roll_max_16", "is_high_rev"
    ]

    X = df[features]

    # predicted values of vibration using our model
    df["Predicted_Vibration"] = model.predict(X)
    df["Predicted_Vibration"] = df["Predicted_Vibration"].round(2)

 
    split = int(len(full_df) * 0.8)
    train_df = full_df[:split]

    zone_a_limit = train_df['vibration'].quantile(0.68)  # Good - Satisfactory boundary
    zone_b_limit = train_df['vibration'].quantile(0.95)  # Satisfactory - Unsatisfactory boundary
    zone_c_limit = train_df['vibration'].quantile(0.99)  # Unsatisfactory - Unacceptable boundary

    # logic for machine staus based on predicted vibration
    def health_status(predicted_vibration):
        """
        Classifies door health based on predicted vibration value.

        Parameters:
            predicted_vibration: the number our XGBoost model predicted

        Returns:
            a string describing the health status

        Thresholds are calculated based on quantile values of vibrations on 68, 95 and 99
        """ 
        if predicted_vibration < zone_a_limit:
            return "HEALTHY"      # safe zone

        elif predicted_vibration < zone_b_limit:
            return "WATCH"        # slightly elevated

        elif predicted_vibration < zone_c_limit:
            return "WARNING"      # needs attention soon

        else:
            return "CRITICAL"     # stop elevator now
        

    df["Status"] = df["Predicted_Vibration"].apply(health_status)



    st.divider()

    # summary of the predicted data
    st.subheader("📊 Summary Metrics")

    # Count how many rows fall into each status
    healthy_count  = (df["Status"] == "HEALTHY").sum()
    watch_count    = (df["Status"] == "WATCH").sum()
    warning_count  = (df["Status"] == "WARNING").sum()
    critical_count = (df["Status"] == "CRITICAL").sum()

    # Display in 4 columns side by side
    col1, col2, col3, col4 = st.columns(4)

    # st.metric() shows a number with a label
    col1.metric("Total Records",       f"{len(df):,}")
    col2.metric("Avg Actual Vibration",    round(df["vibration"].mean(), 2))
    col3.metric("Avg Predicted Vibration", round(df["Predicted_Vibration"].mean(), 2))
    col4.metric("Critical Readings",   critical_count)


    # ── FOUR STATUS COUNTS ────────────────────────────────────
    st.subheader("🔢 Status Breakdown")

    c1, c2, c3, c4 = st.columns(4)

    # Each status gets its own colored box with count
    # adding count of status and it's percentage in it
    c1.success(f"HEALTHY\n\n{healthy_count} readings\n\n({healthy_count/len(df)*100:.1f}%)")
    c2.info(   f"WATCH\n\n{watch_count} readings\n\n({watch_count/len(df)*100:.1f}%)")
    c3.warning(f"WARNING\n\n{warning_count} readings\n\n({warning_count/len(df)*100:.1f}%)")
    c4.error(  f"CRITICAL\n\n{critical_count} readings\n\n({critical_count/len(df)*100:.1f}%)")


    # alert for current door status based on last row's values
    # Show the status of the LAST prediction (most recent reading)
    st.subheader("🔔 Current Door Status (Latest Reading)")

    last_predicted = df["Predicted_Vibration"].iloc[-1]   # .iloc[-1] = last row
    last_status    = df["Status"].iloc[-1]

    # Show appropriate colored box based on last status
    if "HEALTHY" in last_status:
        st.success(f"""
        HEALTHY
        Latest predicted vibration: **{last_predicted:.1f}**
        The door is operating normally. No action needed.
        """)

    elif "WATCH" in last_status:
        st.info(f"""
        WATCH
        Latest predicted vibration: **{last_predicted:.1f}**
        Vibration is slightly elevated. Increase monitoring frequency.
        """)

    elif "WARNING" in last_status:
        st.warning(f"""
        WARNING
        Latest predicted vibration: **{last_predicted:.1f}**
        High vibration detected! Schedule a maintenance inspection within 24 hours.
        """)

    else:
        st.error(f"""
        CRITICAL
        Latest predicted vibration: **{last_predicted:.1f}**
        STOP THE ELEVATOR IMMEDIATELY. Call maintenance now.
        """)


    # displyaing table with actual values, predicted values and status
    st.divider()
    st.subheader("📋 Prediction Results Table")
    st.write("Showing actual vibration, our model's prediction, and the health status for each reading.")

    # Show selected columns only (not all 20+ engineered columns)
    display_df = df[["vibration", "Predicted_Vibration", "Status"]].copy()

    # Rename columns to be more human-readable
    display_df.columns = ["Actual Vibration", "Predicted Vibration", "Health Status"]

    st.dataframe(display_df, use_container_width=True)


    # line chart actual vs predicted
    st.divider()
    st.subheader("📈 Actual vs Predicted Vibration Over Time")
    st.write("The closer the two lines are, the more accurate our model is.")

    # Create a smaller DataFrame just for the chart
    chart_data = df[["vibration", "Predicted_Vibration"]].rename(columns={
        "vibration"            : "Actual Vibration",
        "Predicted_Vibration"  : "Predicted Vibration"
    })

    st.line_chart(chart_data)


    # maintenance report with bar chart based on status
    st.divider()
    st.subheader("📋 Maintenance Report")

    # Build a summary table
    report_data = {
        "Status"     : ["HEALTHY", "WATCH", "WARNING", "CRITICAL"],
        "Count"      : [healthy_count, watch_count, warning_count, critical_count],
        "Percentage" : [
            f"{healthy_count  / len(df) * 100:.1f}%",
            f"{watch_count    / len(df) * 100:.1f}%",
            f"{warning_count  / len(df) * 100:.1f}%",
            f"{critical_count / len(df) * 100:.1f}%"
        ],
        "Action Required": [
            "None — monitor normally",
            "Increase monitoring frequency",
            "Book inspection within 24 hours",
            "Stop elevator — immediate repair"
        ]
    }

    # pd.DataFrame() turns a dictionary into a table
    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    # Bar chart of status counts
    st.write("Status distribution:")
    st.bar_chart(
        df["Status"].value_counts()   # .value_counts() counts each unique value
    )


    # ── OVERALL RECOMMENDATION ────────────────────────────────
    st.divider()
    st.subheader("💡 Overall Recommendation")

    if critical_count > 0:
        st.error(f"""
        🚨 URGENT ACTION REQUIRED

        {critical_count} CRITICAL readings were detected in this dataset.

        What to do:
        1. Stop the elevator from accepting passengers immediately
        2. Call your maintenance team now
        3. Do not resume operation until the bearing is inspected and replaced if needed

        Time to act: RIGHT NOW
        """)

    elif warning_count > 0:
        st.warning(f"""
        ⚠️ MAINTENANCE RECOMMENDED

        {warning_count} WARNING readings were detected.

        What to do:
        1. Notify your maintenance team today
        2. Schedule an inspection within the next 24 hours
        3. Keep the elevator running but monitor closely
        4. Order a replacement bearing in advance

        Time to act: Within 24 hours
        """)

    elif watch_count > (len(df) * 0.3):
        # If more than 30% of readings are in WATCH zone
        st.info(f"""
        👀 MONITORING RECOMMENDED

        {watch_count} WATCH readings detected ({watch_count/len(df)*100:.0f}% of total).

        What to do:
        1. Increase monitoring frequency
        2. Check again in a few hours
        3. If it continues rising, schedule inspection

        Time to act: Within 48 hours
        """)

    else:
        st.success(f"""
         ELEVATOR IS HEALTHY

        All readings are in the normal range.

        What to do:
        1. Continue regular monitoring
        2. No maintenance action needed at this time
        3. Next scheduled check: follow your normal maintenance calendar
        """)


    # allowing users to download the results
    st.divider()
    st.subheader("⬇️ Download Results")
    st.write("Download the predictions as a CSV file to share with your maintenance team.")

    # Prepare the download file
    download_df = df[["vibration", "Predicted_Vibration", "Status"]].copy()
    download_df.columns = ["Actual_Vibration", "Predicted_Vibration", "Health_Status"]
    csv_string = download_df.to_csv(index=False)

    # st.download_button() shows a button that downloads the file when clicked
    st.download_button(
        label     = "⬇️ Download Predictions as CSV",
        data      = csv_string,
        file_name = "elevator_predictions.csv",
        mime      = "text/csv"   
    )

else:
    # This runs when NO file has been uploaded yet
    # Give the user clear instructions

    st.divider()
    st.info("👆 Upload your elevator CSV file above to get started")

    st.markdown("""
    ### What columns should my CSV have?

    **Required (must have these):**
    - `revolutions` — motor speed in RPM
    - `humidity` — air humidity in %
    - `vibration` — door vibration (this is also what we predict)

    **Optional (we compute these if missing):**
    - `x1`, `x2`, `x3`, `x4`, `x5` — derived sensor features

    **Never include in CSV (app computes these automatically):**
    - lag features (vib_lag1, vib_lag4, etc.)
    - rolling features (roll_mean_16, roll_std_16, etc.)
    - is_high_rev

    ### Minimum rows needed:
    At least **100 rows** are needed for lag features to work properly.
    The more rows the better. Our original dataset had 112,001 rows.
    """)