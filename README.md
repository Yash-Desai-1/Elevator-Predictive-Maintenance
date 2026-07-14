# Elevator Vibration Prediction for Predictive Maintenance

## Project Overview

This project explores whether machine learning can be used to predict elevator vibration using historical sensor data. The goal is to support predictive maintenance by identifying patterns that may indicate developing mechanical issues before they become severe.

Predictive maintenance is important because it helps reduce unexpected breakdowns, lowers maintenance costs, and improves the reliability of equipment. In the context of elevators, vibration is a critical signal because abnormal vibration can indicate wear, instability, or early mechanical degradation.

The motivation behind this research was to study whether past sensor behavior can be used to forecast future vibration levels and to understand which features contribute most to that prediction.

## Research Objective

The research question of this project is simple:

> Can historical elevator sensor data be used to predict future vibration values in a way that supports predictive maintenance?

Time-series prediction was chosen because vibration is not an isolated measurement. It depends on previous values and recent system behavior. The model therefore attempts to predict continuous vibration values using current and engineered historical features.

This approach is relevant to predictive maintenance because it enables condition-based monitoring rather than reacting only after failures occur.

## Dataset

The project uses the dataset stored in the data/raw folder and processed through the notebooks in the notebooks folder.

### Data characteristics

- The notebook workflow identifies the data as an elevator door IoT sensor dataset.
- The recordings were sampled at 4 Hz, meaning one data point every 0.25 seconds.
- The total duration of the recording is approximately 7.8 hours.
- The dataset contains an ID column and several sensor-based variables.

### Features used

The project initially used the following base variables:

- revolutions
- humidity
- vibration (target)
- x1
- x2
- x3
- x4
- x5

In the final modeling pipeline, the project expanded this into a richer feature set by adding engineered temporal features.

### Target variable

- Target: vibration

### Number of features

- Base modeling features: 7 input variables
- Final engineered feature set: 19 features used in the XGBoost workflow

## Exploratory Data Analysis

The notebooks perform a detailed exploratory analysis of the dataset before modeling.

### Key findings from EDA

- The dataset shows clear time-series behavior.
- Vibration values are mostly concentrated in lower ranges, with some outliers.
- Revolutions and humidity appear to have meaningful relationships with vibration.
- The project investigated the relationship between revolutions, humidity, and vibration using scatter plots and correlation analysis.
- The notebooks also explored the temporal structure of the data through line plots and boxplots over time.

### Missing values

The notebooks identified missing values in the vibration column and investigated their pattern. A structured missingness pattern was observed, and the project handled this using forward-fill and backward-fill methods to avoid introducing leakage in a time-series setting.

### Correlation analysis

The correlation analysis showed that the auxiliary variables x1 to x4 are almost perfectly correlated with revolutions, while x5 is almost perfectly correlated with humidity. This suggests that these engineered variables are mathematical transformations of the main sensor variables.

### Visualizations created

The notebooks include:

- line plots of revolutions, humidity, and vibration over time
- histograms of vibration distribution
- scatter plots of revolutions vs vibration and humidity vs vibration
- boxplots of sensor distributions
- correlation heatmaps
- event plots for missing values

## Feature Engineering

Feature engineering was a central part of the project. The experiments show that temporal context significantly improves prediction performance.

### 1. Lag Features

Lag features were created by shifting vibration and revolutions values backward in time.

Examples used in the notebooks:

- vib_lag1
- vib_lag4
- vib_lag16
- vib_lag32
- rev_lag1
- rev_lag4

Why they were created:

- Vibration is strongly dependent on recent history.
- Past values carry useful information about the current state of the elevator.

Why they improve prediction:

- They give the model a short memory of previous states.
- They help capture temporal dependencies that simple current-value features cannot represent.

The lag-feature experiment showed a major improvement over the baseline.

### 2. Rolling Mean

Rolling mean features were computed over recent windows of vibration values.

Examples:

- roll_mean_16
- roll_mean_64

Why they were created:

- They summarize recent average vibration behavior.
- They help capture short-term trends.

Why they improve prediction:

- A model can learn whether the system has been moving upward, downward, or staying stable over time.

### 3. Rolling Standard Deviation

Rolling standard deviation features were created to capture recent variability.

Examples:

- roll_std_16
- roll_std_64

Why they were created:

- They describe how unstable recent vibration behavior has been.

Why they improve prediction:

- High variability may suggest abnormal behavior or developing faults.

### 4. Rolling Maximum

A rolling maximum feature was included to highlight recent peaks in vibration.

Example:

- roll_max_16

Why it was created:

- It captures recent high-intensity vibration events.

Why it improves prediction:

- Sudden spikes often matter in predictive maintenance problems.

### 5. Regime Feature

A regime feature was created by comparing revolutions against the median value.

Example:

- is_high_rev

Why it was created:

- The project observed that different operating conditions may produce different vibration behavior.
- The notebook treated this as a simple operating-condition indicator.

Why it helps:

- It gives the model additional context about the current operating regime.

## Machine Learning Pipeline

The project follows a complete machine learning workflow from raw data to prediction.

```text
Raw Data
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Feature Engineering
↓
Temporal Train/Test Split
↓
Model Training
↓
Model Evaluation
↓
Hyperparameter Tuning
↓
Final XGBoost Model
↓
Prediction and Dashboard Output
```

### Workflow summary

1. Load the raw dataset.
2. Clean the data and handle structured missing values.
3. Explore the statistical and visual characteristics of the data.
4. Engineer lag and rolling features.
5. Use a temporal split to evaluate models realistically.
6. Train and compare regression models.
7. Select the strongest-performing model.
8. Save the trained model and use it in the Streamlit app.

## Models Evaluated

The project evaluated several models throughout the experiments.

### 1. Linear Regression

A baseline linear regression model was trained using the original feature set. It was useful as a simple starting point but did not perform well on the temporal test set.

### 2. Random Forest Regressor

Random Forest was tested in several forms:

- with the original features
- with lag features
- with regime-based splitting
- with rolling statistics

### 3. XGBoost Regressor

XGBoost was tested as the final advanced model. It was trained using a richer set of features including original inputs, lag features, rolling statistics, and the regime indicator.

### Why XGBoost became the final model

XGBoost was selected as the final model because it achieved the best overall performance in the notebook experiments after feature engineering and evaluation. It outperformed the earlier baselines and produced the strongest results among the tested models.

## Model Evaluation

The notebooks report the following evaluation metrics for the main experiments.

| Model | MAE | RMSE | R² Score | Notes |
|---|---:|---:|---:|---|
| Mean baseline prediction | 11.1700 | 14.9800 | -0.0510 | Simple baseline using the mean vibration value |
| Linear Regression | 10.9689 | 16.2670 | -0.2390 | Baseline regression model |
| Random Forest with lag features | 3.4104 | 9.3217 | 0.5931 | Significant improvement after temporal features |
| Random Forest with regime split | 4.9501 | 10.2259 | 0.5103 | Regime-specific approach did not outperform the lag-feature model |
| Random Forest with rolling stats | 2.9792 | 7.3781 | 0.7449 | Strong improvement from rolling trend features |
| XGBoost | 2.9375 | 7.2720 | 0.7522 | Best-performing model reported in the notebooks |

### Metric interpretation

- MAE (Mean Absolute Error): Measures the average absolute difference between predicted and actual vibration values. Lower is better.
- RMSE (Root Mean Squared Error): Penalizes larger errors more strongly. Lower is better.
- R² Score: Indicates how much of the variation in vibration is explained by the model. Higher is better. A negative value means the model performed worse than simply predicting the mean.

### Strengths

- The project clearly improved performance through feature engineering.
- Temporal features were highly effective.
- The modeling workflow was designed to avoid unrealistic evaluation by using temporal splitting.

### Limitations

- The project is still a research-oriented and educational implementation rather than a fully production-grade predictive maintenance system.
- The model predicts vibration values rather than directly predicting failure events.
- The dataset appears to be a single monitored system experiment, so generalization to other elevators requires additional data.

## Results

The main results of the project are as follows:

- The initial baseline models performed poorly, showing that simple current-value features were not sufficient.
- Adding lag features led to a major improvement in prediction quality.
- Rolling statistical features further improved performance by capturing short-term trends and variability.
- The best-performing model reported in the notebooks was XGBoost.
- The results suggest that historical sensor information is highly relevant for predicting elevator vibration and supporting predictive maintenance decisions.

## Streamlit Application

A Streamlit dashboard is included in the app folder to make the model usable interactively.

### Dashboard features

- CSV upload for new elevator sensor files
- validation of required input columns
- automatic feature engineering to match the trained pipeline
- vibration prediction using the saved XGBoost model
- health-status classification based on prediction error
- visualization of actual vs predicted vibration
- maintenance summary charts
- download of prediction results as CSV

The app is implemented in [app/app.py](app/app.py) and uses the trained model from [models/xgb_model.pkl](models/xgb_model.pkl).

## Project Structure

```text
ElevatorPredictiveSystem/
│
├── app/
│   ├── app.py
│   └── templates/
├── data/
│   ├── processed/
│   └── raw/
├── models/
│   ├── rf_model.pkl
│   └── xgb_model.pkl
├── notebooks/
│   ├── 01_eda_Feature_engg.ipynb
│   ├── 02_metrics.ipynb
│   ├── 03_linear_regression.ipynb
│   ├── 04_data_leakage_experiment.ipynb
│   ├── 05_lag_features_rf_test.ipynb
│   ├── 06_regmie_split.ipynb
│   ├── 07_rolling_stats.ipynb
│   └── 08_xgBoost.ipynb
├── requirements.txt
└── README.md
```

## Technologies Used

The project uses the following technologies and libraries:

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

## Installation

Clone the project and install the dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

### Run the Streamlit app

```bash
streamlit run app/app.py
```

### Reproduce the notebooks

Open the notebooks in the notebooks folder in order and run them sequentially.

## Future Scope

This project can be expanded in several realistic ways:

- Multi-step forecasting for future vibration values
- Remaining Useful Life (RUL) estimation
- Real-time IoT integration
- Live sensor streaming and monitoring
- Automated maintenance scheduling
- Deep learning models such as LSTM or Transformers
- Cloud deployment and MLOps integration
- Explainable AI using SHAP or feature attribution methods

## Key Learning Outcomes

This project helped develop practical understanding in the following areas:

- end-to-end machine learning workflow design
- time-series feature engineering
- evaluation of models under realistic temporal splits
- comparison of classical and gradient-boosting models
- development of an interactive prediction dashboard
- interpretation of predictive maintenance results

## Key Highlights

- Built a time-series predictive maintenance project for elevator vibration forecasting using Python, scikit-learn, and XGBoost.
- Conducted exploratory data analysis, handled missing values, and investigated temporal patterns in the sensor data.
- Engineered lag-based and rolling-statistical features to improve prediction performance.
- Compared baseline, regression, Random Forest, and XGBoost models using MAE, RMSE, and R² metrics.
- Developed a Streamlit dashboard for CSV-based prediction, visualization, health status classification, and downloadable outputs.

## Conclusion

This project demonstrates that machine learning can be used to model elevator vibration behavior and support predictive maintenance decision-making. By combining sensor analysis, temporal feature engineering, and modern regression models, the work shows how historical system behavior can be transformed into useful predictions. Although the project is still research-oriented and educational in scope, it provides a strong foundation for future work in intelligent maintenance systems and real-time industrial monitoring.

## References

1. Shen, L. J., Lukose, J., & Young, L. C. (2021). *Predictive Maintenance on an Elevator System Using Machine Learning*. Journal of Applied Technology and Innovation.

2. Carvalho, T. P., Soares, F. A. A. M. N., Vita, R., Francisco, R. P., Basto, J. P., & Alcalá, S. G. S. (2019). *A Systematic Literature Review of Machine Learning Methods Applied to Predictive Maintenance*. Computers & Industrial Engineering.
[Link](https://www.sciencedirect.com/science/article/abs/pii/S0360835219304838)

3. Pundir, A., Maheshwari, P., & Prajapati, P. (2022). *Machine Learning Based Predictive Maintenance Model*. Proceedings of the 2nd Indian International Conference on Industrial Engineering and Operations Management.
[Link](https://ieomsociety.org/proceedings/2022india/528.pdf)

4. Stenström, C., et al. *Machine Learning for Predictive Maintenance: A Multiple Classifier Approach*. Chalmers University of Technology.
[Link](https://research.chalmers.se/publication/531022/file/531022_Fulltext.pdf)

5. Sayeed, A., et al. (2025). *Predictive Maintenance Using Machine Learning for Industrial Systems*. Statistics, Optimization & Information Computing.
[Link](https://iapress.org/index.php/soic/article/view/3058)

