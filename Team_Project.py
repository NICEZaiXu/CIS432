import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

df = pd.read_csv("heloc_dataset_v1.csv")
df.head()

# Convert target variable to numeric (1 for Good, 0 for Bad)
df["RiskPerformance"] = df["RiskPerformance"].map({"Good": 1, "Bad": 0})

# Replace special values (-9, -8, -7) with NaN
special_values_list = [-9, -8, -7]
df.replace(special_values_list, np.nan, inplace=True)

# Display summary of missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Drop columns with excessive missing values (e.g., more than 40% missing)
threshold = 0.4 * len(df)  # 40% of total rows
df_cleaned = df.dropna(thresh=threshold, axis=1)

# Impute remaining missing values
for col in df_cleaned.columns:
    if df_cleaned[col].dtype == "float64" or df_cleaned[col].dtype == "int64":
        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)  # Using median imputation

# Outlier detection using IQR
Q1 = df_cleaned.quantile(0.25)
Q3 = df_cleaned.quantile(0.75)
IQR = Q3 - Q1

# Define outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Replace outliers with NaN and then impute with median
for col in df_cleaned.columns:
    if df_cleaned[col].dtype in ["float64", "int64"]:
        df_cleaned[col] = np.where((df_cleaned[col] < lower_bound[col]) | (df_cleaned[col] > upper_bound[col]),
                                   np.nan, df_cleaned[col])
        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)

# Display final cleaned dataset info
print("\nFinal Dataset Info:")
print(df_cleaned.info())

# EDA - Patterns of all variables
numerical_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(20, 15))
for i, col in enumerate(numerical_columns):
    plt.subplot(5, 5, i+1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(col)
    plt.tight_layout()
plt.show()

# EDA - Distribution of target variable
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x='RiskPerformance')
plt.title('Distribution of RiskPerformance')

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 10), 
                textcoords='offset points')

plt.show()

# corr analysis
corr_matrix = df_cleaned.corr()

target_corr = corr_matrix['RiskPerformance'].sort_values(ascending=False)

plt.figure(figsize=(10, 6))

sns.barplot(x=target_corr.index, y=target_corr.values, palette='viridis')

plt.title('Correlation with RiskPerformance')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=90)
plt.show()

# 📌 Separate features and target variable

y = df_cleaned["RiskPerformance"]
X = df_cleaned.drop(columns=["RiskPerformance"])


# 📌 Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 📌 Standardize the data (for Logistic Regression & SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 📌 Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# 📌 Train models and evaluate performance
model_performance = {}
for name, model in models.items():
    # Select scaled or unscaled data based on model type
    if name in ["Logistic Regression", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for ROC-AUC
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

    # 📌 Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Store results
    model_performance[name] = {
        "Accuracy": accuracy,
        "ROC-AUC": auc,
        "Classification Report": report
    }

# Print model performance results in a readable format
for model_name, metrics in model_performance.items():
    print(f"\n🔹 Model: {model_name}")
    print(f"   Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   ROC-AUC: {metrics['ROC-AUC']:.4f}")
    print("   Classification Report:")
    print("-" * 50)
    print(pd.DataFrame(metrics["Classification Report"]).transpose())  # Print classification report as table
    print("-" * 50)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 📌 Retrieve the trained Logistic Regression model
logistic_regression_model = models["Logistic Regression"]

# 📌 Extract feature importance (coefficients)
coefficients = logistic_regression_model.coef_[0]  # Logistic regression coefficients
feature_names = X_train.columns  # Get original feature names before scaling

# 📌 Sort features by absolute importance
sorted_indices = np.argsort(np.abs(coefficients))[::-1]
sorted_features = feature_names[sorted_indices]
sorted_coefficients = coefficients[sorted_indices]

# 📌 Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(sorted_features[:10], sorted_coefficients[:10], color='blue')  # Top 10 features
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.title("🔍 Top 10 Feature Importance in Logistic Regression")
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()

import joblib

# Save models
for name, model in models.items():
    joblib.dump(model, f"{name.replace(' ', '_').lower()}.pkl")  # Saves as "logistic_regression.pkl", etc.

# Save scaler
joblib.dump(scaler, "scaler.pkl")

import streamlit as st
import pandas as pd
import joblib  # Load trained models

# ✅ 确保 session_state 变量已初始化，防止 KeyError
st.session_state.setdefault("prediction", None)
st.session_state.setdefault("button_clicked", False)

# 📌 加载模型和 Scaler
models = {
    "Logistic Regression": joblib.load("logistic_regression.pkl"),
    "Random Forest": joblib.load("random_forest.pkl"),
    "Gradient Boosting": joblib.load("gradient_boosting.pkl"),
    "SVM": joblib.load("svm.pkl"),
    "Decision Tree": joblib.load("decision_tree.pkl")
}
scaler = joblib.load("scaler.pkl")

# 📌 Streamlit UI
st.title("🏦 Credit Risk Prediction App")
st.write("🔹 Enter applicant details to predict loan approval or rejection.")

st.sidebar.header("📊 Applicant Details")

# 📌 用户输入（23个特征）
external_risk = st.sidebar.slider("External Risk Estimate", 0, 100, 50)
msince_oldest_trade = st.sidebar.slider("Months Since Oldest Trade", 0, 500, 100)
msince_most_recent_trade = st.sidebar.slider("Months Since Most Recent Trade", 0, 100, 10)
average_m_in_file = st.sidebar.slider("Average Months in File", 0, 200, 50)
num_satisfactory_trades = st.sidebar.slider("Number of Satisfactory Trades", 0, 50, 15)
num_trades_60_ever = st.sidebar.slider("Num Trades 60+ Ever", 0, 20, 5)
num_trades_90_ever = st.sidebar.slider("Num Trades 90+ Ever", 0, 20, 5)
percent_trades_never_delq = st.sidebar.slider("Percent of Trades Never Delinquent", 0, 100, 80)
msince_most_recent_delq = st.sidebar.slider("Months Since Most Recent Delinquency", 0, 100, 10)
max_delq_12m = st.sidebar.slider("Max Delinquency in Last 12M", 0, 10, 5)
max_delq_ever = st.sidebar.slider("Max Delinquency Ever", 0, 10, 5)
num_total_trades = st.sidebar.slider("Number of Total Trades", 0, 50, 10)
num_trades_open_12m = st.sidebar.slider("Num Trades Open in Last 12M", 0, 20, 5)
percent_install_trades = st.sidebar.slider("Percent Installment Trades", 0, 100, 50)
msince_most_recent_inq = st.sidebar.slider("Months Since Most Recent Inquiry", 0, 100, 10)
num_inq_last_6m = st.sidebar.slider("Num Inquiries Last 6M", 0, 20, 2)
num_inq_last_6m_excl7 = st.sidebar.slider("Num Inquiries Last 6M Excl 7 Days", 0, 20, 2)
net_fraction_revolving_burden = st.sidebar.slider("Net Fraction Revolving Burden", 0, 100, 50)
net_fraction_install_burden = st.sidebar.slider("Net Fraction Install Burden", 0, 100, 50)
num_revolving_trades_balance = st.sidebar.slider("Num Revolving Trades With Balance", 0, 20, 5)
num_install_trades_balance = st.sidebar.slider("Num Install Trades With Balance", 0, 20, 5)
num_bank_natl_trades_high_util = st.sidebar.slider("Num Bank/National Trades High Utilization", 0, 20, 5)
percent_trades_balance = st.sidebar.slider("Percent Trades With Balance", 0, 100, 50)

# 📌 Model Selection
st.sidebar.header("🔍 Select Model")
model_choice = st.sidebar.radio("Choose a Model:", list(models.keys()))

# ✅ 预测函数
def make_prediction():
    st.session_state["button_clicked"] = True  # ✅ 确保按钮点击后 session_state 变量被正确更新

    # 📌 创建 DataFrame
    input_data = pd.DataFrame({
        "ExternalRiskEstimate": [external_risk],
        "MSinceOldestTradeOpen": [msince_oldest_trade],
        "MSinceMostRecentTradeOpen": [msince_most_recent_trade],
        "AverageMInFile": [average_m_in_file],
        "NumSatisfactoryTrades": [num_satisfactory_trades],
        "NumTrades60Ever2DerogPubRec": [num_trades_60_ever],
        "NumTrades90Ever2DerogPubRec": [num_trades_90_ever],
        "PercentTradesNeverDelq": [percent_trades_never_delq],
        "MSinceMostRecentDelq": [msince_most_recent_delq],
        "MaxDelq2PublicRecLast12M": [max_delq_12m],
        "MaxDelqEver": [max_delq_ever],
        "NumTotalTrades": [num_total_trades],
        "NumTradesOpeninLast12M": [num_trades_open_12m],
        "PercentInstallTrades": [percent_install_trades],
        "MSinceMostRecentInqexcl7days": [msince_most_recent_inq],
        "NumInqLast6M": [num_inq_last_6m],
        "NumInqLast6Mexcl7days": [num_inq_last_6m_excl7],
        "NetFractionRevolvingBurden": [net_fraction_revolving_burden],
        "NetFractionInstallBurden": [net_fraction_install_burden],
        "NumRevolvingTradesWBalance": [num_revolving_trades_balance],
        "NumInstallTradesWBalance": [num_install_trades_balance],
        "NumBank2NatlTradesWHighUtilization": [num_bank_natl_trades_high_util],
        "PercentTradesWBalance": [percent_trades_balance]
    })

    # 📌 确保特征顺序匹配训练模型
    input_data = input_data[scaler.feature_names_in_]

    # 📌 标准化数据（仅适用于 Logistic Regression 和 SVM）
    if model_choice in ["Logistic Regression", "SVM"]:
        input_data = scaler.transform(input_data)

    # 📌 进行预测
    selected_model = models[model_choice]
    prediction = selected_model.predict(input_data)[0]
    st.session_state["prediction"] = prediction  # 存储预测结果

# ✅ 触发预测的按钮
predict_clicked = st.button("Predict Credit Risk")
if predict_clicked:
    make_prediction()
    
# 📌 显示预测结果
if st.session_state["button_clicked"]:
    if st.session_state["prediction"] is not None:
        result = "✅ Approved" if st.session_state["prediction"] == 1 else "❌ Rejected"
        st.write(f"### 🏦 Loan Decision: {result}")
    else:
        st.warning("⚠️ Click the button to make a prediction.")