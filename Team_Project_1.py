import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# =============================================================================
# 模型训练部分：训练所有模型并保存
# =============================================================================
def train_all_models():
    # 读取数据
    df = pd.read_csv("heloc_dataset_v1.csv")
    
    # 将目标变量转换为数值：Good 为 1，Bad 为 0
    df["RiskPerformance"] = df["RiskPerformance"].map({"Good": 1, "Bad": 0})
    
    # 将特殊值(-9, -8, -7)替换为 NaN
    special_values_list = [-9, -8, -7]
    df.replace(special_values_list, np.nan, inplace=True)
    
    # 删除缺失值过多的列（超过 40% 的缺失值）
    threshold = 0.4 * len(df)
    df_cleaned = df.dropna(thresh=threshold, axis=1)
    
    # 对数值型特征用中位数填充缺失值
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype in ["float64", "int64"]:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    
    # 使用 IQR 方法检测异常值，并将超出边界的值替换为 NaN，再用中位数填充
    Q1 = df_cleaned.quantile(0.25)
    Q3 = df_cleaned.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype in ["float64", "int64"]:
            df_cleaned[col] = np.where((df_cleaned[col] < lower_bound[col]) | (df_cleaned[col] > upper_bound[col]),
                                       np.nan, df_cleaned[col])
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    
    # 分离特征和目标变量
    y = df_cleaned["RiskPerformance"]
    X = df_cleaned.drop(columns=["RiskPerformance"])
    
    # 划分训练集和测试集（80% 训练，20% 测试，分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 对于 Logistic Regression 和 SVM 使用标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 定义所有模型
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }
    
    # 训练所有模型并评估（评估结果仅作参考）
    for name, model in models.items():
        if name in ["Logistic Regression", "SVM"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    # 保存所有模型
    joblib.dump(models["Logistic Regression"], "logistic_regression.pkl")
    joblib.dump(models["Random Forest"], "random_forest.pkl")
    joblib.dump(models["Decision Tree"], "decision_tree.pkl")
    joblib.dump(models["Gradient Boosting"], "gradient_boosting.pkl")
    joblib.dump(models["SVM"], "svm.pkl")
    # 同时保存 scaler（用于 Logistic Regression 和 SVM）
    joblib.dump(scaler, "scaler.pkl")
    print("所有模型和 scaler 已保存。")

# 如果所有模型或 scaler 文件不存在，则训练所有模型
required_files = ["logistic_regression.pkl", "random_forest.pkl", "decision_tree.pkl", 
                  "gradient_boosting.pkl", "svm.pkl", "scaler.pkl"]
if not all(os.path.exists(f) for f in required_files):
    train_all_models()

# =============================================================================
# Streamlit 部署部分：仅使用 Logistic Regression 模型进行预测及拒绝原因反馈
# =============================================================================

# 加载已保存的 Logistic Regression 模型和 scaler
logistic_regression_model = joblib.load("logistic_regression.pkl")
scaler = joblib.load("scaler.pkl")

# 多语言翻译字典
translations = {
    "app_title": {
        "English": "🏦 Credit Risk Prediction App",
        "中文": "🏦 信用风险预测应用",
        "한국어": "🏦 신용 위험 예측 앱",
        "हिंदी": "🏦 क्रेडिट जोखिम भविष्यवाणी ऐप"
    },
    "app_description": {
        "English": "Enter applicant details to predict loan approval or rejection.",
        "中文": "请输入申请人信息以预测贷款批准或拒绝。",
        "한국어": "대출 승인 또는 거절을 예측하기 위해 신청자 세부 정보를 입력하세요.",
        "हिंदी": "ऋण स्वीकृति या अस्वीकृति की भविष्यवाणी करने के लिए आवेदक विवरण दर्ज करें।"
    },
    "sidebar_details": {
        "English": "Applicant Details",
        "中文": "申请人信息",
        "한국어": "신청자 정보",
        "हिंदी": "आवेदक विवरण"
    },
    "predict_button": {
        "English": "Predict Credit Risk",
        "中文": "预测信用风险",
        "한국어": "신용 위험 예측",
        "हिंदी": "क्रेडिट जोखिम की भविष्यवाणी करें"
    },
    "loan_decision": {
        "English": "Loan Decision",
        "中文": "贷款决定",
        "한국어": "대출 결정",
        "हिंदी": "ऋण निर्णय"
    },
    "approved": {
        "English": "✅ Approved",
        "中文": "✅ 批准",
        "한국어": "✅ 승인됨",
        "हिंदी": "✅ स्वीकृत"
    },
    "rejected": {
        "English": "❌ Rejected",
        "中文": "❌ 拒绝",
        "한국어": "❌ 거절됨",
        "हिंदी": "❌ अस्वीकृत"
    },
    "rejection_reasons": {
        "English": "Reasons for Rejection:",
        "中文": "拒绝原因：",
        "한국어": "거절 사유:",
        "हिंदी": "अस्वीकृति के कारण:"
    },
    "enter_value": {
        "English": "Enter value for",
        "中文": "请输入",
        "한국어": "값 입력:",
        "हिंदी": "के लिए मान दर्ज करें"
    },
    # 以下为针对所有变量的通用原因提示模板
    "reason_positive": {
        "English": "{} contributes positively with a value of {:.2f}, increasing the likelihood of approval.",
        "中文": "{} 对批准有正面影响（贡献值：{:.2f}），有助于贷款批准。",
        "한국어": "{} 는 긍정적으로 기여합니다 (기여값: {:.2f}), 승인 가능성을 높입니다.",
        "हिंदी": "{} सकारात्मक रूप से योगदान देता है (योगदान: {:.2f}), स्वीकृति की संभावना बढ़ाता है।"
    },
    "reason_negative": {
        "English": "{} contributes negatively with a value of {:.2f}, reducing the likelihood of approval.",
        "中文": "{} 对批准有负面影响（贡献值：{:.2f}），降低贷款批准的可能性。",
        "한국어": "{} 는 부정적으로 기여합니다 (기여값: {:.2f}), 승인 가능성을 낮춥니다.",
        "हिंदी": "{} नकारात्मक रूप से योगदान देता है (योगदान: {:.2f}), स्वीकृति की संभावना कम करता है।"
    },
    "reason_neutral": {
        "English": "{} has no significant influence (contribution: {:.2f}).",
        "中文": "{} 对批准没有显著影响（贡献值：{:.2f}）。",
        "한국어": "{} 는 유의미한 영향을 미치지 않습니다 (기여값: {:.2f}).",
        "हिंदी": "{} का कोई महत्वपूर्ण प्रभाव नहीं है (योगदान: {:.2f})।"
    }
}

# 语言选择
language = st.sidebar.selectbox("Language / 语言 / 언어 / भाषा", 
                                options=["English", "中文", "한국어", "हिंदी"])

# 应用标题和描述
st.title(translations["app_title"][language])
st.write(translations["app_description"][language])

# 手动输入各项申请人参数（使用 number_input 组件）
st.sidebar.header(translations["sidebar_details"][language])
external_risk = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} External Risk Estimate",
    min_value=0, max_value=100, value=50, step=1
)
msince_oldest_trade = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Months Since Oldest Trade",
    min_value=0, max_value=500, value=100, step=1
)
msince_most_recent_trade = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Months Since Most Recent Trade",
    min_value=0, max_value=100, value=10, step=1
)
average_m_in_file = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Average Months in File",
    min_value=0, max_value=200, value=50, step=1
)
num_satisfactory_trades = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Number of Satisfactory Trades",
    min_value=0, max_value=50, value=15, step=1
)
num_trades_60_ever = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Num Trades 60+ Ever",
    min_value=0, max_value=20, value=5, step=1
)
num_trades_90_ever = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Num Trades 90+ Ever",
    min_value=0, max_value=20, value=5, step=1
)
percent_trades_never_delq = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Percent of Trades Never Delinquent",
    min_value=0, max_value=100, value=80, step=1
)
msince_most_recent_delq = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Months Since Most Recent Delinquency",
    min_value=0, max_value=100, value=10, step=1
)
max_delq_12m = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Max Delinquency in Last 12M",
    min_value=0, max_value=10, value=5, step=1
)
max_delq_ever = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Max Delinquency Ever",
    min_value=0, max_value=10, value=5, step=1
)
num_total_trades = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Number of Total Trades",
    min_value=0, max_value=50, value=10, step=1
)
num_trades_open_12m = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Number of Trades Open in Last 12M",
    min_value=0, max_value=20, value=5, step=1
)
percent_install_trades = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Percent Installment Trades",
    min_value=0, max_value=100, value=50, step=1
)
msince_most_recent_inq = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Months Since Most Recent Inquiry (excl 7 days)",
    min_value=0, max_value=100, value=10, step=1
)
num_inq_last_6m = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Num Inquiries Last 6M",
    min_value=0, max_value=20, value=2, step=1
)
num_inq_last_6m_excl7 = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Num Inquiries Last 6M (excl 7 days)",
    min_value=0, max_value=20, value=2, step=1
)
net_fraction_revolving_burden = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Net Fraction Revolving Burden",
    min_value=0, max_value=100, value=50, step=1
)
net_fraction_install_burden = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Net Fraction Install Burden",
    min_value=0, max_value=100, value=50, step=1
)
num_revolving_trades_balance = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Num Revolving Trades With Balance",
    min_value=0, max_value=20, value=5, step=1
)
num_install_trades_balance = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Num Install Trades With Balance",
    min_value=0, max_value=20, value=5, step=1
)
num_bank_natl_trades_high_util = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Num Bank/National Trades High Utilization",
    min_value=0, max_value=20, value=5, step=1
)
percent_trades_balance = st.sidebar.number_input(
    label=f"{translations['enter_value'][language]} Percent Trades With Balance",
    min_value=0, max_value=100, value=50, step=1
)

# =============================================================================
# 定义基于 Logistic Regression 模型贡献计算的拒绝原因反馈函数
# 对每个变量都提供原因说明
# =============================================================================
def get_logistic_rejection_reasons(input_data, lang):
    # 对输入数据进行标准化
    input_scaled = scaler.transform(input_data)
    # 计算各特征贡献：贡献 = 标准化值 * 模型系数
    contributions = input_scaled[0] * logistic_regression_model.coef_[0]
    # 构建 {特征名: 贡献值} 字典
    feature_contribs = {feature: contrib for feature, contrib in zip(scaler.feature_names_in_, contributions)}
    
    reasons = []
    # 遍历所有变量，生成对应原因说明
    for feature, contrib in feature_contribs.items():
        # 根据贡献值判断说明类型
        if abs(contrib) < 0.05:
            reason = translations["reason_neutral"][lang].format(feature, contrib)
        elif contrib < 0:
            reason = translations["reason_negative"][lang].format(feature, contrib)
        else:
            reason = translations["reason_positive"][lang].format(feature, contrib)
        reasons.append(reason)
    return reasons

# =============================================================================
# 预测函数（仅使用 Logistic Regression 模型）
# =============================================================================
def make_prediction():
    st.session_state["button_clicked"] = True  # 更新状态

    # 构造输入数据，字段名称须与训练时一致
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
    
    # 确保特征顺序与训练时一致
    input_data = input_data[scaler.feature_names_in_]
    
    # 对输入数据进行标准化（因为模型训练时使用了标准化数据）
    input_scaled = scaler.transform(input_data)
    
    # 进行预测
    prediction = logistic_regression_model.predict(input_scaled)[0]
    st.session_state["prediction"] = prediction
    
    # 如果预测为拒绝（0），计算每个变量的贡献并给出详细原因
    if prediction == 0:
        reasons = get_logistic_rejection_reasons(input_data, language)
        st.session_state["reasons"] = reasons
    else:
        st.session_state["reasons"] = []

# =============================================================================
# 初始化状态并展示预测按钮与结果
# =============================================================================
st.session_state.setdefault("prediction", None)
st.session_state.setdefault("button_clicked", False)
st.session_state.setdefault("reasons", [])

if st.button(translations["predict_button"][language]):
    make_prediction()

if st.session_state["button_clicked"]:
    if st.session_state["prediction"] is not None:
        if st.session_state["prediction"] == 1:
            result = translations["approved"][language]
            st.write(f"### {translations['loan_decision'][language]}: {result}")
        else:
            result = translations["rejected"][language]
            st.write(f"### {translations['loan_decision'][language]}: {result}")
            if st.session_state["reasons"]:
                st.write(f"**{translations['rejection_reasons'][language]}**")
                for reason in st.session_state["reasons"]:
                    st.write(f"- {reason}")
            else:
                st.write("No specific rejection reasons provided.")
    else:
        st.warning("⚠️ " + translations["predict_button"][language] + " to make a prediction.")