import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

##########################################
# 1. 多语言支持的配置
##########################################
language_options = {
    "English": {
        "title": "🏦 Credit Risk Prediction App",
        "description": "Enter applicant details to predict loan approval or rejection.",
        "external_risk": "External Risk Estimate",
        "msince_oldest_trade": "Months Since Oldest Trade",
        "msince_most_recent_trade": "Months Since Most Recent Trade",
        "average_m_in_file": "Average Months in File",
        "num_satisfactory_trades": "Number of Satisfactory Trades",
        "num_trades_60_ever": "Num Trades 60+ Ever",
        "num_trades_90_ever": "Num Trades 90+ Ever",
        "percent_trades_never_delq": "Percent of Trades Never Delinquent",
        "msince_most_recent_delq": "Months Since Most Recent Delinquency",
        "max_delq_12m": "Max Delinquency in Last 12M",
        "max_delq_ever": "Max Delinquency Ever",
        "num_total_trades": "Number of Total Trades",
        "num_trades_open_12m": "Num Trades Open in Last 12M",
        "percent_install_trades": "Percent Installment Trades",
        "msince_most_recent_inq": "Months Since Most Recent Inquiry",
        "num_inq_last_6m": "Num Inquiries Last 6M",
        "num_inq_last_6m_excl7": "Num Inquiries Last 6M Excl 7 Days",
        "net_fraction_revolving_burden": "Net Fraction Revolving Burden",
        "net_fraction_install_burden": "Net Fraction Install Burden",
        "num_revolving_trades_balance": "Num Revolving Trades With Balance",
        "num_install_trades_balance": "Num Install Trades With Balance",
        "num_bank_natl_trades_high_util": "Num Bank/National Trades High Utilization",
        "percent_trades_balance": "Percent Trades With Balance",
        "model_selection": "Select Model",
        "predict_button": "Predict Credit Risk",
        "approved": "✅ Approved",
        "rejected": "❌ Rejected",
        "rejection_reason": "Rejection Reasons",
        "train_button": "Train/Reload Models"
    },
    "中文": {
        "title": "🏦 信用风险预测应用",
        "description": "请输入申请人信息以预测贷款批准或拒绝。",
        "external_risk": "外部风险评估",
        "msince_oldest_trade": "距最早交易月份",
        "msince_most_recent_trade": "距最近交易月份",
        "average_m_in_file": "文件中的平均月份数",
        "num_satisfactory_trades": "满意交易数",
        "num_trades_60_ever": "60+天交易数",
        "num_trades_90_ever": "90+天交易数",
        "percent_trades_never_delq": "从未逾期交易百分比",
        "msince_most_recent_delq": "距最近逾期月份",
        "max_delq_12m": "近12个月最大逾期",
        "max_delq_ever": "历史最大逾期",
        "num_total_trades": "总交易数",
        "num_trades_open_12m": "近12个月开启交易数",
        "percent_install_trades": "分期交易百分比",
        "msince_most_recent_inq": "距最近询问月份",
        "num_inq_last_6m": "近6个月询问数",
        "num_inq_last_6m_excl7": "近6个月排除7天询问数",
        "net_fraction_revolving_burden": "循环负债比例",
        "net_fraction_install_burden": "分期负债比例",
        "num_revolving_trades_balance": "有余额循环交易数",
        "num_install_trades_balance": "有余额分期交易数",
        "num_bank_natl_trades_high_util": "银行/全国高利用率交易数",
        "percent_trades_balance": "有余额交易百分比",
        "model_selection": "选择模型",
        "predict_button": "预测信用风险",
        "approved": "✅ 批准",
        "rejected": "❌ 拒绝",
        "rejection_reason": "拒绝原因",
        "train_button": "训练/重新加载模型"
    },
    "한국어": {
        "title": "🏦 신용 위험 예측 앱",
        "description": "대출 승인 또는 거부를 예측하기 위해 신청자 정보를 입력하세요.",
        "external_risk": "외부 위험 평가",
        "msince_oldest_trade": "가장 오래된 거래 이후 개월 수",
        "msince_most_recent_trade": "가장 최근 거래 이후 개월 수",
        "average_m_in_file": "파일의 평균 개월 수",
        "num_satisfactory_trades": "만족스러운 거래 수",
        "num_trades_60_ever": "60일 이상 거래 수",
        "num_trades_90_ever": "90일 이상 거래 수",
        "percent_trades_never_delq": "연체 없는 거래 비율",
        "msince_most_recent_delq": "가장 최근 연체 이후 개월 수",
        "max_delq_12m": "최근 12개월 최대 연체",
        "max_delq_ever": "역대 최대 연체",
        "num_total_trades": "총 거래 수",
        "num_trades_open_12m": "최근 12개월 개시 거래 수",
        "percent_install_trades": "할부 거래 비율",
        "msince_most_recent_inq": "가장 최근 문의 이후 개월 수",
        "num_inq_last_6m": "최근 6개월 문의 수",
        "num_inq_last_6m_excl7": "최근 6개월 7일 제외 문의 수",
        "net_fraction_revolving_burden": "회전 부담 비율",
        "net_fraction_install_burden": "할부 부담 비율",
        "num_revolving_trades_balance": "잔액 있는 회전 거래 수",
        "num_install_trades_balance": "잔액 있는 할부 거래 수",
        "num_bank_natl_trades_high_util": "은행/전국 고활용 거래 수",
        "percent_trades_balance": "잔액 거래 비율",
        "model_selection": "모델 선택",
        "predict_button": "신용 위험 예측",
        "approved": "✅ 승인됨",
        "rejected": "❌ 거부됨",
        "rejection_reason": "거부 사유",
        "train_button": "모델 학습/재로드"
    },
    "हिंदी": {
        "title": "🏦 क्रेडिट जोखिम पूर्वानुमान ऐप",
        "description": "कृपया ऋण स्वीकृति या अस्वीकृति के लिए आवेदनकर्ता विवरण दर्ज करें।",
        "external_risk": "बाहरी जोखिम आकलन",
        "msince_oldest_trade": "सबसे पुराने व्यापार से महीने",
        "msince_most_recent_trade": "सबसे हाल के व्यापार से महीने",
        "average_m_in_file": "फ़ाइल में औसत महीने",
        "num_satisfactory_trades": "संतोषजनक व्यापार की संख्या",
        "num_trades_60_ever": "60+ दिनों के व्यापार",
        "num_trades_90_ever": "90+ दिनों के व्यापार",
        "percent_trades_never_delq": "कभी डिफ़ॉल्ट न होने वाले व्यापार का प्रतिशत",
        "msince_most_recent_delq": "सबसे हाल के डिफ़ॉल्ट से महीने",
        "max_delq_12m": "पिछले 12 महीनों में अधिकतम डिफ़ॉल्ट",
        "max_delq_ever": "अब तक का अधिकतम डिफ़ॉल्ट",
        "num_total_trades": "कुल व्यापारों की संख्या",
        "num_trades_open_12m": "पिछले 12 महीनों में खुले व्यापार",
        "percent_install_trades": "किस्त व्यापार का प्रतिशत",
        "msince_most_recent_inq": "सबसे हाल के पूछताछ से महीने",
        "num_inq_last_6m": "पिछले 6 महीनों में पूछताछ",
        "num_inq_last_6m_excl7": "पिछले 6 महीनों में 7 दिनों को छोड़कर पूछताछ",
        "net_fraction_revolving_burden": "रिवॉल्विंग बर्डन का प्रतिशत",
        "net_fraction_install_burden": "किस्त बर्डन का प्रतिशत",
        "num_revolving_trades_balance": "बैलेंस वाले रिवॉल्विंग व्यापार",
        "num_install_trades_balance": "बैलेंस वाले किस्त व्यापार",
        "num_bank_natl_trades_high_util": "उच्च उपयोग वाले बैंक/राष्ट्रीय व्यापार",
        "percent_trades_balance": "बैलेंस वाले व्यापार का प्रतिशत",
        "model_selection": "मॉडल चुनें",
        "predict_button": "क्रेडिट जोखिम पूर्वानुमान",
        "approved": "✅ स्वीकृत",
        "rejected": "❌ अस्वीकृत",
        "rejection_reason": "अस्वीकृति के कारण",
        "train_button": "मॉडल ट्रेन/रीलोड"
    }
}

# 侧边栏选择语言
selected_language = st.sidebar.selectbox("Select Language / 选择语言 / 언어 선택 / भाषा चुनें", list(language_options.keys()))
lang = language_options[selected_language]

##########################################
# 2. 模型训练函数（含数据清洗、分割、标准化、模型训练与评估）
##########################################
@st.cache(allow_output_mutation=True)
def train_models():
    # 读取数据
    df = pd.read_csv("/Users/lizaixu/Desktop/heloc_dataset_v1.csv")
    # 将目标变量转换为数值（Good:1, Bad:0）
    df["RiskPerformance"] = df["RiskPerformance"].map({"Good": 1, "Bad": 0})
    # 将特殊值替换为 NaN
    special_values_list = [-9, -8, -7]
    df.replace(special_values_list, np.nan, inplace=True)
    # 删除缺失值过多的列（例如超过 40% 的缺失）
    threshold = 0.4 * len(df)
    df_cleaned = df.dropna(thresh=threshold, axis=1)
    # 对数值型变量使用中位数填充缺失值
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype in ["float64", "int64"]:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    # 利用 IQR 方法处理异常值
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
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    # 针对 Logistic Regression 和 SVM 进行数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # 初始化各模型
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }
    model_performance = {}
    # 训练模型并评估
    for name, model in models.items():
        if name in ["Logistic Regression", "SVM"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)
        report = classification_report(y_test, y_pred, output_dict=True)
        model_performance[name] = {
            "Accuracy": accuracy,
            "ROC-AUC": auc,
            "Classification Report": report
        }
    # 保存训练时使用的特征顺序
    feature_names = X.columns.tolist()
    return models, scaler, model_performance, feature_names

##########################################
# 3. 模型训练（点击按钮触发）及展示评估结果
##########################################
st.sidebar.header("Model Training")
if st.sidebar.button(lang["train_button"]):
    st.session_state["trained"] = False  # 强制重新训练

if "trained" not in st.session_state or not st.session_state["trained"]:
    with st.spinner("Training models..."):
        models, scaler, model_performance, feature_names = train_models()
        st.session_state["models"] = models
        st.session_state["scaler"] = scaler
        st.session_state["feature_names"] = feature_names
        st.session_state["model_performance"] = model_performance
        st.session_state["trained"] = True
    st.success("Models trained successfully!")

with st.expander("View Model Performance Metrics"):
    for m, metrics in st.session_state["model_performance"].items():
        st.write(f"**{m}**")
        st.write(f"Accuracy: {metrics['Accuracy']:.4f}, ROC-AUC: {metrics['ROC-AUC']:.4f}")
        st.write("Classification Report:")
        st.write(pd.DataFrame(metrics["Classification Report"]).transpose())

##########################################
# 4. Streamlit 应用预测部分
##########################################
st.title(lang["title"])
st.write(lang["description"])

st.sidebar.header(lang["model_selection"])
# 手动输入参数（使用 number_input）
external_risk = st.sidebar.number_input(lang["external_risk"], min_value=0, max_value=100, value=50)
msince_oldest_trade = st.sidebar.number_input(lang["msince_oldest_trade"], min_value=0, max_value=500, value=100)
msince_most_recent_trade = st.sidebar.number_input(lang["msince_most_recent_trade"], min_value=0, max_value=100, value=10)
average_m_in_file = st.sidebar.number_input(lang["average_m_in_file"], min_value=0, max_value=200, value=50)
num_satisfactory_trades = st.sidebar.number_input(lang["num_satisfactory_trades"], min_value=0, max_value=50, value=15)
num_trades_60_ever = st.sidebar.number_input(lang["num_trades_60_ever"], min_value=0, max_value=20, value=5)
num_trades_90_ever = st.sidebar.number_input(lang["num_trades_90_ever"], min_value=0, max_value=20, value=5)
percent_trades_never_delq = st.sidebar.number_input(lang["percent_trades_never_delq"], min_value=0, max_value=100, value=80)
msince_most_recent_delq = st.sidebar.number_input(lang["msince_most_recent_delq"], min_value=0, max_value=100, value=10)
max_delq_12m = st.sidebar.number_input(lang["max_delq_12m"], min_value=0, max_value=10, value=5)
max_delq_ever = st.sidebar.number_input(lang["max_delq_ever"], min_value=0, max_value=10, value=5)
num_total_trades = st.sidebar.number_input(lang["num_total_trades"], min_value=0, max_value=50, value=10)
num_trades_open_12m = st.sidebar.number_input(lang["num_trades_open_12m"], min_value=0, max_value=20, value=5)
percent_install_trades = st.sidebar.number_input(lang["percent_install_trades"], min_value=0, max_value=100, value=50)
msince_most_recent_inq = st.sidebar.number_input(lang["msince_most_recent_inq"], min_value=0, max_value=100, value=10)
num_inq_last_6m = st.sidebar.number_input(lang["num_inq_last_6m"], min_value=0, max_value=20, value=2)
num_inq_last_6m_excl7 = st.sidebar.number_input(lang["num_inq_last_6m_excl7"], min_value=0, max_value=20, value=2)
net_fraction_revolving_burden = st.sidebar.number_input(lang["net_fraction_revolving_burden"], min_value=0, max_value=100, value=50)
net_fraction_install_burden = st.sidebar.number_input(lang["net_fraction_install_burden"], min_value=0, max_value=100, value=50)
num_revolving_trades_balance = st.sidebar.number_input(lang["num_revolving_trades_balance"], min_value=0, max_value=20, value=5)
num_install_trades_balance = st.sidebar.number_input(lang["num_install_trades_balance"], min_value=0, max_value=20, value=5)
num_bank_natl_trades_high_util = st.sidebar.number_input(lang["num_bank_natl_trades_high_util"], min_value=0, max_value=20, value=5)
percent_trades_balance = st.sidebar.number_input(lang["percent_trades_balance"], min_value=0, max_value=100, value=50)

# 模型选择
model_choice = st.sidebar.radio(lang["model_selection"], list(st.session_state["models"].keys()))

# 定义拒绝原因生成函数（示例逻辑，可根据需要扩展）
def get_rejection_reasons(data):
    reasons = []
    if data["ExternalRiskEstimate"].iloc[0] > 70:
        reasons.append(lang["external_risk"] + " is too high.")
    if data["NumSatisfactoryTrades"].iloc[0] < 10:
        reasons.append(lang["num_satisfactory_trades"] + " is too low.")
    if not reasons:
        reasons.append("The input parameters did not trigger any specific rejection criteria.")
    return reasons

# 定义预测函数
def make_prediction():
    # 构造输入 DataFrame（确保特征顺序与训练时一致）
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
    input_data = input_data[st.session_state["feature_names"]]
    # 对 Logistic Regression 和 SVM 进行标准化处理
    if model_choice in ["Logistic Regression", "SVM"]:
        input_data_transformed = st.session_state["scaler"].transform(input_data)
    else:
        input_data_transformed = input_data
    selected_model = st.session_state["models"][model_choice]
    prediction = selected_model.predict(input_data_transformed)[0]
    st.session_state["prediction"] = prediction
    if prediction == 0:
        reasons = get_rejection_reasons(input_data)
        st.session_state["rejection_reasons"] = reasons
    else:
        st.session_state["rejection_reasons"] = None

# 点击按钮触发预测
if st.button(lang["predict_button"]):
    make_prediction()

# 显示预测结果与拒绝原因（如适用）
if "prediction" in st.session_state and st.session_state["prediction"] is not None:
    if st.session_state["prediction"] == 1:
        st.write(f"### {lang['approved']}")
    else:
        st.write(f"### {lang['rejected']}")
        if st.session_state["rejection_reasons"]:
            st.write("#### " + lang["rejection_reason"] + ":")
            for reason in st.session_state["rejection_reasons"]:
                st.write("- " + reason)