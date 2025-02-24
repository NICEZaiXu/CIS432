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

# =============================================================================
# 多语言翻译字典（应用界面相关）
# =============================================================================
translations = {
    "app_title": {
        "English": "🏦 Credit Risk Prediction App",
        "中文": "🏦 信用风险预测应用",
        "한국어": "🏦 신용 위험 예측 앱",
        "हिंदी": "🏦 क्रेडिट जोखिम भविष्यवाणी ऐप",
        "עברית": "אפליקציית חיזוי סיכון אשראי"
    },
    "app_description": {
        "English": "Enter applicant details to predict loan approval or rejection.",
        "中文": "请输入申请人信息以预测贷款批准或拒绝。",
        "한국어": "대출 승인 또는 거절을 예측하기 위해 신청자 세부 정보를 입력하세요.",
        "हिंदी": "ऋण स्वीकृति या अस्वीकृति की भविष्यवाणी करने के लिए आवेदक विवरण दर्ज करें।",
        "עברית": "הזן פרטי מבקש ההלוואה כדי לחזות אם ההלוואה תאושר או תידחה."
    },
    "sidebar_details": {
        "English": "Applicant Details",
        "中文": "申请人信息",
        "한국어": "신청자 정보",
        "हिंदी": "आवेदक विवरण",
        "עברית": "פרטי המבקש"
    },
    "predict_button": {
        "English": "Predict Credit Risk",
        "中文": "预测信用风险",
        "한국어": "신용 위험 예측",
        "हिंदी": "क्रेडिट जोखिम की भविष्यवाणी करें",
        "עברית": "נבא סיכון אשראי"
    },
    "loan_decision": {
        "English": "Loan Decision",
        "中文": "贷款决定",
        "한국어": "대출 결정",
        "हिंदी": "ऋण निर्णय",
        "עברית": "החלטת הלוואה"
    },
    "approved": {
        "English": "✅ Approved",
        "中文": "✅ 批准",
        "한국어": "✅ 승인됨",
        "हिंदी": "✅ स्वीकृत",
        "עברית": "✅ מאושר"
    },
    "rejected": {
        "English": "❌ Rejected",
        "中文": "❌ 拒绝",
        "한국어": "❌ 거절됨",
        "हिंदी": "❌ अस्वीकृत",
        "עברית": "❌ נדחה"
    },
    "rejection_reasons": {
        "English": "Reasons for Rejection:",
        "中文": "拒绝原因：",
        "한국어": "거절 사유:",
        "हिंदी": "अस्वीकृति के कारण:",
        "עברית": "סיבות לדחייה:"
    },
    "enter_value": {
        "English": "Enter value for",
        "中文": "请输入",
        "한국어": "값 입력:",
        "हिंदी": "के लिए मान दर्ज करें",
        "עברית": "הזן ערך עבור"
    },
    # 以下为针对贡献说明的多语言模板
    "reason_positive": {
        "English": "{} contributes positively with a value of {:.2f}, increasing the likelihood of approval.",
        "中文": "{} 对批准有正面影响（贡献值：{:.2f}），有助于贷款批准。",
        "한국어": "{} 는 긍정적으로 기여합니다 (기여값: {:.2f}), 승인 가능성을 높입니다.",
        "हिंदी": "{} सकारात्मक रूप से योगदान देता है (योगदान: {:.2f}), स्वीकृति की संभावना बढ़ाता है।",
        "עברית": "{} תורם באופן חיובי עם ערך של {:.2f}, ומעלה את הסיכוי לאישור."
    },
    "reason_negative": {
        "English": "{} contributes negatively with a value of {:.2f}, reducing the likelihood of approval.",
        "中文": "{} 对批准有负面影响（贡献值：{:.2f}），降低贷款批准的可能性。",
        "한국어": "{} 는 부정적으로 기여합니다 (기여값: {:.2f}), 승인 가능성을 낮춥니다.",
        "हिंदी": "{} नकारात्मक रूप से योगदान देता है (योगदान: {:.2f}), स्वीकृति की संभावना कम करता है।",
        "עברית": "{} תורם באופן שלילי עם ערך של {:.2f}, ומפחית את הסיכוי לאישור."
    },
    "reason_neutral": {
        "English": "{} has no significant influence (contribution: {:.2f}).",
        "中文": "{} 对批准没有显著影响（贡献值：{:.2f}）。",
        "한국어": "{} 는 유의미한 영향을 미치지 않습니다 (기여값: {:.2f}).",
        "हिंदी": "{} का कोई महत्वपूर्ण प्रभाव नहीं है (योगदान: {:.2f})।",
        "עברית": "{} אינו משפיע באופן משמעותי (תרומה: {:.2f})."
    }
}

# =============================================================================
# 变量名称翻译字典（feature_translations）
# =============================================================================
feature_translations = {
    "ExternalRiskEstimate": {
        "English": "External Risk Estimate",
        "中文": "外部风险评估",
        "한국어": "외부 위험 평가",
        "हिंदी": "बाहरी जोखिम अनुमान",
        "עברית": "הערכת סיכון חיצונית"
    },
    "MSinceOldestTradeOpen": {
        "English": "Months Since Oldest Trade Open",
        "中文": "最早交易开放至今的月份数",
        "한국어": "최초 거래 개시 이후 경과 개월",
        "हिंदी": "सबसे पुरानी ट्रेड खुलने के बाद से महीने",
        "עברית": "מספר החודשים מאז פתיחת העסקה הוותיקה ביותר"
    },
    "MSinceMostRecentTradeOpen": {
        "English": "Months Since Most Recent Trade Open",
        "中文": "最近交易开放至今的月份数",
        "한국어": "최근 거래 개시 이후 경과 개월",
        "हिंदी": "हाल की ट्रेड खुलने के बाद से महीने",
        "עברית": "מספר החודשים מאז פתיחת העסקה האחרונה"
    },
    "AverageMInFile": {
        "English": "Average Months in File",
        "中文": "文件中的平均月份数",
        "한국어": "파일 내 평균 개월 수",
        "हिंदी": "फ़ाइल में औसत महीने",
        "עברית": "ממוצע החודשים בתיק"
    },
    "NumSatisfactoryTrades": {
        "English": "Number of Satisfactory Trades",
        "中文": "满意交易数量",
        "한국어": "만족 거래 수",
        "हिंदी": "संतोषजनक ट्रेड की संख्या",
        "עברית": "מספר עסקאות מספקות"
    },
    "NumTrades60Ever2DerogPubRec": {
        "English": "Number of Trades 60+ Ever",
        "中文": "60+交易次数",
        "한국어": "60+ 거래 수",
        "हिंदी": "60+ ट्रेड की संख्या",
        "עברית": "מספר עסקאות 60+ אי פעם"
    },
    "NumTrades90Ever2DerogPubRec": {
        "English": "Number of Trades 90+ Ever",
        "中文": "90+交易次数",
        "한국어": "90+ 거래 수",
        "हिंदी": "90+ ट्रेड की संख्या",
        "עברית": "מספר עסקאות 90+ אי פעם"
    },
    "PercentTradesNeverDelq": {
        "English": "Percent of Trades Never Delinquent",
        "中文": "从未违约的交易百分比",
        "한국어": "연체 기록 없는 거래 비율",
        "हिंदी": "कभी डिफॉल्ट न हुए ट्रेड का प्रतिशत",
        "עברית": "אחוז העסקאות שמעולם לא היו בפיגור"
    },
    "MSinceMostRecentDelq": {
        "English": "Months Since Most Recent Delinquency",
        "中文": "最近违约至今的月份数",
        "한국어": "최근 연체 이후 경과 개월",
        "हिंदी": "हाल ही में डिफॉल्ट से बीते महीने",
        "עברית": "מספר החודשים מאז הפיגור האחרון"
    },
    "MaxDelq2PublicRecLast12M": {
        "English": "Max Delinquency in Last 12M",
        "中文": "过去12个月内最大违约次数",
        "한국어": "최근 12개월 내 최대 연체",
        "हिंदी": "पिछले 12 महीनों में अधिकतम डिफॉल्ट",
        "עברית": "הפיגור המרבי ב-12 החודשים האחרונים"
    },
    "MaxDelqEver": {
        "English": "Max Delinquency Ever",
        "中文": "历史最大违约次数",
        "한국어": "전체 기간 최대 연체",
        "हिंदी": "अब तक का अधिकतम डिफॉल्ट",
        "עברית": "הפיגור המרבי אי פעם"
    },
    "NumTotalTrades": {
        "English": "Number of Total Trades",
        "中文": "交易总数",
        "한국어": "전체 거래 수",
        "हिंदी": "कुल ट्रेड की संख्या",
        "עברית": "מספר כל העסקאות"
    },
    "NumTradesOpeninLast12M": {
        "English": "Number of Trades Open in Last 12M",
        "中文": "过去12个月内开启的交易数量",
        "한국어": "최근 12개월 내 개시된 거래 수",
        "हिंदी": "पिछले 12 महीनों में खुली ट्रेड की संख्या",
        "עברית": "מספר העסקאות הפתוחות ב-12 החודשים האחרונים"
    },
    "PercentInstallTrades": {
        "English": "Percent Installment Trades",
        "中文": "分期交易百分比",
        "한국어": "할부 거래 비율",
        "हिंदी": "किस्त ट्रेड का प्रतिशत",
        "עברית": "אחוז העסקאות בתשלומים"
    },
    "MSinceMostRecentInqexcl7days": {
        "English": "Months Since Most Recent Inquiry (excl 7 days)",
        "中文": "最近查询（排除7天）至今的月份数",
        "한국어": "최근 문의(7일 제외) 이후 경과 개월",
        "हिंदी": "हाल की पूछताछ (7 दिनों को छोड़कर) के बाद से महीने",
        "עברית": "מספר החודשים מאז השאילתה האחרונה (ללא 7 ימים)"
    },
    "NumInqLast6M": {
        "English": "Number of Inquiries Last 6M",
        "中文": "过去6个月内查询次数",
        "한국어": "최근 6개월 내 문의 수",
        "हिंदी": "पिछले 6 महीनों में पूछताछ की संख्या",
        "עברית": "מספר השאילתות ב-6 החודשים האחרונים"
    },
    "NumInqLast6Mexcl7days": {
        "English": "Number of Inquiries Last 6M (excl 7 days)",
        "中文": "过去6个月内查询次数（排除7天）",
        "한국어": "최근 6개월 내 문의 수 (7일 제외)",
        "हिंदी": "पिछले 6 महीनों में पूछताछ की संख्या (7 दिनों को छोड़कर)",
        "עברית": "מספר השאילתות ב-6 החודשים האחרונים (ללא 7 ימים)"
    },
    "NetFractionRevolvingBurden": {
        "English": "Net Fraction Revolving Burden",
        "中文": "循环负债净比例",
        "한국어": "순환 부채 비율",
        "हिंदी": "घूर्णन ऋण का शुद्ध अनुपात",
        "עברית": "חלק נטו של נטל מחזורי"
    },
    "NetFractionInstallBurden": {
        "English": "Net Fraction Install Burden",
        "中文": "分期负债净比例",
        "한국어": "할부 부채 비율",
        "हिंदी": "किस्त ऋण का शुद्ध अनुपात",
        "עברית": "חלק נטו של נטל בתשלומים"
    },
    "NumRevolvingTradesWBalance": {
        "English": "Number of Revolving Trades With Balance",
        "中文": "有余额的循环交易数量",
        "한국어": "잔액이 있는 순환 거래 수",
        "हिंदी": "बैलेंस वाले घूर्णन ट्रेड की संख्या",
        "עברית": "מספר העסקאות המחזוריות עם יתרה"
    },
    "NumInstallTradesWBalance": {
        "English": "Number of Install Trades With Balance",
        "中文": "有余额的分期交易数量",
        "한국어": "잔액이 있는 할부 거래 수",
        "हिंदी": "बैलेंस वाले किस्त ट्रेड की संख्या",
        "עברית": "מספר העסקאות התשלומיות עם יתרה"
    },
    "NumBank2NatlTradesWHighUtilization": {
        "English": "Number of Bank/National Trades High Utilization",
        "中文": "银行/全国高利用率交易数量",
        "한국어": "은행/국가 거래 중 높은 이용률 거래 수",
        "हिंदी": "बैंक/राष्ट्रीय ट्रेड जिनकी उच्च उपयोग दर है",
        "עברית": "מספר העסקאות הבנקאיות/לאומיות עם ניצול גבוה"
    },
    "PercentTradesWBalance": {
        "English": "Percent Trades With Balance",
        "中文": "有余额的交易百分比",
        "한국어": "잔액 있는 거래 비율",
        "हिंदी": "बैलेंस वाले ट्रेड का प्रतिशत",
        "עברית": "אחוז העסקאות עם יתרה"
    }
}

# =============================================================================
# 语言选择（增加 עברית）
# =============================================================================
language = st.sidebar.selectbox("Language / 语言 / 언어 / भाषा / Language (עברית)", 
                                options=["English", "中文", "한국어", "हिंदी", "עברית"])

# =============================================================================
# 应用标题和描述
# =============================================================================
st.title(translations["app_title"][language])
st.write(translations["app_description"][language])

# =============================================================================
# 手动输入各项申请人参数（使用 number_input 组件，利用 feature_translations 显示本地化名称）
# =============================================================================
st.sidebar.header(translations["sidebar_details"][language])
def localized_input(key, min_val, max_val, value, step):
    """生成带有本地化变量名称的输入组件"""
    label = f"{translations['enter_value'][language]} {feature_translations[key][language]}"
    return st.sidebar.number_input(label=label, min_value=min_val, max_value=max_val, value=value, step=step)

external_risk = localized_input("ExternalRiskEstimate", 0, 100, 50, 1)
msince_oldest_trade = localized_input("MSinceOldestTradeOpen", 0, 500, 100, 1)
msince_most_recent_trade = localized_input("MSinceMostRecentTradeOpen", 0, 100, 10, 1)
average_m_in_file = localized_input("AverageMInFile", 0, 200, 50, 1)
num_satisfactory_trades = localized_input("NumSatisfactoryTrades", 0, 50, 15, 1)
num_trades_60_ever = localized_input("NumTrades60Ever2DerogPubRec", 0, 20, 5, 1)
num_trades_90_ever = localized_input("NumTrades90Ever2DerogPubRec", 0, 20, 5, 1)
percent_trades_never_delq = localized_input("PercentTradesNeverDelq", 0, 100, 80, 1)
msince_most_recent_delq = localized_input("MSinceMostRecentDelq", 0, 100, 10, 1)
max_delq_12m = localized_input("MaxDelq2PublicRecLast12M", 0, 10, 5, 1)
max_delq_ever = localized_input("MaxDelqEver", 0, 10, 5, 1)
num_total_trades = localized_input("NumTotalTrades", 0, 50, 10, 1)
num_trades_open_12m = localized_input("NumTradesOpeninLast12M", 0, 20, 5, 1)
percent_install_trades = localized_input("PercentInstallTrades", 0, 100, 50, 1)
msince_most_recent_inq = localized_input("MSinceMostRecentInqexcl7days", 0, 100, 10, 1)
num_inq_last_6m = localized_input("NumInqLast6M", 0, 20, 2, 1)
num_inq_last_6m_excl7 = localized_input("NumInqLast6Mexcl7days", 0, 20, 2, 1)
net_fraction_revolving_burden = localized_input("NetFractionRevolvingBurden", 0, 100, 50, 1)
net_fraction_install_burden = localized_input("NetFractionInstallBurden", 0, 100, 50, 1)
num_revolving_trades_balance = localized_input("NumRevolvingTradesWBalance", 0, 20, 5, 1)
num_install_trades_balance = localized_input("NumInstallTradesWBalance", 0, 20, 5, 1)
num_bank_natl_trades_high_util = localized_input("NumBank2NatlTradesWHighUtilization", 0, 20, 5, 1)
percent_trades_balance = localized_input("PercentTradesWBalance", 0, 100, 50, 1)

# =============================================================================
# 定义基于 Logistic Regression 模型贡献计算的拒绝原因反馈函数
# 对每个变量都提供详细原因说明（使用本地化变量名称）
# =============================================================================
def get_logistic_rejection_reasons(input_data, lang):
    # 标准化输入数据
    input_scaled = scaler.transform(input_data)
    # 计算各特征贡献：贡献 = 标准化值 * 模型系数
    contributions = input_scaled[0] * logistic_regression_model.coef_[0]
    # 构建 {特征名: 贡献值} 字典
    feature_contribs = {feature: contrib for feature, contrib in zip(scaler.feature_names_in_, contributions)}
    
    reasons = []
    # 遍历所有变量，生成对应原因说明
    for feature, contrib in feature_contribs.items():
        # 取对应语言的变量名称
        localized_name = feature_translations[feature][lang]
        # 根据贡献值判断说明类型
        if abs(contrib) < 0.05:
            reason = translations["reason_neutral"][lang].format(localized_name, contrib)
        elif contrib < 0:
            reason = translations["reason_negative"][lang].format(localized_name, contrib)
        else:
            reason = translations["reason_positive"][lang].format(localized_name, contrib)
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
