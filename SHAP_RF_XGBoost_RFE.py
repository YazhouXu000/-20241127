import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE

# 读取数据
file_path = 'C:/Users/xyz/Desktop/Data_microalgae_1.xlsx'
data = pd.read_excel(file_path)

# 丢弃目标变量中含有缺失值的行
data = data.dropna(subset=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])

# 选择特征和目标变量
features = data.drop(columns=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])
targets = {
    '24h': data['Amount_24'],
    '48h': data['Amount_48h'],
    '72h': data['Amount_72h'],
    '96h': data['Amount_96h']
}

# 分离数值型和非数值型特征
numeric_features = features.select_dtypes(include=['number']).columns
categorical_features = features.select_dtypes(include=['object']).columns

# 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # 填补数值特征中的缺失值
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # 填补分类特征中的缺失值
            ('onehot', OneHotEncoder())
        ]), categorical_features)
    ])

# RFE特征选择函数
def get_rfe_selected_features(model, X_train, y_train, num_features):
    rfe = RFE(estimator=model, n_features_to_select=num_features)
    rfe.fit(X_train, y_train)
    selected_features = np.array(numeric_features.tolist() + list(
        preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)))[rfe.support_]
    return rfe, selected_features

# 函数：训练模型并进行SHAP分析
def train_model_shap_analysis(model, target_name, target, features, num_features):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # 预处理数据
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 通过RFE选择特征
    rfe, selected_features = get_rfe_selected_features(model, X_train_transformed, y_train, num_features)
    X_train_selected = X_train_transformed[:, rfe.support_]
    X_test_selected = X_test_transformed[:, rfe.support_]

    # 训练模型
    model.fit(X_train_selected, y_train)

    # 使用SHAP解释模型
    explainer = shap.Explainer(model, X_train_selected)
    shap_values = explainer(X_train_selected, check_additivity=False)  # 关闭SHAP加性检查

    # 绘制SHAP图
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train_selected, feature_names=selected_features, show=False)
    plt.title(f'SHAP Summary Plot - {target_name}')
    plt.savefig(f'SHAP_Summary_Plot_{target_name}.png')
    plt.close()

    # 保存重要性数据到表格中
    shap_df = pd.DataFrame(shap_values.values, columns=selected_features)
    shap_df_abs_mean = shap_df.abs().mean().sort_values(ascending=False).reset_index()
    shap_df_abs_mean.columns = ['Feature', 'Mean |SHAP Value|']
    shap_df_abs_mean.to_excel(f'SHAP_Feature_Importance_{target_name}.xlsx', index=False)

    print(f"SHAP分析完成并保存至: SHAP_Feature_Importance_{target_name}.xlsx 和 SHAP_Summary_Plot_{target_name}.png")

# 对每个时间点设置不同的特征数量
num_features_dict = {
    '24h': 15,
    '48h': 10,
    '72h': 15,
    '96h': 15
}

# 分别对24h, 48h, 72h, 96h的目标变量进行模型训练和SHAP分析
for model_name, model in {
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42)
}.items():
    for target_name, target in targets.items():
        num_features = num_features_dict[target_name]
        train_model_shap_analysis(model, f"{model_name}_{target_name}", target, features, num_features)

print("所有目标的SHAP分析完成！")
