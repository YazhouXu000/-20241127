import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.impute import SimpleImputer

# 读取数据
file_path = 'C:/Users/xyz/Desktop/Data_microalgae_1.xlsx'
data = pd.read_excel(file_path)

# 丢弃目标变量中含有缺失值的行
data = data.dropna(subset=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])

# 选择特征和目标变量
features = data.drop(columns=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])
target_24 = data['Amount_24']  # 示例，选择24小时作为目标变量

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
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 独热编码分类特征
        ]), categorical_features)
    ])

# 使用随机森林回归作为基准模型，也可以使用XGBoost等其他模型
model = RandomForestRegressor(random_state=42)

# 创建管道，将预处理和模型结合起来
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target_24, test_size=0.2, random_state=42)

# 对训练集和测试集进行预处理
X_train_transformed = pipeline.named_steps['preprocessor'].fit_transform(X_train)
X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)

# 递归特征消除，选择重要特征
rfe_selector = RFE(estimator=model, n_features_to_select=5, step=1)
rfe_selector.fit(X_train_transformed, y_train)

# 获取特征选择的支持
support = rfe_selector.support_

# 获取所有特征的名字，包括数值型和编码后的分类特征
numeric_feature_names = numeric_features.tolist()
categorical_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features).tolist()
all_feature_names = numeric_feature_names + categorical_feature_names

# 输出选出的重要特征
important_features = [feature for feature, selected in zip(all_feature_names, support) if selected]
print("Selected Important Features:", important_features)
