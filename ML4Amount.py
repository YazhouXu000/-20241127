
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
file_path = 'C:/Users/xyz/Desktop/Data_microalgae_1.xlsx'
data = pd.read_excel(file_path)

# 丢弃目标变量中含有缺失值的行
data = data.dropna(subset=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])

# 选择特征和目标变量
features = data.drop(columns=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])
target_24 = data['Amount_24']
target_48 = data['Amount_48h']
target_72 = data['Amount_72h']
target_96 = data['Amount_96h']

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

# 定义模型列表
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'ANN': MLPRegressor(random_state=42, max_iter=2000, learning_rate_init=0.001, hidden_layer_sizes=(100, 50)),  # 增加最大迭代次数和调整其他参数
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42)
}

# 函数：训练模型并绘制结果
def train_and_plot(model, model_name, features, target, target_name, ax):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    ax.scatter(y_train, y_train_pred, label='Train', alpha=0.7, s=100, edgecolor='k', linewidths=1.5)
    ax.scatter(y_test, y_test_pred, label='Test', alpha=0.7, s=100, edgecolor='k', linewidths=1.5)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
    ax.set_xlabel('Actual', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted', fontsize=14, fontweight='bold')
    ax.set_title(f'{model_name} - {target_name}', fontsize=16, fontweight='bold')

    # 加粗图例中的标签
    ax.legend(fontsize=14, markerscale=1.5, frameon=True, prop={'weight': 'bold'})

    # 设置刻度字体大小和粗细
    ax.tick_params(axis='both', which='major', labelsize=12, width=2)
    plt.setp(ax.get_xticklabels(), fontsize=12, fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight='bold')

    # 设置科学计数法格式化
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    ax.text(0.05, 0.85, f'Train MSE: {train_mse:.2e}\\nTrain R&sup2;: {train_r2:.2f}', ha='left', va='top',
            transform=ax.transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))
    ax.text(0.95, 0.15, f'Test MSE: {test_mse:.2e}\\nTest R&sup2;: {test_r2:.2f}', ha='right', va='bottom',
            transform=ax.transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

# 目标列表
targets = {
    '24 Hour Amount': target_24,
    '48 Hour Amount': target_48,
    '72 Hour Amount': target_72,
    '96 Hour Amount': target_96
}

# 创建图像
for model_name, model in models.items():
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    for (target_name, target), ax in zip(targets.items(), axs.ravel()):
        train_and_plot(model, model_name, features, target, target_name, ax)

    plt.tight_layout()
    plt.savefig(f'{model_name}_Actual_vs_Predicted_All.png')

plt.show()