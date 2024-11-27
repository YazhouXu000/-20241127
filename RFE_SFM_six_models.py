import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns

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
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder())
        ]), categorical_features)
    ])

# 定义模型列表
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'SVR': SVR(),
    'KNeighbors': KNeighborsRegressor(),
    'MLP': MLPRegressor(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42)
}


# 函数：训练模型并返回性能指标和预测结果
def train_and_evaluate_model(pipeline, features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # 计算性能指标
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # 返回指标和预测结果
    return {
        'metrics': {
            'Train MSE': train_mse,
            'Train R²': train_r2,
            'Train RMSE': train_rmse,
            'Train MAE': train_mae,
            'Test MSE': test_mse,
            'Test R²': test_r2,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae
        },
        'predictions': {
            'train': (y_train, y_train_pred),
            'test': (y_test, y_test_pred)
        }
    }


# 目标列表
targets = {
    '24 Hour Amount': target_24,
    '48 Hour Amount': target_48,
    '72 Hour Amount': target_72,
    '96 Hour Amount': target_96
}

# 存储所有结果的字典
all_results = {}

# 训练所有模型，记录结果和预测
for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    model_results = {}
    for target_name, target in targets.items():
        results = train_and_evaluate_model(pipeline, features, target)
        model_results[target_name] = results
        all_results[f'{model_name}_{target_name}'] = results


# 可视化模型性能的函数
def plot_predictions_with_metrics(results):
    for model_target, result in results.items():
        train_true, train_pred = result['predictions']['train']
        test_true, test_pred = result['predictions']['test']

        # 获取性能指标
        train_r2 = result['metrics']['Train R²']
        train_mse = result['metrics']['Train MSE']
        train_rmse = result['metrics']['Train RMSE']
        train_mae = result['metrics']['Train MAE']

        test_r2 = result['metrics']['Test R²']
        test_mse = result['metrics']['Test MSE']
        test_rmse = result['metrics']['Test RMSE']
        test_mae = result['metrics']['Test MAE']

        # 绘制图像，显示训练集和测试集的预测值与真实值
        plt.figure(figsize=(8, 8))

        # 训练集
        plt.scatter(train_true, train_pred, color='blue', alpha=0.5, label='Train Predictions')
        # 测试集
        plt.scatter(test_true, test_pred, color='red', alpha=0.5, label='Test Predictions')

        # 对角线：真实值 = 预测值
        plt.plot([min(train_true), max(train_true)], [min(train_true), max(train_true)], color='black', linestyle='--')

        # 添加标题和标签
        plt.title(f'Predictions: {model_target}', fontsize=20,  fontweight='bold' )  # 标题字体大小粗细
        plt.xlabel('True Value', fontsize=18, fontweight='bold')  # x轴标签字体大小
        plt.ylabel('Predicted Value', fontsize=18, fontweight='bold')  # y轴标签字体大小
        plt.legend()

        # 设置图例的字体大小
        plt.legend(fontsize=16)  # 修改图例的字体大小

        # 设置x轴和y轴的刻度字体大小
        plt.xticks(fontsize=16)  # 设置x轴刻度标签字体大小为16
        plt.yticks(fontsize=16)  # 设置y轴刻度标签字体大小为16

        # 在图上标注性能指标（训练集和测试集的性能）
        textstr = '\n'.join((
            f'Train R²: {train_r2:.2f}',
            f'Train MSE: {train_mse:.2e}',
            f'Train RMSE: {train_rmse:.2f}',
            f'Train MAE: {train_mae:.2f}',
            '',
            f'Test R²: {test_r2:.2f}',
            f'Test MSE: {test_mse:.2e}',
            f'Test RMSE: {test_rmse:.2f}',
            f'Test MAE: {test_mae:.2f}'
        ))

        # 将性能指标显示在图的左上角，避免和图表内容重叠
        plt.gcf().text(0.12, 0.8, textstr, fontsize=18, verticalalignment='top',
                       bbox=dict(facecolor='white', alpha=0.2))

        # 显示图像
        plt.tight_layout()
        plt.show()


# 调用可视化函数
plot_predictions_with_metrics(all_results)
