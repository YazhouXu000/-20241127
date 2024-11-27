# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
# from sklearn.feature_selection import RFE
# import math
#
# # 读取数据
# file_path = 'C:/Users/xyz/Desktop/Data_microalgae_1.xlsx'
# data = pd.read_excel(file_path)
#
# # 丢弃目标变量中含有缺失值的行
# data = data.dropna(subset=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])
#
# # 选择特征和目标变量
# features = data.drop(columns=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])
# targets = {
#     '24h': data['Amount_24'],
#     '48h': data['Amount_48h'],
#     '72h': data['Amount_72h'],
#     '96h': data['Amount_96h']
# }
#
# # 分离数值型和非数值型特征
# numeric_features = features.select_dtypes(include=['number']).columns
# categorical_features = features.select_dtypes(include=['object']).columns
#
# # 创建预处理管道
# preprocessor = ColumnTransformer(
#     transformers=[('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),  # 填补数值特征中的缺失值
#                                           ('scaler', StandardScaler())]), numeric_features),
#                   ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),  # 填补分类特征中的缺失值
#                                           ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
#     ])
#
# # 评估模型性能指标
# def evaluate_model_performance(y_true, y_pred):
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = math.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     return mse, rmse, mae, r2
#
# # 函数：训练加权投票模型（RandomForest + XGBoost），评估模型性能并可视化
# def train_weighted_voting_model_and_evaluate(target_name, target, features, rf_weight=0.5, xgb_weight=0.5, num_features=None):
#     # 拆分训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
#
#     # 预处理数据
#     X_train_transformed = preprocessor.fit_transform(X_train)
#     X_test_transformed = preprocessor.transform(X_test)
#
#     # 特征选择 - 使用RFE进行特征选择
#     if num_features is not None:
#         # 使用RandomForestRegressor作为基学习器进行RFE特征选择
#         rfe_selector = RFE(estimator=RandomForestRegressor(random_state=42), n_features_to_select=num_features)
#         X_train_selected = rfe_selector.fit_transform(X_train_transformed, y_train)
#         X_test_selected = rfe_selector.transform(X_test_transformed)
#     else:
#         X_train_selected = X_train_transformed
#         X_test_selected = X_test_transformed
#
#     # 定义XGBoost和Random Forest模型
#     xgboost_model = xgb.XGBRegressor(random_state=42)
#     rf_model = RandomForestRegressor(random_state=42)
#
#     # 训练XGBoost模型
#     xgboost_model.fit(X_train_selected, y_train)
#
#     # 训练Random Forest模型
#     rf_model.fit(X_train_selected, y_train)
#
#     # 预测
#     y_train_pred_rf = rf_model.predict(X_train_selected)
#     y_test_pred_rf = rf_model.predict(X_test_selected)
#
#     y_train_pred_xgb = xgboost_model.predict(X_train_selected)
#     y_test_pred_xgb = xgboost_model.predict(X_test_selected)
#
#     # 加权投票合并预测结果
#     y_train_pred_weighted = rf_weight * y_train_pred_rf + xgb_weight * y_train_pred_xgb
#     y_test_pred_weighted = rf_weight * y_test_pred_rf + xgb_weight * y_test_pred_xgb
#
#     # 评估训练集和测试集的模型性能
#     train_mse, train_rmse, train_mae, train_r2 = evaluate_model_performance(y_train, y_train_pred_weighted)
#     test_mse, test_rmse, test_mae, test_r2 = evaluate_model_performance(y_test, y_test_pred_weighted)
#
#     # 可视化模型预测与实际值的对比
#     plot_true_vs_predicted(y_train, y_train_pred_weighted, y_test, y_test_pred_weighted, target_name,
#                            train_mse, train_rmse, train_mae, train_r2,
#                            test_mse, test_rmse, test_mae, test_r2)
#
#     return y_test, y_test_pred_weighted
#
# # 可视化：真实值与预测值的对比图，并在图中添加训练集和测试集的性能指标
# def plot_true_vs_predicted(y_train_true, y_train_pred, y_test_true, y_test_pred, target_name,
#                            train_mse, train_rmse, train_mae, train_r2,
#                            test_mse, test_rmse, test_mae, test_r2):
#     plt.figure(figsize=(10, 6))
#
#     # 绘制训练集的真实值和预测值
#     plt.scatter(y_train_true, y_train_pred, color='blue', label='Train Set', alpha=0.6)
#     # 绘制测试集的真实值和预测值
#     plt.scatter(y_test_true, y_test_pred, color='red', label='Test Set', alpha=0.6)
#
#     # 绘制理想情况下真实值等于预测值的对角线
#     max_val = max(y_train_true.max(), y_test_true.max())
#     plt.plot([0, max_val], [0, max_val], '--', color='black', label='Ideal Fit (y=x)')
#
#     # 设置标题和轴标签，并调整字体大小
#     plt.title(f'{target_name} - True vs Predicted Values', fontsize=18)
#     plt.xlabel('True Values', fontsize=14)
#     plt.ylabel('Predicted Values', fontsize=14)
#     plt.legend(fontsize=12)  # 调整图例的字体大小
#
#     # 在图中添加训练集和测试集的模型性能指标
#     textstr_train = f'Train Set:\nMSE: {train_mse:.2f}\nRMSE: {train_rmse:.2f}\nMAE: {train_mae:.2f}\nR²: {train_r2:.2f}'
#     textstr_test = f'Test Set:\nMSE: {test_mse:.2f}\nRMSE: {test_rmse:.2f}\nMAE: {test_mae:.2f}\nR²: {test_r2:.2f}'
#
#     # 将训练集的指标放在左上角，测试集的指标放在右下角
#     plt.gca().text(0.05, 0.75, textstr_train, transform=plt.gca().transAxes, fontsize=12,
#                    verticalalignment='top',
#                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgrey'))
#     plt.gca().text(0.95, 0.05, textstr_test, transform=plt.gca().transAxes, fontsize=12,
#                    verticalalignment='bottom', horizontalalignment='right',
#                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgrey'))
#
#     # 美化图表
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#
#     # 保存并显示图表
#     plt.savefig(f'{target_name}_True_vs_Predicted.png')
#     plt.show()
#
# # 对24h, 48h, 72h, 96h进行加权投票模型训练、评估、可视化
# # 设置每个时间段的权重和特征选择数目
# weights = {
#     '24h': {'rf_weight': 0.25, 'xgb_weight': 0.85, 'num_features': 15},
#     '48h': {'rf_weight': 0.5, 'xgb_weight': 0.5, 'num_features': 10},
#     '72h': {'rf_weight': 0.5, 'xgb_weight': 0.5, 'num_features': 15},
#     '96h': {'rf_weight': 0.5, 'xgb_weight': 0.5, 'num_features': 15},
#     }
#
# # 对每个时间段进行加权投票模型训练、评估、可视化
# for target_name, target in targets.items():
#     print(f"Processing target: {target_name}")
#     rf_weight = weights[target_name]['rf_weight']
#     xgb_weight = weights[target_name]['xgb_weight']
#     num_features = weights[target_name]['num_features']
#
#     train_weighted_voting_model_and_evaluate(target_name, target, features,
#                                              rf_weight=rf_weight, xgb_weight=xgb_weight,
#                                              num_features=num_features)


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
# from sklearn.feature_selection import RFE
# import math
#
# # 读取数据
# file_path = 'C:/Users/xyz/Desktop/Data_microalgae_1.xlsx'
# data = pd.read_excel(file_path)
#
# # 丢弃目标变量中含有缺失值的行
# data = data.dropna(subset=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])
#
# # 选择特征和目标变量
# features = data.drop(columns=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])
# targets = {
#     '24h': data['Amount_24'],
#     '48h': data['Amount_48h'],
#     '72h': data['Amount_72h'],
#     '96h': data['Amount_96h']
# }
#
# # 分离数值型和非数值型特征
# numeric_features = features.select_dtypes(include=['number']).columns
# categorical_features = features.select_dtypes(include=['object']).columns
#
# # 创建预处理管道
# preprocessor = ColumnTransformer(
#     transformers=[('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),  # 填补数值特征中的缺失值
#                                           ('scaler', StandardScaler())]), numeric_features),
#                   ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),  # 填补分类特征中的缺失值
#                                           ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
#                   ])
#
# # 评估模型性能指标
# def evaluate_model_performance(y_true, y_pred):
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = math.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     return mse, rmse, mae, r2
#
# # 函数：训练加权投票模型（RandomForest + XGBoost），评估模型性能并可视化
# def train_weighted_voting_model(target_name, target, features, rf_weight=0.5, xgb_weight=0.5, num_features=None):
#     # 拆分训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
#
#     # 预处理数据
#     X_train_transformed = preprocessor.fit_transform(X_train)
#     X_test_transformed = preprocessor.transform(X_test)
#
#     # 特征选择 - 使用RFE进行特征选择
#     if num_features is not None:
#         # 使用RandomForestRegressor作为基学习器进行RFE特征选择
#         rfe_selector = RFE(estimator=RandomForestRegressor(random_state=42), n_features_to_select=num_features)
#         X_train_selected = rfe_selector.fit_transform(X_train_transformed, y_train)
#         X_test_selected = rfe_selector.transform(X_test_transformed)
#     else:
#         X_train_selected = X_train_transformed
#         X_test_selected = X_test_transformed
#
#     # 定义XGBoost和Random Forest模型
#     xgboost_model = xgb.XGBRegressor(random_state=42)
#     rf_model = RandomForestRegressor(random_state=42)
#
#     # 训练XGBoost模型
#     xgboost_model.fit(X_train_selected, y_train)
#
#     # 训练Random Forest模型
#     rf_model.fit(X_train_selected, y_train)
#
#     # 预测
#     y_train_pred_rf = rf_model.predict(X_train_selected)
#     y_test_pred_rf = rf_model.predict(X_test_selected)
#
#     y_train_pred_xgb = xgboost_model.predict(X_train_selected)
#     y_test_pred_xgb = xgboost_model.predict(X_test_selected)
#
#     # 加权投票合并预测结果
#     y_train_pred_weighted = rf_weight * y_train_pred_rf + xgb_weight * y_train_pred_xgb
#     y_test_pred_weighted = rf_weight * y_test_pred_rf + xgb_weight * y_test_pred_xgb
#
#     # 评估训练集和测试集的模型性能
#     train_mse, train_rmse, train_mae, train_r2 = evaluate_model_performance(y_train, y_train_pred_weighted)
#     test_mse, test_rmse, test_mae, test_r2 = evaluate_model_performance(y_test, y_test_pred_weighted)
#
#     # 返回预测值和模型性能
#     return y_test, y_test_pred_weighted, test_r2
#
# # 寻找最优的加权组合
# def find_optimal_weights_for_target(target_name, target, features, num_features=15):
#     best_r2 = -np.inf
#     best_weights = None
#     best_y_test_pred_weighted = None
#
#     # 在[0, 1]之间步长为0.1遍历权重
#     for rf_weight in np.arange(0, 1.1, 0.05):
#         xgb_weight = 1 - rf_weight
#         print(f"Finding optimal weights for target: {target_name} | RF Weight: {rf_weight:.2f} | XGB Weight: {xgb_weight:.2f}")
#
#         # 训练加权投票模型并获取性能
#         _, y_test_pred_weighted, r2 = train_weighted_voting_model(target_name, target, features, rf_weight=rf_weight, xgb_weight=xgb_weight, num_features=num_features)
#
#         # 判断当前组合的R2是否优于当前最优结果
#         if r2 > best_r2:
#             best_r2 = r2
#             best_weights = (rf_weight, xgb_weight)
#             best_y_test_pred_weighted = y_test_pred_weighted
#
#     print(f"Best weights for {target_name}: RF Weight = {best_weights[0]:.2f}, XGB Weight = {best_weights[1]:.2f}, R2 = {best_r2:.2f}")
#     return best_weights
#
# # 主函数：遍历每个目标（24h, 48h, 72h, 96h）并寻找最优权重
# for target_name, target in targets.items():
#     print(f"Processing target: {target_name}")
#     best_weights = find_optimal_weights_for_target(target_name, target, features, num_features=15)
#     print(f"Optimal weights for {target_name}: RF Weight = {best_weights[0]:.2f}, XGB Weight = {best_weights[1]:.2f}")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.feature_selection import RFE
import math

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
    transformers=[('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),  # 填补数值特征中的缺失值
                                          ('scaler', StandardScaler())]), numeric_features),
                  ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),  # 填补分类特征中的缺失值
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
                  ])

# 评估模型性能指标
def evaluate_model_performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

# 函数：训练Blending模型（RandomForest + XGBoost），评估模型性能
def train_blending_model_and_evaluate(target_name, target, features, rf_weight=0.5, xgb_weight=0.5, num_features=None):
    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # 预处理数据
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 特征选择 - 使用RFE进行特征选择
    if num_features is not None:
        # 使用RandomForestRegressor作为基学习器进行RFE特征选择
        rfe_selector = RFE(estimator=RandomForestRegressor(random_state=42), n_features_to_select=num_features)
        X_train_selected = rfe_selector.fit_transform(X_train_transformed, y_train)
        X_test_selected = rfe_selector.transform(X_test_transformed)
    else:
        X_train_selected = X_train_transformed
        X_test_selected = X_test_transformed

    # 定义XGBoost和Random Forest模型
    xgboost_model = xgb.XGBRegressor(random_state=42)
    rf_model = RandomForestRegressor(random_state=42)

    # 训练XGBoost模型
    xgboost_model.fit(X_train_selected, y_train)

    # 训练Random Forest模型
    rf_model.fit(X_train_selected, y_train)

    # 预测
    y_train_pred_rf = rf_model.predict(X_train_selected)
    y_test_pred_rf = rf_model.predict(X_test_selected)

    y_train_pred_xgb = xgboost_model.predict(X_train_selected)
    y_test_pred_xgb = xgboost_model.predict(X_test_selected)

    # Blending：加权平均合并预测结果
    y_train_pred_blended = rf_weight * y_train_pred_rf + xgb_weight * y_train_pred_xgb
    y_test_pred_blended = rf_weight * y_test_pred_rf + xgb_weight * y_test_pred_xgb

    # 评估训练集和测试集的模型性能
    train_mse, train_rmse, train_mae, train_r2 = evaluate_model_performance(y_train, y_train_pred_blended)
    test_mse, test_rmse, test_mae, test_r2 = evaluate_model_performance(y_test, y_test_pred_blended)

    # 可视化模型预测与实际值的对比
    plot_true_vs_predicted(y_train, y_train_pred_blended, y_test, y_test_pred_blended, target_name,
                            train_mse, train_rmse, train_mae, train_r2,
                            test_mse, test_rmse, test_mae, test_r2)

    return y_test, y_test_pred_blended

# 可视化：真实值与预测值的对比图，并在图中添加训练集和测试集的性能指标
def plot_true_vs_predicted(y_train_true, y_train_pred, y_test_true, y_test_pred, target_name,
                            train_mse, train_rmse, train_mae, train_r2,
                            test_mse, test_rmse, test_mae, test_r2):
    plt.figure(figsize=(10, 6))

    # 绘制训练集的真实值和预测值
    plt.scatter(y_train_true, y_train_pred, color='blue', label='Train Set', alpha=0.6)
    # 绘制测试集的真实值和预测值
    plt.scatter(y_test_true, y_test_pred, color='red', label='Test Set', alpha=0.6)

    # 绘制理想情况下真实值等于预测值的对角线
    max_val = max(y_train_true.max(), y_test_true.max())
    plt.plot([0, max_val], [0, max_val], '--', color='black', label='Ideal Fit (y=x)')

    # 设置标题和轴标签，并调整字体大小
    plt.title(f'{target_name} - True vs Predicted Values', fontsize=18)
    plt.xlabel('True Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.legend(fontsize=12)  # 调整图例的字体大小

    # 在图中添加训练集和测试集的模型性能指标
    textstr_train = f'Train Set:\nMSE: {train_mse:.2e}\nRMSE: {train_rmse:.2e}\nMAE: {train_mae:.2e}\nR²: {train_r2:.2f}'
    textstr_test = f'Test Set:\nMSE: {test_mse:.2e}\nRMSE: {test_rmse:.2e}\nMAE: {test_mae:.2e}\nR²: {test_r2:.2f}'

    # 将训练集的指标放在左上角，测试集的指标放在右下角
    plt.gca().text(0.05, 0.75, textstr_train, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgrey'))
    plt.gca().text(0.95, 0.05, textstr_test, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgrey'))

    # 美化图表
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 保存并显示图表
    plt.savefig(f'{target_name}_True_vs_Predicted.png')
    plt.show()

 # 设置每个时间段的权重和特征选择数目
weights = {
    '24h': {'rf_weight': 0.5, 'xgb_weight': 0.5, 'num_features': 40},
    '48h': {'rf_weight': 0.5, 'xgb_weight': 0.5, 'num_features': 40},
    '72h': {'rf_weight': 0.5, 'xgb_weight': 0.5, 'num_features': 40},
    '96h': {'rf_weight': 0.5, 'xgb_weight': 0.5, 'num_features': 40},
    }

# 对每个时间段进行Blending模型训练、评估、可视化
for target_name, target in targets.items():
    print(f"Processing target: {target_name}")
    rf_weight = weights[target_name]['rf_weight']
    xgb_weight = weights[target_name]['xgb_weight']
    num_features = weights[target_name]['num_features']

    train_blending_model_and_evaluate(target_name, target, features,
                                      rf_weight=rf_weight, xgb_weight=xgb_weight,
                                      num_features=num_features)