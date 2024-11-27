# # # import pandas as pd
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # from matplotlib.ticker import ScalarFormatter
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # # from sklearn.compose import ColumnTransformer
# # # from sklearn.pipeline import Pipeline
# # # from sklearn.ensemble import RandomForestRegressor
# # # from sklearn.svm import SVR
# # # from sklearn.neighbors import KNeighborsRegressor
# # # from sklearn.neural_network import MLPRegressor
# # # from sklearn.tree import DecisionTreeRegressor
# # # import xgboost as xgb
# # # from sklearn.impute import SimpleImputer
# # # from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# # # from sklearn.feature_selection import RFE, SelectFromModel
# # #
# # # # 读取数据
# # # file_path = 'C:/Users/xyz/Desktop/Data_microalgae_1.xlsx'
# # # data = pd.read_excel(file_path)
# # #
# # # # 丢弃目标变量中含有缺失值的行
# # # data = data.dropna(subset=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])
# # #
# # # # 选择特征和目标变量
# # # features = data.drop(columns=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])
# # # target_24 = data['Amount_24']
# # # target_48 = data['Amount_48h']
# # # target_72 = data['Amount_72h']
# # # target_96 = data['Amount_96h']
# # #
# # # # 分离数值型和非数值型特征
# # # numeric_features = features.select_dtypes(include=['number']).columns
# # # categorical_features = features.select_dtypes(include=['object']).columns
# # #
# # # # 创建预处理管道
# # # preprocessor = ColumnTransformer(
# # #     transformers=[
# # #         ('num', Pipeline(steps=[
# # #             ('imputer', SimpleImputer(strategy='mean')),  # 填补数值特征中的缺失值
# # #             ('scaler', StandardScaler())
# # #         ]), numeric_features),
# # #         ('cat', Pipeline(steps=[
# # #             ('imputer', SimpleImputer(strategy='most_frequent')),  # 填补分类特征中的缺失值
# # #             ('onehot', OneHotEncoder())
# # #         ]), categorical_features)
# # #     ])
# # #
# # # # 定义使用RFE和SFM的模型列表（增加特征数量选择范围）
# # # n_features_to_select_options = [5, 10, 15, 20, 25, 30, 35, 40, 46]  # 不同的RFE特征选择数量
# # #
# # # models_with_selection = {
# # #     f'Random Forest with RFE (n_features_to_select={n})': Pipeline(steps=[
# # #         ('preprocessor', preprocessor),
# # #         ('feature_selection', RFE(estimator=RandomForestRegressor(random_state=42), n_features_to_select=n)),
# # #         # RFE选择n个特征
# # #         ('regressor', RandomForestRegressor(random_state=42))
# # #     ]) for n in n_features_to_select_options
# # # }
# # #
# # # # 函数：训练模型并保存性能指标
# # # def train_and_evaluate_model(pipeline, features, target):
# # #     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# # #
# # #     pipeline.fit(X_train, y_train)
# # #
# # #     y_train_pred = pipeline.predict(X_train)
# # #     y_test_pred = pipeline.predict(X_test)
# # #
# # #     # 计算MSE、R²、RMSE、MAE
# # #     train_mse = mean_squared_error(y_train, y_train_pred)
# # #     train_r2 = r2_score(y_train, y_train_pred)
# # #     train_rmse = np.sqrt(train_mse)
# # #     train_mae = mean_absolute_error(y_train, y_train_pred)
# # #
# # #     test_mse = mean_squared_error(y_test, y_test_pred)
# # #     test_r2 = r2_score(y_test, y_test_pred)
# # #     test_rmse = np.sqrt(test_mse)
# # #     test_mae = mean_absolute_error(y_test, y_test_pred)
# # #
# # #     return {
# # #         'Train MSE': train_mse,
# # #         'Train R²': train_r2,
# # #         'Train RMSE': train_rmse,
# # #         'Train MAE': train_mae,
# # #         'Test MSE': test_mse,
# # #         'Test R²': test_r2,
# # #         'Test RMSE': test_rmse,
# # #         'Test MAE': test_mae
# # #     }
# # #
# # # # 目标列表
# # # targets = {
# # #     '24 Hour Amount': target_24,
# # #     '48 Hour Amount': target_48,
# # #     '72 Hour Amount': target_72,
# # #     '96 Hour Amount': target_96
# # # }
# # #
# # # # 存储所有结果的字典
# # # all_results = {}
# # #
# # # # 训练所有模型，记录每种特征选择下的结果
# # # for model_name, pipeline in models_with_selection.items():
# # #     model_results = {}
# # #     for target_name, target in targets.items():
# # #         results = train_and_evaluate_model(pipeline, features, target)
# # #         model_results[target_name] = results
# # #     all_results[model_name] = model_results
# # #
# # # # 输出所有模型的结果
# # # for model_name, model_results in all_results.items():
# # #     print(f"\nResults for {model_name}:")
# # #     for target_name, metrics in model_results.items():
# # #         print(f"  Target: {target_name}")
# # #         for metric_name, value in metrics.items():
# # #             print(f"    {metric_name}: {value:.2e}")
# # #
# #
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # from sklearn.compose import ColumnTransformer
# # from sklearn.pipeline import Pipeline
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.svm import SVR
# # from sklearn.neighbors import KNeighborsRegressor
# # from sklearn.neural_network import MLPRegressor
# # from sklearn.tree import DecisionTreeRegressor
# # import xgboost as xgb
# # from sklearn.impute import SimpleImputer
# # from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# # from openpyxl import Workbook
# #
# # # 读取数据
# # file_path = 'C:/Users/xyz/Desktop/Data_microalgae_1.xlsx'
# # data = pd.read_excel(file_path)
# #
# # # 丢弃目标变量中含有缺失值的行
# # data = data.dropna(subset=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])
# #
# # # 选择特征和目标变量
# # features = data.drop(columns=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])
# # target_24 = data['Amount_24']
# # target_48 = data['Amount_48h']
# # target_72 = data['Amount_72h']
# # target_96 = data['Amount_96h']
# #
# # # 分离数值型和非数值型特征
# # numeric_features = features.select_dtypes(include=['number']).columns
# # categorical_features = features.select_dtypes(include=['object']).columns
# #
# # # 创建预处理管道
# # preprocessor = ColumnTransformer(
# #     transformers=[
# #         ('num', Pipeline(steps=[
# #             ('imputer', SimpleImputer(strategy='mean')),  # 填补数值特征中的缺失值
# #             ('scaler', StandardScaler())
# #         ]), numeric_features),
# #         ('cat', Pipeline(steps=[
# #             ('imputer', SimpleImputer(strategy='most_frequent')),  # 填补分类特征中的缺失值
# #             ('onehot', OneHotEncoder())
# #         ]), categorical_features)
# #     ])
# #
# # # 定义模型列表
# # models = {
# #     'Random Forest': RandomForestRegressor(random_state=42),
# #     'SVR': SVR(),
# #     'KNeighbors': KNeighborsRegressor(),
# #     'MLP': MLPRegressor(random_state=42, max_iter=1000),
# #     'Decision Tree': DecisionTreeRegressor(random_state=42),
# #     'XGBoost': xgb.XGBRegressor(random_state=42)
# # }
# #
# #
# # # 函数：训练模型并返回性能指标和预测结果
# # def train_and_evaluate_model(pipeline, features, target):
# #     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# #
# #     pipeline.fit(X_train, y_train)
# #
# #     y_train_pred = pipeline.predict(X_train)
# #     y_test_pred = pipeline.predict(X_test)
# #
# #     # 计算MSE、R²、RMSE、MAE
# #     train_mse = mean_squared_error(y_train, y_train_pred)
# #     train_r2 = r2_score(y_train, y_train_pred)
# #     train_rmse = np.sqrt(train_mse)
# #     train_mae = mean_absolute_error(y_train, y_train_pred)
# #
# #     test_mse = mean_squared_error(y_test, y_test_pred)
# #     test_r2 = r2_score(y_test, y_test_pred)
# #     test_rmse = np.sqrt(test_mse)
# #     test_mae = mean_absolute_error(y_test, y_test_pred)
# #
# #     # 返回指标和预测结果
# #     return {
# #         'metrics': {
# #             'Train MSE': train_mse,
# #             'Train R²': train_r2,
# #             'Train RMSE': train_rmse,
# #             'Train MAE': train_mae,
# #             'Test MSE': test_mse,
# #             'Test R²': test_r2,
# #             'Test RMSE': test_rmse,
# #             'Test MAE': test_mae
# #         },
# #         'predictions': {
# #             'train': (y_train, y_train_pred),  # 返回训练集的真实值和预测值
# #             'test': (y_test, y_test_pred)  # 返回测试集的真实值和预测值
# #         }
# #     }
# #
# #
# # # 目标列表
# # targets = {
# #     '24 Hour Amount': target_24,
# #     '48 Hour Amount': target_48,
# #     '72 Hour Amount': target_72,
# #     '96 Hour Amount': target_96
# # }
# #
# # # 创建Excel工作簿
# # wb = Workbook()
# #
# # # 写入每个模型的性能指标到一个工作表
# # ws_metrics = wb.active
# # ws_metrics.title = "Model Metrics"
# #
# # # 写入表头
# # ws_metrics.append(['Model', 'Target', 'Train MSE', 'Train R²', 'Train RMSE', 'Train MAE',
# #                    'Test MSE', 'Test R²', 'Test RMSE', 'Test MAE'])
# #
# # # 存储所有结果的字典
# # all_results = {}
# #
# # # 训练所有模型，记录结果和预测
# # for model_name, model in models.items():
# #     pipeline = Pipeline(steps=[
# #         ('preprocessor', preprocessor),
# #         ('regressor', model)
# #     ])
# #
# #     model_results = {}
# #     for target_name, target in targets.items():
# #         results = train_and_evaluate_model(pipeline, features, target)
# #         model_results[target_name] = results
# #
# #         # 写入性能指标到Excel
# #         ws_metrics.append([model_name, target_name] + [results['metrics'][metric] for metric in results['metrics']])
# #
# #         # 创建新工作表来存储预测结果
# #         ws_predictions = wb.create_sheet(title=f"{model_name}_{target_name}_Predictions")
# #         ws_predictions.append(['Data Set', 'True Value', 'Predicted Value'])
# #
# #         # 训练集结果
# #         y_train_true, y_train_pred = results['predictions']['train']
# #         for true_val, pred_val in zip(y_train_true, y_train_pred):
# #             ws_predictions.append(['Train', true_val, pred_val])
# #
# #         # 测试集结果
# #         y_test_true, y_test_pred = results['predictions']['test']
# #         for true_val, pred_val in zip(y_test_true, y_test_pred):
# #             ws_predictions.append(['Test', true_val, pred_val])
# #
# #     all_results[model_name] = model_results
# #
# # # 保存到Excel文件
# # wb.save("model_metrics_and_predictions.xlsx")
# #
# # print("模型的性能指标和预测结果已成功保存到 'model_metrics_and_predictions.xlsx' 文件中。")
#
#
# import shap
# import xgboost as xgb
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# import matplotlib.pyplot as plt
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
#     transformers=[
#         ('num', Pipeline(steps=[
#             ('imputer', SimpleImputer(strategy='mean')),  # 填补数值特征中的缺失值
#             ('scaler', StandardScaler())
#         ]), numeric_features),
#         ('cat', Pipeline(steps=[
#             ('imputer', SimpleImputer(strategy='most_frequent')),  # 填补分类特征中的缺失值
#             ('onehot', OneHotEncoder())
#         ]), categorical_features)
#     ])
#
#
# # 函数：训练XGBoost模型并进行SHAP分析
# def train_xgboost_shap(target_name, target, features):
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
#
#     # 预处理数据
#     X_train_transformed = preprocessor.fit_transform(X_train)
#     X_test_transformed = preprocessor.transform(X_test)
#
#     # 训练XGBoost模型
#     xgboost_model = xgb.XGBRegressor(random_state=42)
#     xgboost_model.fit(X_train_transformed, y_train)
#
#     # 使用SHAP解释模型
#     explainer = shap.Explainer(xgboost_model, X_train_transformed)
#     shap_values = explainer(X_train_transformed)
#
#     # 绘制特征重要性排序图
#     plt.figure(figsize=(10, 6))
#     shap.summary_plot(shap_values, X_train_transformed, feature_names=numeric_features.tolist() + list(
#         preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)), show=False)
#     plt.title(f'SHAP Summary Plot - {target_name}')
#     plt.savefig(f'SHAP_Summary_Plot_{target_name}.png')
#     plt.close()
#
#     # 保存重要性数据到表格中
#     shap_df = pd.DataFrame(shap_values.values, columns=numeric_features.tolist() + list(
#         preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)))
#     shap_df_abs_mean = shap_df.abs().mean().sort_values(ascending=False).reset_index()
#     shap_df_abs_mean.columns = ['Feature', 'Mean |SHAP Value|']
#     shap_df_abs_mean.to_excel(f'SHAP_Feature_Importance_{target_name}.xlsx', index=False)
#
#     print(f"SHAP分析完成并保存至: SHAP_Feature_Importance_{target_name}.xlsx 和 SHAP_Summary_Plot_{target_name}.png")
#
#
# # 分别对24h, 48h, 72h, 96h的目标变量进行XGBoost模型训练和SHAP分析
# for target_name, target in targets.items():
#     train_xgboost_shap(target_name, target, features)
#
# print("所有目标的SHAP分析完成！")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import xgboost as xgb

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


# 函数：训练XGBoost模型并进行Permutation Importance分析
def train_xgboost_permutation_importance(target_name, target, features):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # 预处理数据
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 训练XGBoost模型
    xgboost_model = xgb.XGBRegressor(random_state=42)
    xgboost_model.fit(X_train_transformed, y_train)

    # 预测并计算基线性能
    baseline_mse = mean_squared_error(y_test, xgboost_model.predict(X_test_transformed))

    # 进行特征置换重要性分析
    perm_importance = permutation_importance(xgboost_model, X_test_transformed, y_test, n_repeats=10, random_state=42,
                                             scoring='neg_mean_squared_error')

    # 获取特征名
    feature_names = numeric_features.tolist() + list(
        preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))

    # 保存结果到表格
    perm_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance Mean': perm_importance.importances_mean,
        'Importance Std': perm_importance.importances_std
    }).sort_values(by='Importance Mean', ascending=False)
    perm_importance_df.to_excel(f'Permutation_Feature_Importance_{target_name}.xlsx', index=False)

    # 可视化重要性排序
    plt.figure(figsize=(10, 6))
    perm_importance_df.sort_values(by='Importance Mean', ascending=True).plot.barh(x='Feature', y='Importance Mean',
                                                                                   xerr='Importance Std', legend=False)
    plt.title(f'Permutation Importance - {target_name}')
    plt.xlabel('Importance Mean Decrease in MSE')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(f'Permutation_Importance_Plot_{target_name}.png')
    plt.close()

    print(
        f"Permutation Importance 分析完成并保存至: Permutation_Feature_Importance_{target_name}.xlsx 和 Permutation_Importance_Plot_{target_name}.png")


# 分别对24h, 48h, 72h, 96h的目标变量进行XGBoost模型训练和Permutation Importance分析
for target_name, target in targets.items():
    train_xgboost_permutation_importance(target_name, target, features)

print("所有目标的Permutation Importance分析完成！")
