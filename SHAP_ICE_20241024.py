# # # # import pandas as pd
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt
# # # # import shap
# # # # import xgboost as xgb
# # # # from sklearn.ensemble import RandomForestRegressor
# # # # from sklearn.model_selection import train_test_split
# # # # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # # # from sklearn.compose import ColumnTransformer
# # # # from sklearn.pipeline import Pipeline
# # # # from sklearn.impute import SimpleImputer
# # # # from sklearn.feature_selection import RFE
# # # # from sklearn.inspection import PartialDependenceDisplay
# # # #
# # # # # 读取数据
# # # # file_path = 'C:/Users/xyz/Desktop/Data_microalgae_1.xlsx'
# # # # data = pd.read_excel(file_path)
# # # #
# # # # # 丢弃目标变量中含有缺失值的行
# # # # data = data.dropna(subset=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])
# # # #
# # # # # 选择特征和目标变量
# # # # features = data.drop(columns=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])
# # # # targets = {
# # # #     '24h': data['Amount_24'],
# # # #     '48h': data['Amount_48h'],
# # # #     '72h': data['Amount_72h'],
# # # #     '96h': data['Amount_96h']
# # # # }
# # # #
# # # # # 分离数值型和非数值型特征
# # # # numeric_features = features.select_dtypes(include=['number']).columns
# # # # categorical_features = features.select_dtypes(include=['object']).columns
# # # #
# # # # # 创建预处理管道
# # # # preprocessor = ColumnTransformer(
# # # #     transformers=[
# # # #         ('num', Pipeline(steps=[
# # # #             ('imputer', SimpleImputer(strategy='mean')),  # 填补数值特征中的缺失值
# # # #             ('scaler', StandardScaler())
# # # #         ]), numeric_features),
# # # #         ('cat', Pipeline(steps=[
# # # #             ('imputer', SimpleImputer(strategy='most_frequent')),  # 填补分类特征中的缺失值
# # # #             ('onehot', OneHotEncoder())
# # # #         ]), categorical_features)
# # # #     ])
# # # #
# # # # # RFE特征选择函数
# # # # def get_rfe_selected_features(model, X_train, y_train, num_features):
# # # #     rfe = RFE(estimator=model, n_features_to_select=num_features)
# # # #     rfe.fit(X_train, y_train)
# # # #     selected_features = np.array(numeric_features.tolist() + list(
# # # #         preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)))[rfe.support_]
# # # #     return rfe, selected_features
# # # #
# # # # # 函数：训练模型并进行ICE和SHAP分析
# # # # def train_model_ice_analysis(model, target_name, target, features, num_features):
# # # #     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# # # #
# # # #     # 预处理数据
# # # #     X_train_transformed = preprocessor.fit_transform(X_train)
# # # #     X_test_transformed = preprocessor.transform(X_test)
# # # #
# # # #     # 通过RFE选择特征
# # # #     rfe, selected_features = get_rfe_selected_features(model, X_train_transformed, y_train, num_features)
# # # #     X_train_selected = X_train_transformed[:, rfe.support_]
# # # #     X_test_selected = X_test_transformed[:, rfe.support_]
# # # #
# # # #     # 训练模型
# # # #     model.fit(X_train_selected, y_train)
# # # #
# # # #     # ICE图分析 - 每个特征生成一个ICE图并导出
# # # #     for i, feature in enumerate(selected_features):
# # # #         fig, ax = plt.subplots(figsize=(10, 6))
# # # #         pdp = PartialDependenceDisplay.from_estimator(model, X_train_selected, [i], ax=ax)  # 针对每个特征i绘制ICE图
# # # #         plt.title(f'ICE Plot - {target_name} Feature {feature}')
# # # #         plt.savefig(f'ICE_Plot_{target_name}_feature_{feature}.png')
# # # #         plt.close()
# # # #
# # # #         # 将ICE数值保存到Excel中
# # # #         for pd_line in pdp.lines_:
# # # #             grid_values = pd_line[0].get_xdata()  # 获取网格值
# # # #             ice_values = pd_line[0].get_ydata()  # 获取ICE值
# # # #
# # # #             # 将每个特征的 ICE 数值保存到Excel
# # # #             ice_df = pd.DataFrame({
# # # #                 'Grid Values': grid_values,
# # # #                 'ICE Values': ice_values
# # # #             })
# # # #             ice_filename = f"ICE_Values_{target_name}_feature_{feature}.xlsx"  # 保存每个特征的ICE
# # # #             ice_df.to_excel(ice_filename, index=False)
# # # #             print(f"ICE数值已保存至: {ice_filename}")
# # # #
# # # # # 对每个时间点设置不同的特征数量
# # # # num_features_dict = {
# # # #     '24h': 15,
# # # #     '48h': 10,
# # # #     '72h': 15,
# # # #     '96h': 15
# # # # }
# # # #
# # # # # 分别对24h, 48h, 72h, 96h的目标变量进行模型训练和ICE分析
# # # # for model_name, model in {
# # # #     'Random Forest': RandomForestRegressor(random_state=42),
# # # #     'XGBoost': xgb.XGBRegressor(random_state=42)
# # # # }.items():
# # # #     for target_name, target in targets.items():
# # # #         num_features = num_features_dict[target_name]
# # # #         train_model_ice_analysis(model, f"{model_name}_{target_name}", target, features, num_features)
# # # #
# # # # print("所有目标的ICE分析完成！")
# # #
# # #
# # # import pandas as pd
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import xgboost as xgb
# # # from sklearn.ensemble import RandomForestRegressor
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # # from sklearn.compose import ColumnTransformer
# # # from sklearn.pipeline import Pipeline
# # # from sklearn.impute import SimpleImputer
# # # from sklearn.feature_selection import RFE
# # # from sklearn.inspection import partial_dependence
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
# # # targets = {
# # #     '24h': data['Amount_24'],
# # #     '48h': data['Amount_48h'],
# # #     '72h': data['Amount_72h'],
# # #     '96h': data['Amount_96h']
# # # }
# # #
# # # # 分离数值型和非数值型特征
# # # numeric_features = features.select_dtypes(include=['number']).columns
# # # categorical_features = features.select_dtypes(include=['object']).columns
# # #
# # # # 创建预处理管道
# # # preprocessor = ColumnTransformer(
# # #     transformers=[
# # #         ('num', Pipeline(steps=[
# # #             ('imputer', SimpleImputer(strategy='mean')),
# # #             ('scaler', StandardScaler())
# # #         ]), numeric_features),
# # #         ('cat', Pipeline(steps=[
# # #             ('imputer', SimpleImputer(strategy='most_frequent')),
# # #             ('onehot', OneHotEncoder())
# # #         ]), categorical_features)
# # #     ])
# # #
# # #
# # # # RFE特征选择函数
# # # def get_rfe_selected_features(model, X_train, y_train, num_features):
# # #     rfe = RFE(estimator=model, n_features_to_select=num_features)
# # #     rfe.fit(X_train, y_train)
# # #     selected_features = np.array(numeric_features.tolist() + list(
# # #         preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)))[rfe.support_]
# # #     return rfe, selected_features
# # #
# # #
# # # # 函数：训练模型并进行ICE分析，并将每个时间点的所有特征绘制在同一张图中
# # # def train_model_ice_analysis_shared_axes(model, model_name, target_name, target, features, num_features):
# # #     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# # #
# # #     # 预处理数据
# # #     X_train_transformed = preprocessor.fit_transform(X_train)
# # #     X_test_transformed = preprocessor.transform(X_test)
# # #
# # #     # 通过RFE选择特征
# # #     rfe, selected_features = get_rfe_selected_features(model, X_train_transformed, y_train, num_features)
# # #     X_train_selected = X_train_transformed[:, rfe.support_]
# # #
# # #     # 训练模型
# # #     model.fit(X_train_selected, y_train)
# # #
# # #     # 设置子图布局
# # #     rows, cols = (3, 5) if num_features > 10 else (2, 5)
# # #     fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
# # #     axes = axes.flatten()
# # #
# # #     # 初始化X轴和Y轴的最小和最大值
# # #     x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
# # #
# # #     # 计算统一的X轴和Y轴范围
# # #     for i in range(num_features):
# # #         pd_results = partial_dependence(model, X_train_selected, [i], grid_resolution=100)
# # #         x_vals = pd_results[1][0]  # 取第一个维度的X值
# # #         y_vals = pd_results[0].ravel()  # 获取预测的平均值并展平
# # #
# # #         # 更新X和Y的全局最小/最大值
# # #         x_min, x_max = min(x_min, x_vals.min()), max(x_max, x_vals.max())
# # #         y_min, y_max = min(y_min, y_vals.min()), max(y_max, y_vals.max())
# # #
# # #     # 绘制每个特征的ICE曲线并设置相同的X和Y轴
# # #     for i, feature in enumerate(selected_features):
# # #         pd_results = partial_dependence(model, X_train_selected, [i], grid_resolution=100)
# # #         x_vals = pd_results[1][0]
# # #         y_vals = pd_results[0].ravel()
# # #
# # #         axes[i].plot(x_vals, y_vals)
# # #         axes[i].set_xlim(x_min, x_max)
# # #         axes[i].set_ylim(y_min, y_max)
# # #         axes[i].set_title(f'{feature}')
# # #
# # #     # 隐藏多余的子图框
# # #     for j in range(num_features, rows * cols):
# # #         axes[j].axis('off')
# # #
# # #     # 调整图形布局和标题
# # #     plt.suptitle(f'ICE Analysis for {target_name} - {model_name} ({num_features} Features)', fontsize=16)
# # #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# # #
# # #     # 保存图像
# # #     plt.savefig(f'Combined_ICE_Plot_{target_name}_{model_name}_SharedAxes.png')
# # #     plt.close()
# # #     print(f"Combined ICE图已保存至: Combined_ICE_Plot_{target_name}_{model_name}_SharedAxes.png")
# # #
# # #
# # # # 对每个时间点设置不同的特征数量
# # # num_features_dict = {
# # #     '24h': 15,
# # #     '48h': 10,
# # #     '72h': 15,
# # #     '96h': 15
# # # }
# # #
# # # # 定义模型
# # # models = {
# # #     'Random Forest': RandomForestRegressor(random_state=42),
# # #     'XGBoost': xgb.XGBRegressor(random_state=42)
# # # }
# # #
# # # # 分别对每个模型和时间段进行ICE分析并绘图
# # # for model_name, model in models.items():
# # #     for target_name, target in targets.items():
# # #         num_features = num_features_dict[target_name]
# # #         train_model_ice_analysis_shared_axes(
# # #             model=model,
# # #             model_name=model_name,
# # #             target_name=target_name,
# # #             target=target,
# # #             features=features,
# # #             num_features=num_features
# # #         )
# # #
# # # print("所有模型和时间段的ICE分析完成并已保存为共享坐标轴的图像！")
# #
# #
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import xgboost as xgb
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # from sklearn.compose import ColumnTransformer
# # from sklearn.pipeline import Pipeline
# # from sklearn.impute import SimpleImputer
# # from sklearn.feature_selection import RFE
# # from sklearn.inspection import PartialDependenceDisplay
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
# # targets = {
# #     '24h': data['Amount_24'],
# #     '48h': data['Amount_48h'],
# #     '72h': data['Amount_72h'],
# #     '96h': data['Amount_96h']
# # }
# #
# # # 分离数值型和非数值型特征
# # numeric_features = features.select_dtypes(include=['number']).columns
# # categorical_features = features.select_dtypes(include=['object']).columns
# #
# # # 创建预处理管道
# # preprocessor = ColumnTransformer(
# #     transformers=[
# #         ('num', Pipeline(steps=[
# #             ('imputer', SimpleImputer(strategy='mean')),
# #             ('scaler', StandardScaler())
# #         ]), numeric_features),
# #         ('cat', Pipeline(steps=[
# #             ('imputer', SimpleImputer(strategy='most_frequent')),
# #             ('onehot', OneHotEncoder())
# #         ]), categorical_features)
# #     ])
# #
# #
# # # RFE特征选择函数
# # def get_rfe_selected_features(model, X_train, y_train, num_features):
# #     rfe = RFE(estimator=model, n_features_to_select=num_features)
# #     rfe.fit(X_train, y_train)
# #     selected_features = np.array(numeric_features.tolist() + list(
# #         preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)))[rfe.support_]
# #     return rfe, selected_features
# #
# #
# # # 函数：训练模型并生成PDP+ICE图
# # def train_model_pdp_ice_plot(model, model_name, target_name, target, features, num_features):
# #     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# #
# #     # 预处理数据
# #     X_train_transformed = preprocessor.fit_transform(X_train)
# #     X_test_transformed = preprocessor.transform(X_test)
# #
# #     # 通过RFE选择特征
# #     rfe, selected_features = get_rfe_selected_features(model, X_train_transformed, y_train, num_features)
# #     X_train_selected = X_train_transformed[:, rfe.support_]
# #
# #     # 训练模型
# #     model.fit(X_train_selected, y_train)
# #
# #     # 设置子图布局
# #     rows, cols = (3, 5) if num_features > 10 else (2, 5)
# #     fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
# #     axes = axes.flatten()
# #
# #     # 绘制每个特征的PDP+ICE图
# #     for i, feature in enumerate(selected_features):
# #         PartialDependenceDisplay.from_estimator(
# #             model, X_train_selected, [i], ax=axes[i], kind='both',  # 'both' 会绘制PDP+ICE图
# #         )
# #         axes[i].set_title(f'{feature}')
# #
# #     # 隐藏多余的子图框
# #     for j in range(num_features, rows * cols):
# #         axes[j].axis('off')
# #
# #     # 调整图形布局和标题
# #     plt.suptitle(f'PDP + ICE for {target_name} - {model_name} ({num_features} Features)', fontsize=16)
# #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# #
# #     # 保存图像
# #     plt.savefig(f'PDP_ICE_{target_name}_{model_name}.png')
# #     plt.close()
# #     print(f"PDP + ICE图已保存至: PDP_ICE_{target_name}_{model_name}.png")
# #
# #
# # # 对每个时间点设置不同的特征数量
# # num_features_dict = {
# #     '24h': 15,
# #     '48h': 10,
# #     '72h': 15,
# #     '96h': 15
# # }
# #
# # # 定义模型
# # models = {
# #     'Random Forest': RandomForestRegressor(random_state=42),
# #     'XGBoost': xgb.XGBRegressor(random_state=42)
# # }
# #
# # # 分别对每个模型和时间段生成PDP+ICE图
# # for model_name, model in models.items():
# #     for target_name, target in targets.items():
# #         num_features = num_features_dict[target_name]
# #         train_model_pdp_ice_plot(
# #             model=model,
# #             model_name=model_name,
# #             target_name=target_name,
# #             target=target,
# #             features=features,
# #             num_features=num_features
# #         )
# #
# # print("所有模型和时间段的PDP + ICE分析完成并已保存为图像！")
#
#
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import xgboost as xgb
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.feature_selection import RFE
# from sklearn.inspection import PartialDependenceDisplay
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
#             ('imputer', SimpleImputer(strategy='mean')),
#             ('scaler', StandardScaler())
#         ]), numeric_features),
#         ('cat', Pipeline(steps=[
#             ('imputer', SimpleImputer(strategy='most_frequent')),
#             ('onehot', OneHotEncoder())
#         ]), categorical_features)
#     ])
#
#
# # RFE特征选择函数
# def get_rfe_selected_features(model, X_train, y_train, num_features):
#     rfe = RFE(estimator=model, n_features_to_select=num_features)
#     rfe.fit(X_train, y_train)
#     selected_features = np.array(numeric_features.tolist() + list(
#         preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)))[rfe.support_]
#     return rfe, selected_features
#
#
# # 函数：训练模型并生成每个特征的单独PDP+ICE图
# def train_model_pdp_ice_plot(model, model_name, target_name, target, features, num_features):
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
#
#     # 预处理数据
#     X_train_transformed = preprocessor.fit_transform(X_train)
#     X_test_transformed = preprocessor.transform(X_test)
#
#     # 通过RFE选择特征
#     rfe, selected_features = get_rfe_selected_features(model, X_train_transformed, y_train, num_features)
#     X_train_selected = X_train_transformed[:, rfe.support_]
#
#     # 训练模型
#     model.fit(X_train_selected, y_train)
#
#     # 为每个特征单独绘制PDP+ICE图
#     for i, feature in enumerate(selected_features):
#         fig, ax = plt.subplots(figsize=(8, 6))
#         PartialDependenceDisplay.from_estimator(
#             model, X_train_selected, [i], ax=ax, kind='both'  # 'both' 会绘制PDP+ICE图
#         )
#         ax.set_title(f'{model_name} at {target_name}', fontsize=14, fontweight='bold')
#         ax.set_xlabel(feature, fontsize=12, fontweight='bold')
#         ax.set_ylabel('Partial Dependence', fontsize=12, fontweight='bold')
#
#         # 调整字体加粗和图形美化
#         for label in ax.get_xticklabels() + ax.get_yticklabels():
#             label.set_fontweight('bold')
#
#         # 保存每个特征的图像
#         file_name = f'PDP_ICE_{target_name}_{model_name}_{feature}.png'
#         plt.savefig(file_name)
#         plt.close()
#         print(f"PDP + ICE图已保存至: {file_name}")
#
#
# # 对每个时间点设置不同的特征数量
# num_features_dict = {
#     '24h': 15,
#     '48h': 10,
#     '72h': 15,
#     '96h': 15
# }
#
# # 定义模型
# models = {
#     'Random Forest': RandomForestRegressor(random_state=42),
#     'XGBoost': xgb.XGBRegressor(random_state=42)
# }
#
# # 分别对每个模型和时间段生成PDP+ICE图
# for model_name, model in models.items():
#     for target_name, target in targets.items():
#         num_features = num_features_dict[target_name]
#         train_model_pdp_ice_plot(
#             model=model,
#             model_name=model_name,
#             target_name=target_name,
#             target=target,
#             features=features,
#             num_features=num_features
#         )
#
# print("所有模型和时间段的PDP + ICE分析完成并已保存为单独的图像！")


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.inspection import PartialDependenceDisplay

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
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
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

# 函数：训练模型并生成每个特征的单独PDP+ICE图
# 为每个特征单独绘制PDP+ICE图
def train_model_pdp_ice_plot(model, model_name, target_name, target, features, num_features, output_folder):
    # 检查并创建保存图像的文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # 预处理数据
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 通过RFE选择特征
    rfe, selected_features = get_rfe_selected_features(model, X_train_transformed, y_train, num_features)
    X_train_selected = X_train_transformed[:, rfe.support_]

    # 训练模型
    model.fit(X_train_selected, y_train)

    # 为每个特征单独绘制PDP+ICE图
    for i, feature in enumerate(selected_features):
        fig, ax = plt.subplots(figsize=(8, 6))

        # 绘制PDP+ICE图
        display = PartialDependenceDisplay.from_estimator(
            model, X_train_selected, [i], ax=ax, kind='both', feature_names=selected_features
        )

        # 手动清除所有标题
        ax.set_xlabel("")  # 清空 X 轴标题
        ax.set_ylabel("")  # 清空 Y 轴标题
        ax.set_title("")  # 清空子图标题

        # 确保主标题被设置
        fig.suptitle(f'{model_name} at {target_name}', fontsize=14, fontweight='bold', y=0.95)

        # 调整刻度字体
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        # 移除任何可能的子图默认标题
        fig.text(0.5, 0.04, '', ha='center')  # 清空底部全局X轴标题
        fig.text(0.04, 0.5, '', va='center', rotation='vertical')  # 清空全局Y轴标题

        # 保存图片
        file_name = os.path.join(output_folder, f'PDP_ICE_{target_name}_{model_name}_{feature}.png')
        plt.savefig(file_name, bbox_inches='tight')  # 确保不裁剪标题
        plt.close()
        print(f"图片已保存: {file_name}")


# 对每个时间点设置不同的特征数量
num_features_dict = {
    '24h': 15,
    '48h': 10,
    '72h': 15,
    '96h': 15
}

# 定义模型
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42)
}

# 设置输出图像的文件夹
output_folder = "C:/Users/xyz/Desktop/机器学习-毕业论文/输出图像"

# 分别对每个模型和时间段生成PDP+ICE图
for model_name, model in models.items():
    for target_name, target in targets.items():
        num_features = num_features_dict[target_name]
        train_model_pdp_ice_plot(
            model=model,
            model_name=model_name,
            target_name=target_name,
            target=target,
            features=features,
            num_features=num_features,
            output_folder=output_folder
        )

print("所有模型和时间段的PDP + ICE分析完成并已保存为单独的图像！")
