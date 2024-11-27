# # # import pandas as pd
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import shap
# # # import xgboost as xgb
# # # from sklearn.ensemble import RandomForestRegressor
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # # from sklearn.compose import ColumnTransformer
# # # from sklearn.pipeline import Pipeline
# # # from sklearn.impute import SimpleImputer
# # # from sklearn.feature_selection import RFE
# # # from sklearn.inspection import PartialDependenceDisplay
# # # import plotly.graph_objects as go
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
# # #             ('imputer', SimpleImputer(strategy='mean')),  # 填补数值特征中的缺失值
# # #             ('scaler', StandardScaler())
# # #         ]), numeric_features),
# # #         ('cat', Pipeline(steps=[
# # #             ('imputer', SimpleImputer(strategy='most_frequent')),  # 填补分类特征中的缺失值
# # #             ('onehot', OneHotEncoder())
# # #         ]), categorical_features)
# # #     ])
# # #
# # # # RFE特征选择函数
# # # def get_rfe_selected_features(model, X_train, y_train, num_features):
# # #     rfe = RFE(estimator=model, n_features_to_select=num_features)
# # #     rfe.fit(X_train, y_train)
# # #     selected_features = np.array(numeric_features.tolist() + list(
# # #         preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)))[rfe.support_]
# # #     return rfe, selected_features
# # #
# # # # 函数：训练模型并进行SHAP分析
# # # def train_model_shap_analysis(model, target_name, target, features, num_features):
# # #     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# # #
# # #     # 预处理数据
# # #     X_train_transformed = preprocessor.fit_transform(X_train)
# # #     X_test_transformed = preprocessor.transform(X_test)
# # #
# # #     # 通过RFE选择特征
# # #     rfe, selected_features = get_rfe_selected_features(model, X_train_transformed, y_train, num_features)
# # #     X_train_selected = X_train_transformed[:, rfe.support_]
# # #     X_test_selected = X_test_transformed[:, rfe.support_]
# # #
# # #     # 训练模型
# # #     model.fit(X_train_selected, y_train)
# # #
# # #     # 使用SHAP解释模型
# # #     explainer = shap.Explainer(model, X_train_selected)
# # #     shap_values = explainer(X_train_selected, check_additivity=False)  # 关闭SHAP加性检查
# # #
# # #     # 绘制SHAP图
# # #     plt.figure(figsize=(10, 6))
# # #     shap.summary_plot(shap_values, X_train_selected, feature_names=selected_features, show=False)
# # #     plt.title(f'SHAP Summary Plot - {target_name}')
# # #     plt.savefig(f'SHAP_Summary_Plot_{target_name}.png')
# # #     plt.close()
# # #
# # #     # ICE图
# # #     fig, ax = plt.subplots(figsize=(10, 6))
# # #     PartialDependenceDisplay.from_estimator(model, X_train_selected, [0], ax=ax)  # 使用 PartialDependenceDisplay
# # #     plt.title(f'ICE Plot - {target_name} Feature 0')
# # #     plt.savefig(f'ICE_Plot_{target_name}.png')
# # #     plt.close()
# # #
# # #     # 2D SHAP交互图
# # #     plt.figure(figsize=(10, 6))
# # #
# # #     # 修正：使用 shap_values.values 来获取 numpy 数组格式的 SHAP 值
# # #     shap.dependence_plot(0, shap_values.values, X_train_selected, feature_names=selected_features, show=False)
# # #     shap.dependence_plot(1, shap_values.values, X_train_selected, feature_names=selected_features, show=False)
# # #     plt.title(f'2D SHAP Interaction - {target_name}')
# # #     plt.savefig(f'SHAP_2D_Interaction_{target_name}.png')
# # #     plt.close()
# # #
# # #     # 3D SHAP交互图
# # #     fig = go.Figure(data=[go.Scatter3d(
# # #         feature_x=selected_features[0]  # 特征0
# # #         feature_y = selected_features[1]  # 特征1
# # #         z=shap_values.values[:, 0],  # 修正：确保 SHAP 值是 numpy 数组格式
# # #         mode='markers',
# # #         marker=dict(
# # #             size=5,
# # #             color=shap_values.values[:, 0],  # 设置颜色为 SHAP 值
# # #             colorscale='Viridis',
# # #             opacity=0.8
# # #         )
# # #     )])
# # #
# # #     fig.update_layout(
# # #         title=f'3D SHAP Interaction - {target_name}',
# # #         scene=dict(
# # #             xaxis_title='Feature 0',
# # #             yaxis_title='Feature 1',
# # #             zaxis_title='SHAP Value'
# # #         )
# # #     )
# # #
# # #     fig.write_html(f'SHAP_3D_Interaction_{target_name}.html')  # 保存为交互式HTML文件
# # #
# # #     print(f"SHAP分析完成并保存至: SHAP_Feature_Importance_{target_name}.xlsx 和 SHAP_Summary_Plot_{target_name}.png")
# # #
# # # # 对每个时间点设置不同的特征数量
# # # num_features_dict = {
# # #     '24h': 15,
# # #     '48h': 10,
# # #     '72h': 15,
# # #     '96h': 15
# # # }
# # #
# # # # 分别对24h, 48h, 72h, 96h的目标变量进行模型训练和SHAP分析
# # # for model_name, model in {
# # #     'Random Forest': RandomForestRegressor(random_state=42),
# # #     'XGBoost': xgb.XGBRegressor(random_state=42)
# # # }.items():
# # #     for target_name, target in targets.items():
# # #         num_features = num_features_dict[target_name]
# # #         train_model_shap_analysis(model, f"{model_name}_{target_name}", target, features, num_features)
# # #
# # # print("所有目标的SHAP分析完成！")
# #
# #
# #
# #
# #
# #
# # #
# # # import pandas as pd
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import shap
# # # import xgboost as xgb
# # # from sklearn.ensemble import RandomForestRegressor
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # # from sklearn.compose import ColumnTransformer
# # # from sklearn.pipeline import Pipeline
# # # from sklearn.impute import SimpleImputer
# # # from sklearn.feature_selection import RFE
# # # from sklearn.inspection import PartialDependenceDisplay
# # # import plotly.graph_objects as go
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
# # #             ('imputer', SimpleImputer(strategy='mean')),  # 填补数值特征中的缺失值
# # #             ('scaler', StandardScaler())
# # #         ]), numeric_features),
# # #         ('cat', Pipeline(steps=[
# # #             ('imputer', SimpleImputer(strategy='most_frequent')),  # 填补分类特征中的缺失值
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
# # # # 函数：训练模型并进行SHAP分析
# # # def train_model_shap_analysis(model, target_name, target, features, num_features):
# # #     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# # #
# # #     # 预处理数据
# # #     X_train_transformed = preprocessor.fit_transform(X_train)
# # #     X_test_transformed = preprocessor.transform(X_test)
# # #
# # #     # 通过RFE选择特征
# # #     rfe, selected_features = get_rfe_selected_features(model, X_train_transformed, y_train, num_features)
# # #     X_train_selected = X_train_transformed[:, rfe.support_]
# # #     X_test_selected = X_test_transformed[:, rfe.support_]
# # #
# # #     # 训练模型
# # #     model.fit(X_train_selected, y_train)
# # #
# # #     # 使用SHAP解释模型
# # #     explainer = shap.Explainer(model, X_train_selected)
# # #     shap_values = explainer(X_train_selected, check_additivity=False)  # 关闭SHAP加性检查
# # #
# # #     # 绘制SHAP图
# # #     plt.figure(figsize=(10, 6))
# # #     shap.summary_plot(shap_values, X_train_selected, feature_names=selected_features, show=False)
# # #     plt.title(f'SHAP Summary Plot - {target_name}')
# # #     plt.savefig(f'SHAP_Summary_Plot_{target_name}.png')
# # #     plt.close()
# # #
# # #     # ICE图
# # #     fig, ax = plt.subplots(figsize=(10, 6))
# # #     PartialDependenceDisplay.from_estimator(model, X_train_selected, [0], ax=ax)  # 使用 PartialDependenceDisplay
# # #     plt.title(f'ICE Plot - {target_name} Feature 0')
# # #     plt.savefig(f'ICE_Plot_{target_name}.png')
# # #     plt.close()
# # #
# # #     # 2D SHAP交互图
# # #     plt.figure(figsize=(10, 6))
# # #
# # #     # 修正：使用 shap_values.values 来获取 numpy 数组格式的 SHAP 值
# # #     shap.dependence_plot(0, shap_values.values, X_train_selected, feature_names=selected_features, show=False)
# # #     shap.dependence_plot(1, shap_values.values, X_train_selected, feature_names=selected_features, show=False)
# # #     plt.title(f'2D SHAP Interaction - {target_name}')
# # #     plt.savefig(f'SHAP_2D_Interaction_{target_name}.png')
# # #     plt.close()
# # #
# # #     # 3D SHAP交互图
# # #     # 确定用于X和Y轴的特征名称
# # #     feature_x = selected_features[0]  # 特征0
# # #     feature_y = selected_features[1]  # 特征1
# # #
# # #     fig = go.Figure(data=[go.Scatter3d(
# # #         x=X_train_selected[:, 0],
# # #         y=X_train_selected[:, 1],
# # #         z=shap_values.values[:, 0],  # 修正：确保 SHAP 值是 numpy 数组格式
# # #         mode='markers',
# # #         marker=dict(
# # #             size=5,
# # #             color=shap_values.values[:, 0],  # 设置颜色为 SHAP 值
# # #             colorscale='Viridis',
# # #             opacity=0.8
# # #         )
# # #     )])
# # #
# # #     # 更新3D图的坐标轴名称，确保清晰标注特征名称
# # #     fig.update_layout(
# # #         title=f'3D SHAP Interaction - {target_name}',
# # #         scene=dict(
# # #             xaxis_title=feature_x,  # X轴对应的特征名称
# # #             yaxis_title=feature_y,  # Y轴对应的特征名称
# # #             zaxis_title='SHAP Value'  # Z轴表示SHAP值
# # #         )
# # #     )
# # #
# # #     fig.write_html(f'SHAP_3D_Interaction_{target_name}.html')  # 保存为交互式HTML文件
# # #
# # #     print(f"SHAP分析完成并保存至: SHAP_Feature_Importance_{target_name}.xlsx 和 SHAP_Summary_Plot_{target_name}.png")
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
# # # # 分别对24h, 48h, 72h, 96h的目标变量进行模型训练和SHAP分析
# # # for model_name, model in {
# # #     'Random Forest': RandomForestRegressor(random_state=42),
# # #     'XGBoost': xgb.XGBRegressor(random_state=42)
# # # }.items():
# # #     for target_name, target in targets.items():
# # #         num_features = num_features_dict[target_name]
# # #         train_model_shap_analysis(model, f"{model_name}_{target_name}", target, features, num_features)
# # #
# # # print("所有目标的SHAP分析完成！")
# #
# #
# #
# #
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import shap
# # import xgboost as xgb
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # from sklearn.compose import ColumnTransformer
# # from sklearn.pipeline import Pipeline
# # from sklearn.impute import SimpleImputer
# # from sklearn.feature_selection import RFE
# # from sklearn.inspection import PartialDependenceDisplay
# # import plotly.graph_objects as go
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
# #             ('imputer', SimpleImputer(strategy='mean')),  # 填补数值特征中的缺失值
# #             ('scaler', StandardScaler())
# #         ]), numeric_features),
# #         ('cat', Pipeline(steps=[
# #             ('imputer', SimpleImputer(strategy='most_frequent')),  # 填补分类特征中的缺失值
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
# # # 函数：训练模型并进行SHAP分析
# # def train_model_shap_analysis(model, target_name, target, features, num_features):
# #     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# #
# #     # 预处理数据
# #     X_train_transformed = preprocessor.fit_transform(X_train)
# #     X_test_transformed = preprocessor.transform(X_test)
# #
# #     # 通过RFE选择特征
# #     rfe, selected_features = get_rfe_selected_features(model, X_train_transformed, y_train, num_features)
# #     X_train_selected = X_train_transformed[:, rfe.support_]
# #     X_test_selected = X_test_transformed[:, rfe.support_]
# #
# #     # 训练模型
# #     model.fit(X_train_selected, y_train)
# #
# #     # 使用SHAP解释模型
# #     explainer = shap.Explainer(model, X_train_selected)
# #     shap_values = explainer(X_train_selected, check_additivity=False)  # 关闭SHAP加性检查
# #
# #     # 绘制SHAP图
# #     plt.figure(figsize=(10, 6))
# #     shap.summary_plot(shap_values, X_train_selected, feature_names=selected_features, show=False)
# #     plt.title(f'SHAP Summary Plot - {target_name}')
# #     plt.savefig(f'SHAP_Summary_Plot_{target_name}.png')
# #     plt.close()
# #
# #     # ICE图
# #     fig, ax = plt.subplots(figsize=(10, 6))
# #     pdp = PartialDependenceDisplay.from_estimator(model, X_train_selected, [0], ax=ax)  # 保存返回值到 pdp
# #     plt.title(f'ICE Plot - {target_name} Feature 0')
# #     plt.savefig(f'ICE_Plot_{target_name}.png')
# #     plt.close()
# #
# #     # 获取X轴的特征值（Grid Values）
# #     grid_values = pdp.lines_[0][0].get_xdata()  # 提取网格特征值
# #
# #
# #     # 将ICE数值保存到Excel中
# #     for pd_line in pdp.lines_:
# #         grid_values = pd_line[0].get_xdata()  # 获取网格值
# #         ice_values = pd_line[0].get_ydata()  # 获取ICE值
# #
# #         # 将每个特征的 ICE 数值保存到Excel
# #         ice_df = pd.DataFrame({
# #             'Grid Values': grid_values,
# #             'ICE Values': ice_values
# #         })
# #         ice_filename = f"ICE_Values_{target_name}_feature_{pdp.features[0]}.xlsx"  # 保存每个特征的ICE
# #         ice_df.to_excel(ice_filename, index=False)
# #         print(f"ICE数值已保存至: {ice_filename}")
# #
# #     # 2D SHAP交互图
# #     plt.figure(figsize=(10, 6))
# #
# #     # 修正：使用 shap_values.values 来获取 numpy 数组格式的 SHAP 值
# #     shap.dependence_plot(0, shap_values.values, X_train_selected, feature_names=selected_features, show=False)
# #     shap.dependence_plot(1, shap_values.values, X_train_selected, feature_names=selected_features, show=False)
# #     plt.title(f'2D SHAP Interaction - {target_name}')
# #     plt.savefig(f'SHAP_2D_Interaction_{target_name}.png')
# #     plt.close()
# #
# #     # 3D SHAP交互图
# #     # 确定用于X和Y轴的特征名称
# #     feature_x = selected_features[0]  # 特征0
# #     feature_y = selected_features[1]  # 特征1
# #
# #     fig = go.Figure(data=[go.Scatter3d(
# #         x=X_train_selected[:, 0],
# #         y=X_train_selected[:, 1],
# #         z=shap_values.values[:, 0],  # 修正：确保 SHAP 值是 numpy 数组格式
# #         mode='markers',
# #         marker=dict(
# #             size=5,
# #             color=shap_values.values[:, 0],  # 设置颜色为 SHAP 值
# #             colorscale='Viridis',
# #             opacity=0.8
# #         )
# #     )])
# #
# #     # 更新3D图的坐标轴名称，确保清晰标注特征名称
# #     fig.update_layout(
# #         title=f'3D SHAP Interaction - {target_name}',
# #         scene=dict(
# #             xaxis_title=feature_x,  # X轴对应的特征名称
# #             yaxis_title=feature_y,  # Y轴对应的特征名称
# #             zaxis_title='SHAP Value'  # Z轴表示SHAP值
# #         )
# #     )
# #
# #     fig.write_html(f'SHAP_3D_Interaction_{target_name}.html')  # 保存为交互式HTML文件
# #
# #     print(f"SHAP分析完成并保存至: SHAP_Feature_Importance_{target_name}.xlsx 和 SHAP_Summary_Plot_{target_name}.png")
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
# # # 分别对24h, 48h, 72h, 96h的目标变量进行模型训练和SHAP分析
# # for model_name, model in {
# #     'Random Forest': RandomForestRegressor(random_state=42),
# #     'XGBoost': xgb.XGBRegressor(random_state=42)
# # }.items():
# #     for target_name, target in targets.items():
# #         num_features = num_features_dict[target_name]
# #         train_model_shap_analysis(model, f"{model_name}_{target_name}", target, features, num_features)
# #
# # print("所有目标的SHAP分析完成！")
#
#
#
#
#
#
#
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import shap
# import xgboost as xgb
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.feature_selection import RFE
# from sklearn.inspection import PartialDependenceDisplay
# import plotly.graph_objects as go
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
# # 函数：训练模型并进行PDP + SHAP 2D/3D交互分析
# def train_model_pdp_shap_analysis(model, target_name, target, features, num_features):
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
#
#     # 预处理数据
#     X_train_transformed = preprocessor.fit_transform(X_train)
#     X_test_transformed = preprocessor.transform(X_test)
#
#     # 通过RFE选择特征
#     rfe = RFE(estimator=model, n_features_to_select=num_features)
#     rfe.fit(X_train_transformed, y_train)
#     selected_features = np.array(numeric_features.tolist() + list(
#         preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)))[rfe.support_]
#     X_train_selected = X_train_transformed[:, rfe.support_]
#     X_test_selected = X_test_transformed[:, rfe.support_]
#
#     # 训练模型
#     model.fit(X_train_selected, y_train)
#
#     # 使用SHAP解释模型
#     explainer = shap.Explainer(model, X_train_selected)
#     shap_values = explainer(X_train_selected, check_additivity=False)
#
#     # 获取最重要的两个特征用于2D交互分析和三个特征用于3D交互分析
#     mean_shap_values = np.abs(shap_values.values).mean(axis=0)
#     important_features = selected_features[np.argsort(-mean_shap_values)]
#     feature_x, feature_y = important_features[0], important_features[1]
#     feature_z = important_features[2] if len(important_features) > 2 else None
#
#     # 绘制2D SHAP交互图
#     plt.figure(figsize=(10, 6))
#     shap.dependence_plot(
#         feature_x, shap_values.values, X_train_selected, feature_names=selected_features, interaction_index=feature_y, show=False
#     )
#     plt.title(f'2D SHAP Interaction - {target_name} ({feature_x} vs {feature_y})')
#     plt.savefig(f'SHAP_2D_Interaction_{feature_x}_{feature_y}_{target_name}.png')
#     plt.close()
#
#     # 使用SHAP绘制3D交互图（如果有足够的特征）
#     if feature_z:
#         fig = go.Figure(data=[go.Scatter3d(
#             x=X_train_selected[:, np.where(selected_features == feature_x)[0][0]],
#             y=X_train_selected[:, np.where(selected_features == feature_y)[0][0]],
#             z=shap_values.values[:, np.where(selected_features == feature_z)[0][0]],
#             mode='markers',
#             marker=dict(
#                 size=5,
#                 color=shap_values.values[:, np.where(selected_features == feature_z)[0][0]],
#                 colorscale='Viridis',
#                 opacity=0.8
#             )
#         )])
#         fig.update_layout(
#             title=f'3D SHAP Interaction - {target_name} ({feature_x}, {feature_y}, {feature_z})',
#             scene=dict(
#                 xaxis_title=feature_x,
#                 yaxis_title=feature_y,
#                 zaxis_title=f'SHAP Value for {feature_z}'
#             )
#         )
#         fig.write_html(f'SHAP_3D_Interaction_{feature_x}_{feature_y}_{feature_z}_{target_name}.html')
#
#     print(f"{target_name} - PDP + SHAP 2D和3D交互分析完成：主要特征为 {feature_x}, {feature_y}, 和 {feature_z}")
#
# # 对每个时间点设置不同的特征数量
# num_features_dict = {
#     '24h': 15,
#     '48h': 10,
#     '72h': 15,
#     '96h': 15
# }
#
# # 分别对24h, 48h, 72h, 96h的目标变量进行模型训练和PDP + SHAP交互分析
# for model_name, model in {
#     'Random Forest': RandomForestRegressor(random_state=42),
#     'XGBoost': xgb.XGBRegressor(random_state=42)
# }.items():
#     for target_name, target in targets.items():
#         num_features = num_features_dict[target_name]
#         train_model_pdp_shap_analysis(model, f"{model_name}_{target_name}", target, features, num_features)
#
# print("所有目标的PDP + SHAP交互分析完成！")
#


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
from sklearn.inspection import PartialDependenceDisplay
import plotly.graph_objects as go

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

    # ICE图
    fig, ax = plt.subplots(figsize=(10, 6))
    pdp = PartialDependenceDisplay.from_estimator(model, X_train_selected, [0], ax=ax)  # 保存返回值到 pdp
    plt.title(f'ICE Plot - {target_name} Feature 0')
    plt.savefig(f'ICE_Plot_{target_name}.png')
    plt.close()

    # 获取X轴的特征值（Grid Values）
    grid_values = pdp.lines_[0][0].get_xdata()  # 提取网格特征值


    # 将ICE数值保存到Excel中
    for pd_line in pdp.lines_:
        grid_values = pd_line[0].get_xdata()  # 获取网格值
        ice_values = pd_line[0].get_ydata()  # 获取ICE值

        # 将每个特征的 ICE 数值保存到Excel
        ice_df = pd.DataFrame({
            'Grid Values': grid_values,
            'ICE Values': ice_values
        })
        ice_filename = f"ICE_Values_{target_name}_feature_{pdp.features[0]}.xlsx"  # 保存每个特征的ICE
        ice_df.to_excel(ice_filename, index=False)
        print(f"ICE数值已保存至: {ice_filename}")

    # 2D SHAP交互图
    plt.figure(figsize=(10, 6))

    # 修正：使用 shap_values.values 来获取 numpy 数组格式的 SHAP 值
    shap.dependence_plot(0, shap_values.values, X_train_selected, feature_names=selected_features, show=False)
    shap.dependence_plot(1, shap_values.values, X_train_selected, feature_names=selected_features, show=False)
    plt.title(f'2D SHAP Interaction - {target_name}')
    plt.savefig(f'SHAP_2D_Interaction_{target_name}.png')
    plt.close()

    # 3D SHAP交互图
    # 确定用于X和Y轴的特征名称
    feature_x = selected_features[0]  # 特征0
    feature_y = selected_features[1]  # 特征1

    fig = go.Figure(data=[go.Scatter3d(
        x=X_train_selected[:, 0],
        y=X_train_selected[:, 1],
        z=shap_values.values[:, 0],  # 修正：确保 SHAP 值是 numpy 数组格式
        mode='markers',
        marker=dict(
            size=5,
            color=shap_values.values[:, 0],  # 设置颜色为 SHAP 值
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    # 更新3D图的坐标轴名称，确保清晰标注特征名称
    fig.update_layout(
        title=f'3D SHAP Interaction - {target_name}',
        scene=dict(
            xaxis_title=feature_x,  # X轴对应的特征名称
            yaxis_title=feature_y,  # Y轴对应的特征名称
            zaxis_title='SHAP Value'  # Z轴表示SHAP值
        )
    )

    fig.write_html(f'SHAP_3D_Interaction_{target_name}.html')  # 保存为交互式HTML文件

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
