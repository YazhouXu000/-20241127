import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.inspection import partial_dependence
import itertools

# 读取数据
file_path = 'Data_microalgae_1.xlsx'
data = pd.read_excel(file_path)

# 丢弃目标变量中含有缺失值的行
data = data.dropna(subset=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])

# 将元素特征归类为“Medium”
medium_features = ['K', 'Na', 'Ca', 'Mg', 'Fe', 'Mn', 'Co', 'Cu', 'Mo', 'Cl', 'Zn', 'B', 'C', 'O', 'N', 'S', 'P', 'Si', 'H']

# 添加一个分类特征列来表示“Medium”类别
data['Medium'] = data[medium_features].sum(axis=1)

# 选择特征和目标变量
features = data.drop(columns=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])
target_24 = data['Amount_24']
target_48 = data['Amount_48h']
target_72 = data['Amount_72h']
target_96 = data['Amount_96h']

# 将类别特征转换为类别类型
categorical_features = features.select_dtypes(include=['object']).columns
for col in categorical_features:
    data[col] = data[col].astype('category')

# 分离数值型和非数值型特征
numeric_features = features.select_dtypes(include=['number']).columns
categorical_features = features.select_dtypes(include=['category']).columns

# 创建预处理管道
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # 填补数值特征中的缺失值
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 填补分类特征中的缺失值
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 独热编码
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 创建XGBoost模型的管道
xgboost_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(random_state=42, enable_categorical=True))
])

# 反标准化函数
def inverse_transform(scaled_values, mean, scale):
    return scaled_values * scale + mean

# 训练并生成部分依赖图的函数
def train_and_plot_2d_pdp(features, target, target_name, features_to_plot):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    xgboost_model.fit(X_train, y_train)

    # 提取预处理后的训练数据
    X_train_preprocessed = xgboost_model.named_steps['preprocessor'].transform(X_train)

    # 获取经过拟合后的OneHotEncoder特征名称
    onehot_encoder = xgboost_model.named_steps['preprocessor'].named_transformers_['cat']['onehot']
    onehot_encoder.fit(features[categorical_features])
    onehot_feature_names = onehot_encoder.get_feature_names_out(categorical_features).tolist()

    # 合并数值特征和独热编码特征的名称
    feature_names = numeric_features.tolist() + onehot_feature_names

    # 确保features_to_plot中的特征在feature_names中
    features_to_plot = [f for f in features_to_plot if f in feature_names]

    # 生成二阶部分依赖图
    pairs = list(itertools.combinations(features_to_plot, 2))

    for pair in pairs:
        feature_1, feature_2 = pair
        feature_idx_1 = feature_names.index(feature_1)
        feature_idx_2 = feature_names.index(feature_2)

        try:
            pdp_results = partial_dependence(
                xgboost_model.named_steps['regressor'],
                features=[feature_idx_1, feature_idx_2],
                X=X_train_preprocessed,
                grid_resolution=50
            )

            # 使用 'grid_values' 而不是 'values'
            x_data_1 = pdp_results['grid_values'][0]
            x_data_2 = pdp_results['grid_values'][1]
            y_data = pdp_results['average']

            # 去除y_data的额外维度
            y_data = y_data[0]

            # 反归一化 x 轴 和 y 轴
            if feature_1 in numeric_features:
                inverse_x_data_1 = inverse_transform(np.array(x_data_1),
                                                     xgboost_model.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler'].mean_[feature_idx_1],
                                                     xgboost_model.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler'].scale_[feature_idx_1])
            else:
                inverse_x_data_1 = x_data_1  # 如果是类别特征，则不变

            if feature_2 in numeric_features:
                inverse_x_data_2 = inverse_transform(np.array(x_data_2),
                                                     xgboost_model.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler'].mean_[feature_idx_2],
                                                     xgboost_model.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler'].scale_[feature_idx_2])
            else:
                inverse_x_data_2 = x_data_2  # 如果是类别特征，则不变

            # 创建等高线图
            X, Y = np.meshgrid(inverse_x_data_1, inverse_x_data_2)
            fig, ax = plt.subplots(figsize=(10, 8))
            contour = ax.contourf(X, Y, y_data.T, cmap=plt.cm.viridis)
            plt.colorbar(contour, ax=ax)

            ax.set_title(f'Two-way PD for {target_name}\n{feature_1} and {feature_2}', fontsize=16, fontweight='bold')
            ax.set_xlabel(feature_1, fontsize=14, fontweight='bold')
            ax.set_ylabel(feature_2,fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=10, width=2)
            for tick in ax.get_xticklabels():
                tick.set_fontsize(12)
                tick.set_fontweight('bold')
            for tick in ax.get_yticklabels():
                tick.set_fontsize(12)
                tick.set_fontweight('bold')

            plt.tight_layout()
            plt.savefig(f'Two_way_Partial_Dependence_Plot_{target_name}_{feature_1}_{feature_2}.png', format='png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot saved: pdp_{target_name}_{feature_1}_{feature_2}.png")

        except Exception as e:
            print(f"Error processing combination {feature_1} & {feature_2}: {e}")

print("All 2-way PDP plots saved.")

# 定义要绘制的特征对
features_to_plot = ['Temperature', 'Illumination', 'Size (μm)', 'MP']

# 对不同时间点进行分析
train_and_plot_2d_pdp(features, target_24, '24 Hour Amount', features_to_plot)
train_and_plot_2d_pdp(features, target_48, '48 Hour Amount', features_to_plot)
train_and_plot_2d_pdp(features, target_72, '72 Hour Amount', features_to_plot)
train_and_plot_2d_pdp(features, target_96, '96 Hour Amount', features_to_plot)
