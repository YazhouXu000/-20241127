import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
file_path = 'Data_microalgae_1.xlsx'
data = pd.read_excel(file_path)

# # 设置绘图风格
# sns.set(style="whitegrid")
#
# # Identify non-numeric and numeric columns
# non_numeric_columns = data.select_dtypes(include=['object']).columns
# numeric_columns = data.select_dtypes(include=['number']).columns
#
# # Determine the total number of plots
# num_features = len(data.columns)
# num_cols = 5
# num_rows = (num_features // num_cols) + (num_features % num_cols > 0)
#
# # Create a figure to plot histograms and boxplots with different colors
# plt.figure(figsize=(20, num_rows * 4))
#
#
# # Define color palette
# colors = sns.color_palette("husl", num_features)
#
# # Font properties
# font_properties = {'fontsize': 14, 'fontweight': 'bold'}
#
# # Generate histograms for non-numeric features with different colors
# for i, (column, color) in enumerate(zip(non_numeric_columns, colors)):
#     plt.subplot(num_rows, num_cols, i + 1)
#     sns.countplot(data[column], color=color)
#     plt.title(column, **font_properties)
#     plt.xlabel(column, **font_properties)
#     plt.ylabel('Count', **font_properties)
#     plt.xticks(fontsize=12, fontweight='bold')
#     plt.yticks(fontsize=12, fontweight='bold')
#     plt.tight_layout()
#
# # Generate boxplots for numeric features with different colors
# for i, (column, color) in enumerate(zip(numeric_columns, colors[len(non_numeric_columns):]), start=len(non_numeric_columns)):
#     plt.subplot(num_rows, num_cols, i + 1)
#     sns.boxplot(x=data[column], color=color)
#     plt.title(column, **font_properties)
#     plt.xlabel(column, **font_properties)
#     plt.ylabel('Values', **font_properties)
#     plt.xticks(fontsize=12, fontweight='bold')
#     plt.yticks(fontsize=12, fontweight='bold')
#     plt.tight_layout()
#
#
# plt.savefig("Data distribution_box_plot.png")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

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

# 创建机器学习管道
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])


# 函数：训练模型并绘制结果
def train_and_plot(features, target, target_name, ax):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    ax.scatter(y_train, y_train_pred, label='Train', alpha=0.7, s=100, edgecolor='k', linewidths=1.5)
    ax.scatter(y_test, y_test_pred, label='Test', alpha=0.7, s=100, edgecolor='k', linewidths=1.5)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
    ax.set_xlabel('Actual', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted', fontsize=14, fontweight='bold')
    ax.set_title(f'Actual vs Predicted for {target_name} by RF', fontsize=16, fontweight='bold')

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

    ax.text(0.05, 0.85, f'Train MSE: {train_mse:.2e}\nTrain R²: {train_r2:.2f}', ha='left', va='top',
            transform=ax.transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))
    ax.text(0.95, 0.15, f'Test MSE: {test_mse:.2e}\nTest R²: {test_r2:.2f}', ha='right', va='bottom',
            transform=ax.transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))


# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(20, 15))

# 训练模型并绘制结果
train_and_plot(features, target_24, '24 Hour Amount', axs[0, 0])
train_and_plot(features, target_48, '48 Hour Amount', axs[0, 1])
train_and_plot(features, target_72, '72 Hour Amount', axs[1, 0])
train_and_plot(features, target_96, '96 Hour Amount', axs[1, 1])

# 调整布局并保存图像
plt.tight_layout()
plt.savefig('Actual_vs_Predicted_All.png')


