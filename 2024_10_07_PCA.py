import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 读取数据
file_path = 'C:/Users/xyz/Desktop/Data_microalgae_1.xlsx'
data = pd.read_excel(file_path)

# 丢弃目标变量中含有缺失值的行
data = data.dropna(subset=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])

# 选择特征和目标变量
features = data.drop(columns=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])

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

# Step 1: 数据预处理
preprocessed_features = preprocessor.fit_transform(features)

# Step 2: Apply PCA
pca = PCA()
pca_features = pca.fit_transform(preprocessed_features)

# Step 3: Plot explained variance ratio (PCA Explained Variance Ratio 图)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 7))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, color='blue', label='Explained Variance')
plt.step(range(1, len(explained_variance) + 1), cumulative_variance, where='mid', label='Cumulative Variance', color='orange')

plt.xlabel('Principal Components', fontsize=14, fontweight='bold')
plt.ylabel('Variance Explained', fontsize=14, fontweight='bold')
plt.title('PCA Explained Variance Ratio', fontsize=16, fontweight='bold')
plt.legend(loc='best', fontsize=12)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(True)

plt.tight_layout()
plt.show()

# Step 4: PCA 2D Scatter Plot (PCA 2D 散点图)
pca_2d = PCA(n_components=2)
pca_2d_features = pca_2d.fit_transform(preprocessed_features)

# Step 5: Plot the 2D PCA results
plt.figure(figsize=(10, 7))
plt.scatter(pca_2d_features[:, 0], pca_2d_features[:, 1], c='blue', edgecolor='k', alpha=0.7)
plt.xlabel('Principal Component 1', fontsize=14, fontweight='bold')
plt.ylabel('Principal Component 2', fontsize=14, fontweight='bold')
plt.title('PCA 2D Projection', fontsize=16, fontweight='bold')

plt.grid(True)
plt.tight_layout()
plt.show()
