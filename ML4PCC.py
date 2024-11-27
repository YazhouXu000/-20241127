import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import seaborn as sns

# Read the data
file_path = 'Data_microalgae_1.xlsx'
data = pd.read_excel(file_path)

# Drop rows with missing values in target variables
data = data.dropna(subset=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])

# Select features and target variables
features = data.drop(columns=['Amount_24', 'Amount_48h', 'Amount_72h', 'Amount_96h'])
target_24 = data['Amount_24']
target_48 = data['Amount_48h']
target_72 = data['Amount_72h']
target_96 = data['Amount_96h']

# Separate numeric and categorical features
numeric_features = features.select_dtypes(include=['number']).columns
categorical_features = features.select_dtypes(include=['object']).columns

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
            ('scaler', StandardScaler())  # Scale numeric features
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values
            ('onehot', OneHotEncoder())  # Encode categorical features
        ]), categorical_features)
    ])

# Fit the preprocessor on the features to apply transformations
preprocessor.fit(features)

# Transform the features
transformed_features = preprocessor.transform(features)

# Convert the transformed features to a DataFrame for easier manipulation
encoded_feature_names = preprocessor.transformers_[0][2].tolist() + list(preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(categorical_features))
transformed_features_df = pd.DataFrame(transformed_features, columns=encoded_feature_names)

# Concatenate the targets to the transformed features DataFrame
complete_data = pd.concat([transformed_features_df, target_24, target_48, target_72, target_96], axis=1)

# Calculate the correlation matrix
correlation_matrix = complete_data.corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(18, 14))

# Adjust the subplot to move the plot up
plt.subplots_adjust(top=0.85, bottom=0.15)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5,
            cbar_kws={'shrink': 0.5})
plt.title('Feature and Target Correlation Heatmap', fontsize=20, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# Enlarge and bold the color bar text
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2)
cbar.set_label('Correlation Coefficient', fontsize=14, fontweight='bold')

# Set font weight for color bar labels to bold
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')


plt.savefig("PCC.png")