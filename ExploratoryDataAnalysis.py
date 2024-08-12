import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the cleaned dataset
dataset = pd.read_csv('cleaned_dataset.csv')

# Remove non-numeric columns for correlation matrix calculation
numeric_dataset = dataset.select_dtypes(include=[np.number])

# Summary statistics for numerical features
print(numeric_dataset.describe())

# Histogram for all numerical features
numerical_features = numeric_dataset.columns.tolist()
numeric_dataset.hist(bins=15, figsize=(15, 10), layout=(4, 4))
plt.tight_layout()
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(14, 12))
correlation_matrix = numeric_dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap=plt.cm.Reds)
plt.show()

# Count plot for the target variable
sns.countplot(x='Phishing attack', data=numeric_dataset)
plt.title('Distribution of Classes for Phishing Attack')
plt.show()

# Select categorical variables
categorical_dataset = dataset.select_dtypes(include=['object']).columns


# Plot count plots for all categorical variables
for col in categorical_dataset:
    plt.figure(figsize=(10, 4))
    sns.countplot(x=col, data=dataset, hue=col, palette='viridis', dodge=False)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=90)  # Rotate the x labels if they overlap
    plt.legend([],[], frameon=False)  # Hide the legend
    plt.show()
