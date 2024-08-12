import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Define the path to your dataset
file_path = 'Updated_Phishing_Detection_Dataset.csv'

# Load the dataset
dataset = pd.read_csv(file_path)

# Display the first few rows of the dataset to confirm it's loaded correctly
print(dataset.head())

# Display a summary of the dataset
print(dataset.info())

# Basic statistics
print(dataset.describe())

# Quick visualization, e.g., distribution of a feature
sns.histplot(dataset['URL Length'], bins=30, kde=True)
plt.show()

# Check for missing values
print(dataset.isnull().sum())

# Drop any rows with missing values
dataset_cleaned = dataset.dropna()

# Drop duplicate rows, keeping the first occurrence
dataset_cleaned = dataset_cleaned.drop_duplicates()

# Save the cleaned dataset
dataset_cleaned.to_csv('cleaned_dataset.csv', index=False)
