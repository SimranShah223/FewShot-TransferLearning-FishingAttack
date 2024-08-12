import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load the categorical dataset
df_categorical = pd.read_csv('Categorical_engineered_dataset.csv')

# Assuming 'Phishing attack' is the target column and 'URL' needs to be excluded
X_categorical = df_categorical.drop(['Phishing attack'], axis=1)
y_categorical = df_categorical['Phishing attack']

# Split the data into training and test sets
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_categorical, y_categorical, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
rf_categorical = RandomForestClassifier(n_estimators=100, random_state=42)
rf_categorical.fit(X_train_cat, y_train_cat)

# Predict on the test set
y_pred_cat = rf_categorical.predict(X_test_cat)

# Evaluate the model
print("Categorical Dataset - Random Forest Performance")
print(classification_report(y_test_cat, y_pred_cat))
print("Accuracy:", accuracy_score(y_test_cat, y_pred_cat))

# Get feature importances
importances = rf_categorical.feature_importances_
# Get the feature names
features = X_train_cat.columns
# Sort the feature importances in descending order and get the indices
indices = np.argsort(importances)[::-1]

# Plotting
plt.figure(figsize=(10, 6))
plt.title('Feature Importances in Random Forest Model')
plt.bar(range(X_train_cat.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train_cat.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()
