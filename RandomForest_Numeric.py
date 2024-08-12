from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df_numeric = pd.read_csv('Numeric_engineered_dataset.csv')

# Define features and target
X_numeric = df_numeric.drop('Phishing attack', axis=1)
y_numeric = df_numeric['Phishing attack']

# Split the data into training and test sets
X_train_num, X_test_num, y_train_num, y_test_num = train_test_split(X_numeric, y_numeric, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
rf_numeric = RandomForestClassifier(n_estimators=100, random_state=42)
rf_numeric.fit(X_train_num, y_train_num)

# Predict on the test set
y_pred_num = rf_numeric.predict(X_test_num)

# Evaluate the model
print("Numeric Dataset - Random Forest Performance")
print(classification_report(y_test_num, y_pred_num))
print("Accuracy:", accuracy_score(y_test_num, y_pred_num))
