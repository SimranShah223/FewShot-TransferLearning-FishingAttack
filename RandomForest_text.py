import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df_text = pd.read_csv('Text_engineered_dataset.csv')

# Drop the 'URL' column and define features and target
X_text = df_text.drop(['Phishing attack', 'URL'], axis=1)
y_text = df_text['Phishing attack']

# Split the data into training and test sets
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier with class_weight='balanced'
rf_text = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the model
rf_text.fit(X_train_text, y_train_text)

# Make predictions on the test set
y_pred_text = rf_text.predict(X_test_text)

# Evaluate the model
print("Text Dataset - Random Forest Performance")
print(classification_report(y_test_text, y_pred_text, zero_division=1))
print("Accuracy:", accuracy_score(y_test_text, y_pred_text))
