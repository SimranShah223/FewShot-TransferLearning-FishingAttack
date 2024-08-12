from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

# Load the numeric dataset
df_cat = pd.read_csv('Text_engineered_dataset.csv')

# Define features and target
X_numeric = df_cat.drop(['Phishing attack', 'URL'], axis=1)
y_numeric = df_cat['Phishing attack']

# Split the data into training and test sets
X_train_num, X_test_num, y_train_num, y_test_num = train_test_split(X_numeric, y_numeric, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

# Initialize and train the Logistic Regression model
lr_numeric = LogisticRegression(max_iter=1000, random_state=42)
lr_numeric.fit(X_train_num_scaled, y_train_num)

# Predict on the test set
y_pred_num = lr_numeric.predict(X_test_num_scaled)

# Evaluate the model
print("Numeric Dataset - Logistic Regression Performance")
print(classification_report(y_test_num, y_pred_num))
print("Accuracy:", accuracy_score(y_test_num, y_pred_num))

# Function to plot ROC Curve
def plot_roc_curve(y_test, y_scores, title):
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {title}')
    plt.legend(loc="lower right")
    plt.show()

# Plot ROC Curve
plot_roc_curve(y_test_num, y_pred_num, 'Categorical Data - Logistic Regression')
