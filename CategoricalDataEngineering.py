import pandas as pd

# Load your dataset
df = pd.read_csv('cleaned_dataset.csv')

# Use pandas get_dummies for one-hot encoding
categorical_vars = ['Having IP Address', 'HTTPS Token', 'URL Shortening', 'SSL Final State', 'Domain Registration Country',
                    'Email in URL', 'Age of Domain', 'Submitting to Email', 'Server Form Handler (SFH)',
                    'HTTPS in URL', 'Favicon Hosting', 'Redirecting "//"', '@ Symbol']
df = pd.get_dummies(df, columns=categorical_vars)

# Save the transformed dataset
df.to_csv('Categorical_engineered_dataset.csv', index=False)
