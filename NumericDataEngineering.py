import pandas as pd

# Load your dataset
df = pd.read_csv('cleaned_dataset.csv')

# Create new interaction features
df['Age_Link_Meta'] = df['Domain Age'] * df['Links in Meta/Script/Link']
df['RegLen_PageRank'] = df['Domain Registration Length'] * df['PageRank']
df['URLLen_ContentLen'] = df['URL Length'] * df['Content Length']
df['Subdomain_URLLen'] = df['Subdomains'] * df['URL Length']
df['RespTime_Link_Meta'] = df['Response Time'] * df['Links in Meta/Script/Link']
df['RespTime_Traffic'] = df['Response Time'] * df['Website Traffic']
df['Subdomain_Age'] = df['Subdomains'] * df['Domain Age']

# Save the transformed dataset
df.to_csv('Numeric_engineered_dataset.csv', index=False)
