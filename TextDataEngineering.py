import pandas as pd

# Load your dataset
df = pd.read_csv('cleaned_dataset.csv')
# Basic text features
df['url_length'] = df['URL'].apply(len)
df['url_digits'] = df['URL'].apply(lambda x: sum(c.isdigit() for c in x))
df['url_is_ip'] = df['URL'].apply(lambda x: int(x.replace('.', '').isdigit()))
df['url_special_chars'] = df['URL'].apply(lambda x: sum(not c.isalnum() for c in x))

df.to_csv('Text_engineered_dataset.csv', index=False)
