import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch

# Load the text dataset
df = pd.read_csv('Text_engineered_dataset.csv')

# Tokenization and encoding the dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_data = tokenizer(df['URL'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")

# Prepare few-shot dataset
few_shot_size = 20  # Assuming a very small dataset for few-shot learning
X_train, X_test, y_train, y_test = train_test_split(encoded_data['input_ids'], df['Phishing attack'], train_size=few_shot_size, random_state=42, stratify=df['Phishing attack'])

# Convert labels to tensors
y_train = torch.tensor(y_train.values)
y_test = torch.tensor(y_test.values)

# Create Tensor datasets
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

# DataLoader
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

# Load BERT for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.train()

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training Loop
for epoch in range(3):  # Limited epochs due to few-shot nature
    total_loss = 0
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

# Evaluate the model
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        outputs = model(inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print(f'Accuracy: {100 * correct / total}%')
