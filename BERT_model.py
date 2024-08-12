import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report

# Load the text dataset
df_text = pd.read_csv('Text_engineered_dataset.csv')

# Define features (URL) and target
X_text = df_text['URL'].values
y_text = df_text['Phishing attack'].values

# Split the data into training and test sets
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.2, random_state=42)

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize and encode the training and test data
encoded_train_text = tokenizer(X_train_text.tolist(), padding=True, truncation=True, return_tensors='pt')
encoded_test_text = tokenizer(X_test_text.tolist(), padding=True, truncation=True, return_tensors='pt')

# Create PyTorch DataLoader for training and test data
train_data = TensorDataset(encoded_train_text['input_ids'], encoded_train_text['attention_mask'], torch.tensor(y_train_text))
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=16)

test_data = TensorDataset(encoded_test_text['input_ids'], encoded_test_text['attention_mask'], torch.tensor(y_test_text))
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=16)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Fine-tune the BERT model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss:.4f}')

# Evaluate the fine-tuned model on the test set
model.eval()
predictions, true_labels = [], []

for batch in test_dataloader:
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    labels = batch[2]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    predictions.extend(torch.argmax(logits, dim=1).tolist())
    true_labels.extend(labels.tolist())

print(classification_report(true_labels, predictions))
