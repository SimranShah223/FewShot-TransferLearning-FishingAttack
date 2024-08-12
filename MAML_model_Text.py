import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load dataset
df = pd.read_csv('/path/to/your/Text_engineered_dataset.csv')

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['URL'])
X = tokenizer.texts_to_sequences(df['URL'])
X = pad_sequences(X, maxlen=50)  # Pad sequences to a fixed length

# Labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Phishing attack'])

# Select a small number for few-shot learning
few_shot_size = 50  # Small number of examples for training
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=few_shot_size, random_state=42, stratify=y)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create data loaders
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, target_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc(self.relu(lstm_out[:, -1, :]))
        return x

# Instantiate the model
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because index starts from 0
model = LSTMModel(embedding_dim=32, hidden_dim=64, vocab_size=vocab_size, target_size=2)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(50):  # Few epochs due to few-shot nature
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

model.eval()
total = 0
correct = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
