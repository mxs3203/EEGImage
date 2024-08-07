import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler

import wandb
from LSTMModel import LSTMModel

if torch.cuda.is_available():
    print("CUDA is available! PyTorch is using GPU acceleration.")
    device = "cuda:1"
else:
    print("CUDA is not available. PyTorch is using CPU.")
    device = "cpu"


with open("data/forLSTM/X.pck", 'rb') as f:
    X = pickle.load(f)
with open("data/forLSTM/Y.pck", 'rb') as f:
    Y = pickle.load(f)

print(np.shape(X), np.shape(Y))



# Define hyperparameters
input_size = 32  # Number of features (channels)
hidden_size = 128  # Number of LSTM units
num_layers = 4 # Number of LSTM layers
num_classes = 10  # Number of unique labels
batch_size = 128
learning_rate = 0.0001
num_epochs = 30

# Create LSTM model
model = LSTMModel(input_size, hidden_size, num_layers, num_classes, device, contrastive=False).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.01, verbose=True)

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.3)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
# Data loader
train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_data = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)


wandb.init(
    # set the wandb project where this run will be logged
    project="EEGImage",
    name="PredictNumber-Visual-LSTM",

    config={
        "learning_rate": learning_rate,
        "architecture": "LSTM",
        "dataset": "EEG",
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers
    }
)


# Training loop
total_steps = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    train_losses = []
    val_losses = []
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_losses.append(loss.item())
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if (i + 1) % 100 == 0:
        #    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}')

    # Validation
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())
            preds = F.softmax(outputs, dim=1).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation F1 Score: {f1:.4f}')
        scheduler.step(np.mean(val_losses))
    wandb.log({"Valid/F1": f1, "Train/Loss": np.mean(train_losses), "Valid/Loss": np.mean(val_losses)})
wandb.finish()
torch.save(model.state_dict(), 'lstm_model.pth')
print('Training finished.')