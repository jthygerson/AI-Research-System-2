import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)  # Increased the number of neurons
        self.fc2 = nn.Linear(256, 128)  # Increased the number of neurons
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)  # Added an extra layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, p=0.5, train=True)  # Added dropout
        x = torch.relu(self.fc2(x))
        x = torch.dropout(x, p=0.5, train=True)  # Added dropout
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = NeuralNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Changed optimizer and learning rate

# Training loop (simplified)
for epoch in range(10):  # Increased number of epochs
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()