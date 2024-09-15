```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(768, 256)
        self.dropout1 = nn.Dropout(0.3)  # Add dropout layer
        self.layer2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)  # Add dropout layer
        self.layer3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.dropout1(x)  # Apply dropout
        x = self.layer2(x)
        x = F.relu(x)
        x = self.dropout2(x)  # Apply dropout
        x = self.layer3(x)
        return x