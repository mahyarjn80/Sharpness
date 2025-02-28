"""
Simple script to test if torch.nn.GELU activation works correctly
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Create a GELU activation function
gelu = nn.GELU()

# Create input tensor with values from -5 to 5
x = torch.linspace(-5, 5, 100)

# Apply GELU activation
y = gelu(x)

# Plot the activation function
plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), y.numpy(), label='GELU')
plt.plot(x.numpy(), torch.relu(x).numpy(), '--', label='ReLU (for comparison)')
plt.grid(True)
plt.title('GELU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

# Test with a small network
class TestNetwork(nn.Module):
    def __init__(self):
        super(TestNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

# Create a test network
model = TestNetwork()
print("Model architecture:", model)

# Test forward pass
test_input = torch.randn(5, 10)
output = model(test_input)
print("Input shape:", test_input.shape)
print("Output shape:", output.shape)
print("Forward pass successful!")

plt.show()