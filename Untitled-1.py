#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import grad
from pathlib import Path
import json
import time
# Force CPU usage
device = torch.device("cpu")
print(f"Using device: {device}")

# Adjusted hyperparameters for CPU training
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 4e-2  # Fixed learning rate
PATCH_SIZE = 4
NUM_CLASSES = 10
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 4
MLP_DIM = 256 * 4
DROPOUT = 0.1

# [Previous PatchEmbedding and ViT class definitions remain the same]
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, E, H', W')
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x

class ViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        mlp_dim=256*4,
        dropout=0.1,
    ):
        super().__init__()
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        
        # Add position embedding
        n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize patch embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.patch_embed.proj.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.proj.bias)

    def forward(self, x):
        # Create patches
        x = self.patch_embed(x)
        
        # Add cls token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply transformer
        x = self.transformer(x)
        
        # Get cls token
        x = x[:, 0]
        
        # MLP head
        x = self.mlp_head(x)
        return x

# # [Previous training and testing functions remain the same]
# def train_epoch(model, trainloader, criterion, optimizer, device, epoch):
#     global current_step
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
    
#     for i, data in enumerate(trainloader, 0):
#         current_step += 1
#         inputs, labels = data[0].to(device), data[1].to(device)
        
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         step_train_losses.append(loss.item())
#         step_numbers.append(current_step)
        
#         running_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()
        
#         if i % 50 == 49:
#             avg_loss = running_loss / 50
#             accuracy = 100. * correct / total
#             print(f'Epoch: {epoch+1}, Batch: {i+1}, Step: {current_step}, Loss: {avg_loss:.3f}, Acc: {accuracy:.2f}%')
#             running_loss = 0.0
#             plot_training_progress()
    
#     return running_loss/len(trainloader), 100.*correct/total



def test_epoch(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return test_loss/len(testloader), 100.*correct/total

# # [Previous plotting functions remain the same]
# def plot_training_progress():
#     plt.figure(figsize=(12, 4))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(step_numbers, step_train_losses, 'b-', alpha=0.3, label='Step Loss')
    
#     window_size = len(step_train_losses)
#     if window_size > 0:
#         cumulative_means = np.cumsum(step_train_losses) / np.arange(1, window_size + 1)
#         plt.plot(step_numbers, cumulative_means, 'r-', 
#                 label='Mean Loss (to current step)')
    
#     plt.xlabel('Optimization Step')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title('Training Loss per Step')
#     plt.grid(True, alpha=0.3)
    
#     plt.subplot(1, 2, 2)
#     plt.hist(step_train_losses, bins=50, alpha=0.75, color='blue')
#     plt.xlabel('Loss Value')
#     plt.ylabel('Frequency')
#     plt.title('Loss Distribution')
    
#     plt.tight_layout()
#     Path('results').mkdir(exist_ok=True)
#     plt.savefig('results/training_progress.png')
#     plt.close()

# Initialize lists to store metrics
step_train_losses = []
step_numbers = []
current_step = 0
step_top_eigenvalues = []

# New function to compute top 3 eigenvalues using power iteration
def compute_top_eigenvalues(model, inputs, labels, criterion, num_iterations=10):
    """
    Approximates top 3 eigenvalues of the Hessian using power iteration
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    
    # Compute loss and gradients
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    grads = grad(loss, params, create_graph=True)
    grad_vector = torch.cat([g.flatten() for g in grads])
    
    top_eigenvalues = []
    v = torch.randn(n_params, device=device)
    
    # Estimate top 3 eigenvalues
    for _ in range(3):
        v = v / torch.norm(v)
        
        # Power iteration
        for _ in range(num_iterations):
            # Compute Hessian-vector product
            Hv = torch.zeros_like(v)
            grad_grads = grad(grad_vector @ v, params, retain_graph=True)
            Hv = torch.cat([gg.flatten() for gg in grad_grads])
            v_new = Hv / torch.norm(Hv)
            v = v_new
        
        # Rayleigh quotient
        Hv = torch.zeros_like(v)
        grad_grads = grad(grad_vector @ v, params, retain_graph=True)
        Hv = torch.cat([gg.flatten() for gg in grad_grads])
        eigenvalue = (v @ Hv) / (v @ v)
        top_eigenvalues.append(eigenvalue.item())
        
        # Deflation: remove component in direction of current eigenvector
        grad_vector = grad_vector - (grad_vector @ v) * v
    
    return top_eigenvalues

# Modified train_epoch function
def train_epoch(model, trainloader, criterion, optimizer, device, epoch):
    global current_step
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(trainloader, 0):
        current_step += 1
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Compute top 3 eigenvalues (every 50 batches to save computation)
        if i % 50 == 0:
            start_time = time.time()
            top_eigs = compute_top_eigenvalues(model, inputs, labels, criterion)
            step_top_eigenvalues.append({
                'step': current_step,
                'eigenvalues': top_eigs,
                'time': time.time() - start_time
            })
        
        loss.backward()
        optimizer.step()
        
        step_train_losses.append(loss.item())
        step_numbers.append(current_step)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if i % 50 == 49:
            avg_loss = running_loss / 50
            accuracy = 100. * correct / total
            print(f'Epoch: {epoch+1}, Batch: {i+1}, Step: {current_step}, Loss: {avg_loss:.3f}, Acc: {accuracy:.2f}%')
            running_loss = 0.0
            plot_training_progress()
    
    return running_loss/len(trainloader), 100.*correct/total

# Modified plot_training_progress function to include eigenvalue plotting
def plot_training_progress():
    plt.figure(figsize=(15, 8))
    
    # Original loss plot
    plt.subplot(2, 2, 1)
    plt.plot(step_numbers, step_train_losses, 'b-', alpha=0.3, label='Step Loss')
    window_size = len(step_train_losses)
    if window_size > 0:
        cumulative_means = np.cumsum(step_train_losses) / np.arange(1, window_size + 1)
        plt.plot(step_numbers, cumulative_means, 'r-', 
                label='Mean Loss')
    plt.xlabel('Optimization Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss per Step')
    plt.grid(True, alpha=0.3)
    
    # Loss distribution
    plt.subplot(2, 2, 2)
    plt.hist(step_train_losses, bins=50, alpha=0.75, color='blue')
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    plt.title('Loss Distribution')
    
    # Eigenvalue plot
    if step_top_eigenvalues:
        steps = [e['step'] for e in step_top_eigenvalues]
        eig1 = [e['eigenvalues'][0] for e in step_top_eigenvalues]
        eig2 = [e['eigenvalues'][1] for e in step_top_eigenvalues]
        eig3 = [e['eigenvalues'][2] for e in step_top_eigenvalues]
        
        plt.subplot(2, 2, 3)
        plt.plot(steps, eig1, 'r-', label='λ1')
        plt.plot(steps, eig2, 'g-', label='λ2')
        plt.plot(steps, eig3, 'b-', label='λ3')
        plt.xlabel('Optimization Step')
        plt.ylabel('Eigenvalue')
        plt.legend()
        plt.title('Top 3 Hessian Eigenvalues')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/training_progress.png')
    plt.close()

# Data transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                       shuffle=False, num_workers=0)

# Create results directory
Path('results').mkdir(exist_ok=True)
Path('results/checkpoints').mkdir(parents=True, exist_ok=True)

# Initialize model, criterion, and optimizer
model = ViT(
    img_size=32,
    patch_size=PATCH_SIZE,
    num_classes=NUM_CLASSES,
    embed_dim=EMBED_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    mlp_dim=MLP_DIM,
    dropout=DROPOUT,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)  # Fixed learning rate

# Training loop
print("Starting training...")
epoch_train_losses = []
epoch_train_accuracies = []
epoch_test_losses = []
epoch_test_accuracies = []

for epoch in range(NUM_EPOCHS):
    print(f'\nEpoch: {epoch+1}/{NUM_EPOCHS}')
    
    train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device, epoch)
    test_loss, test_acc = test_epoch(model, testloader, criterion, device)
    
    epoch_train_losses.append(train_loss)
    epoch_train_accuracies.append(train_acc)
    epoch_test_losses.append(test_loss)
    epoch_test_accuracies.append(test_acc)
    
    print(f'Epoch Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')
    print(f'Epoch Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
        'step_train_losses': step_train_losses,
        'step_numbers': step_numbers
    }
    torch.save(checkpoint, f'results/checkpoints/epoch_{epoch+1}.pth')

# Save final results
results = {
    'epoch_train_losses': epoch_train_losses,
    'epoch_train_accuracies': epoch_train_accuracies,
    'epoch_test_losses': epoch_test_losses,
    'epoch_test_accuracies': epoch_test_accuracies,
    'step_train_losses': step_train_losses,
    'step_numbers': step_numbers
}

with open('results/training_metrics.json', 'w') as f:
    json.dump(results, f)

torch.save(model.state_dict(), 'results/vit_cifar10.pth')

# Final plots
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epoch_train_losses, label='Train Loss')
plt.plot(epoch_test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss per Epoch')

plt.subplot(1, 2, 2)
plt.plot(epoch_train_accuracies, label='Train Accuracy')
plt.plot(epoch_test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Test Accuracy per Epoch')

plt.tight_layout()
plt.savefig('results/epoch_metrics.png')
plt.close()

plot_training_progress()

print("\nTraining completed!")
print(f"Final Test Accuracy: {epoch_test_accuracies[-1]:.2f}%")
print("\nResults saved in 'results' directory:")
print("- Model weights: results/vit_cifar10.pth")
print("- Training metrics: results/training_metrics.json")
print("- Step-wise loss plot: results/training_progress.png")
print("- Epoch metrics plot: results/epoch_metrics.png")
print("- Checkpoints: results/checkpoints/")


# In[ ]:




