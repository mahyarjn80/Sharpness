import torch
from torchvision import datasets, transforms

# Define the data folder
DATASETS_FOLDER = "./data"

# Load CIFAR-10 training dataset without transforms
cifar10_train_raw = datasets.CIFAR10(root=DATASETS_FOLDER, train=True, download=True)

# Load CIFAR-10 training dataset with transforms (e.g., for ViT)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 for ViT
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet norms
])
cifar10_train_transformed = datasets.CIFAR10(root=DATASETS_FOLDER, train=True, download=True, transform=transform)
aa = torch.stack([x for x, _ in cifar10_train_transformed])





# Check dimensions and data type of raw and transformed datasets
# Raw dataset (without transform)
# print(aa.shape)
sample_raw= aa.tensors[0] # Access first image and label tuple, get image
print(f"Raw CIFAR-10 train sample shape (without transform): {sample_raw.shape}")
print(f"Total samples in raw dataset: {len(cifar10_train_raw)}")

# Transformed dataset
sample_transformed, _ = cifar10_train_transformed  # Access first transformed image
print(f"Transformed CIFAR-10 train sample shape (with transform): {sample_transformed.shape}")
print(f"Transformed CIFAR-10 train data type: {type(sample_transformed)}")
print(f"Total samples in transformed dataset: {len(cifar10_train_transformed)}")