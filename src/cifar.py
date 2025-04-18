import numpy as np
from torchvision.datasets import CIFAR10
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
from torchvision import transforms
import os
import torch
from torch import Tensor
import torch.nn.functional as F

DATASETS_FOLDER = os.environ["DATASETS"]

def center(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(0)
    return X_train - mean, X_test - mean

def standardize(X_train: np.ndarray, X_test: np.ndarray):
    std = X_train.std(0)
    return (X_train / std, X_test / std)

def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)

def unflatten(arr: np.ndarray, shape: Tuple):
    return arr.reshape(arr.shape[0], *shape)

def _one_hot(tensor: Tensor, num_classes: int, default=0):
    M = F.one_hot(tensor, num_classes)
    M[M == 0] = default
    return M.float()

def make_labels(y, loss):
    if loss == "ce":
        return y
    elif loss == "mse":
        return _one_hot(y, 10, 0)


def load_cifar(loss: str) -> (TensorDataset, TensorDataset):
    cifar10_train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True)
    cifar10_test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), \
        make_labels(torch.tensor(cifar10_test.targets), loss)
    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_train)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_test)
    return train, test

def load_cifar_vit(loss: str) -> (TensorDataset, TensorDataset):
    # For ViT, we need to resize images to 224x224 and use standard normalization
    
    # No need for ToPILImage since CIFAR10 dataset will return PIL images by default
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    cifar10_train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True, transform=transform)
    cifar10_test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False, transform=transform)


    y_train = make_labels(torch.tensor(cifar10_train.targets), loss)
    y_test = make_labels(torch.tensor(cifar10_test.targets), loss)


    # Create datasets with transformed images and labels
    train = TensorDataset(torch.stack([x for x, _ in cifar10_train]).float(), y_train)
    test = TensorDataset(torch.stack([x for x, _ in cifar10_test]).float(), y_test)


    return train, test



