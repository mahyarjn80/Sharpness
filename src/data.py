import torch
import numpy as np
from typing import Tuple
from torch.utils.data import TensorDataset
from cifar import load_cifar, load_cifar_vit
from synthetic import make_chebyshev_dataset, make_linear_dataset
# from wikitext import load_wikitext_2

DATASETS = [
    "cifar10", "cifar10-1k", "cifar10-2k", "cifar10-5k", "cifar10-10k", "cifar10-20k", "chebyshev-3-20",
    "chebyshev-4-20", "chebyshev-5-20", "linear-50-50", "cifar10-vit", "cifar10-vit-1k", "cifar10-vit-2k", "cifar10-vit-5k", "cifar10-vit-10k", "cifar10-vit-20k"
]

def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)

def unflatten(arr: np.ndarray, shape: Tuple):
    return arr.reshape(arr.shape[0], *shape)

def num_input_channels(dataset_name: str) -> int:
    if dataset_name.startswith("cifar10"):
        return 3
    elif dataset_name == 'fashion':
        return 1

def image_size(dataset_name: str) -> int:
    if dataset_name.startswith("cifar10"):
        return 32
    elif dataset_name == 'fashion':
        return 28

def num_classes(dataset_name: str) -> int:
    if dataset_name.startswith('cifar10'):
        return 10
    elif dataset_name == 'fashion':
        return 10

def get_pooling(pooling: str):
    if pooling == 'max':
        return torch.nn.MaxPool2d((2, 2))
    elif pooling == 'average':
        return torch.nn.AvgPool2d((2, 2))
    else:
        raise NotImplementedError("unknown pooling: {}".format(pooling))

def num_pixels(dataset_name: str) -> int:
    return num_input_channels(dataset_name) * image_size(dataset_name)**2

def take_first(dataset: TensorDataset, num_to_keep: int) -> TensorDataset:
    """Takes the first num_to_keep examples from a dataset to create a smaller subset.
    This is used to create reduced versions of CIFAR-10 with fewer training examples (1k, 2k, 5k, etc)
    to study how model performance scales with dataset size."""
    return TensorDataset(dataset.tensors[0][:num_to_keep], dataset.tensors[1][:num_to_keep])

def load_dataset(dataset_name: str, loss: str) -> Tuple[TensorDataset, TensorDataset]:
    if dataset_name == "cifar10":
        return load_cifar(loss)
    elif dataset_name == "cifar10-1k":
        train, test = load_cifar(loss)
        return take_first(train, 1000), test  # Use only first 1000 training examples
    elif dataset_name == "cifar10-2k":
        train, test = load_cifar(loss)
        return take_first(train, 2000), test  # Use only first 2000 training examples
    elif dataset_name == "cifar10-5k":
        train, test = load_cifar(loss)
        return take_first(train, 5000), test  # Use only first 5000 training examples
    elif dataset_name == "cifar10-10k":
        train, test = load_cifar(loss)
        return take_first(train, 10000), test  # Use only first 10000 training examples
    elif dataset_name == "cifar10-20k":
        train, test = load_cifar(loss)
        return take_first(train, 20000), test  # Use only first 20000 training examples
    elif dataset_name == "cifar10-vit":
        return load_cifar_vit(loss)
    elif dataset_name == "cifar10-vit-1k":
        train, test = load_cifar_vit(loss)
        return take_first(train, 1000), test  # Use only first 1000 training examples
    elif dataset_name == "cifar10-vit-2k":
        train, test = load_cifar_vit(loss)
        return take_first(train, 2000), test  # Use only first 2000 training examples
    elif dataset_name == "cifar10-vit-5k":
        train, test = load_cifar_vit(loss)
        return take_first(train, 5000), test  # Use only first 5000 training examples
    elif dataset_name == "cifar10-vit-10k":
        train, test = load_cifar_vit(loss)
        return take_first(train, 10000), test 
    elif dataset_name == "cifar10-vit-20k":
        train, test = load_cifar_vit(loss)
        return take_first(train, 20000), test  # Use only first 20000 training examples
    elif dataset_name == "chebyshev-5-20":
        return make_chebyshev_dataset(k=5, n=20)
    elif dataset_name == "chebyshev-4-20":
        return make_chebyshev_dataset(k=4, n=20)
    elif dataset_name == "chebyshev-3-20":
        return make_chebyshev_dataset(k=3, n=20)
    elif dataset_name == 'linear-50-50':
        return make_linear_dataset(n=50, d=50)
