import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from itertools import combinations
import torch.optim as optim
import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

import tensorflow as tf
from torchvision import datasets, transforms


import random
import time

torch.manual_seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_value = 12345
np.random.seed(seed_value)
random.seed(seed_value)



# MNIST autoencoder
class TripletAutoencoder(nn.Module):
    def __init__(self, input_dim=784, bottleneck_dim=8):
        super(TripletAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return encoded, reconstructed

def generate_triplets(labels):
    triplets = []
    labels = labels.cpu().numpy()

    for label in set(labels):
        pos_idx = [i for i in range(len(labels)) if labels[i] == label]
        neg_idx = [i for i in range(len(labels)) if labels[i] != label]
        if len(pos_idx) < 2:
            continue
        for anchor, positive in combinations(pos_idx, 2):
            negative = random.choice(neg_idx)
            triplets.append((anchor, positive, negative))
    return triplets

def extract_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        encoded = model.encoder(data_tensor)
        return encoded.cpu().numpy()

# feature map
def encoding_features_h_ry(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits)

    for i in range(num_qubits):
        qc.h(i)
        qc.ry(feature_params[i], i)

    return qc

def data_load_and_process_mnist(
        num_classes,
        all_samples,
        seed,
        num_examples_per_class,
        pca=True,
        n_features=8,
        epochs = 300,
        margin=.2,
        alpha=1.,
        type_model='linear'
):
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    x_train = mnist_train.data.numpy().astype(np.float32) / 255.0
    y_train = mnist_train.targets.numpy()

    x_test = mnist_test.data.numpy().astype(np.float32) / 255.0
    y_test = mnist_test.targets.numpy()

    if type_model != 'linear':
        x_train = np.expand_dims(x_train, 1)
        x_test = np.expand_dims(x_test, 1)
    else:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    if not all_samples:
        selected_indices = []

        for class_label in range(10):
            indices = np.where(y_train == class_label)[0][:num_examples_per_class]
            selected_indices.extend(indices)

        x_train_subset = x_train[selected_indices]
        y_train_subset = y_train[selected_indices]

        shuffle_indices = np.random.permutation(len(x_train_subset))
        x_train = x_train_subset[shuffle_indices]
        y_train = y_train_subset[shuffle_indices]

    logger.info("Shape of subset training data: {}", x_train.shape)
    logger.info("Shape of subset training labels: {}", y_train.shape)

    mask_train = np.isin(y_train, range(0, num_classes))
    mask_test = np.isin(y_test, range(0, num_classes))

    X_train = x_train[mask_train].reshape(-1, 784)
    X_test = x_test[mask_test].reshape(-1, 784)


    Y_train = y_train[mask_train]
    Y_test = y_test[mask_test]

    logger.info("Shape of subset training data: {}", X_train.shape)
    logger.info("Shape of subset training labels: {}", Y_train.shape)
    logger.info("Shape of testing data: {}", X_test.shape)
    logger.info("Shape of testing labels: {}", Y_test.shape)

    X_train = (X_train - X_train.min()) * (np.pi / (X_train.max() - X_train.min()))
    X_test = (X_test - X_test.min()) * (np.pi / (X_test.max() - X_test.min()))

    return X_train, X_test, Y_train, Y_test

