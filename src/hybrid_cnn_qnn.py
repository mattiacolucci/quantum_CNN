"""
Hybrid CNN-Quantum Neural Network Classifier for MNIST
Project 10: Quantum Classifiers with Data Re-uploading on MNIST

Architecture:
1. Classical CNN extracts features from MNIST images
2. CNN outputs 4 features
3. VQC (Variational Quantum Circuit) with data re-uploading processes these 4 features
4. Quantum circuit is measured for classification
5. Final classification is done classically based on quantum measurements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import logging
import pickle
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter threading issues
import matplotlib.pyplot as plt
import argparse
import dagshub
import mlflow
from tqdm import tqdm
from sklearn.model_selection import KFold


from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.quantum_info import SparsePauliOp
import json
import os
from datetime import datetime
from scipy.stats import entropy
from scipy.special import comb

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.feature_maps import (
    standard_reuploading_feature_map,
    partial_reuploading_feature_map,
    partial_encoding_RA_feature_map
)
from src.utils.ansatz import construct_tensor_ring_ansatz_circuit, tensor_ring

logger = logging.getLogger(__name__)

dagshub.init(repo_owner='mattiacolucci', repo_name='quantum_CNN', mlflow=True)


class CNNFeatureExtractor(nn.Module):
    """
    Classical CNN that extracts 4 features from MNIST images (28x28).
    These 4 features will be fed to the quantum circuit.
    """
    def __init__(self):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 7x7 -> 7x7
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)
        
        # Calculate size after convolutions: 28 -> 14 -> 7 -> 3 (after 3 pools)
        # But we'll do 2 pools: 28 -> 14 -> 7
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 4)  # Output 4 features for quantum circuit
        
    def forward(self, x):
        # Conv blocks
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # 28 -> 14
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 14 -> 7
        
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)  # Flatten: (batch, 128, 7, 7) -> (batch, 128*7*7)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output 4 features
        
        # Normalize to [0, π] range for quantum encoding
        x = torch.sigmoid(x) * np.pi
        
        return x


def create_quantum_circuit(feature_map_type='standard', num_qubits=4, reps=2, ansatz_type='tensor_ring'):
    """
    Create the quantum neural network with data re-uploading and repetitions.
    
    Args:
        feature_map_type (str): Type of feature map:
            - 'standard': Standard re-uploading (Ry + Rx)
            - 'partial': Partial re-uploading (sequential encoding)
            - 'partial_RA': Partial encoding with only first 2 qubits (Rx + Rz per qubit)
        num_qubits (int): Number of qubits (must be 4)
        reps (int): Number of [feature_map → ansatz] repetitions
        ansatz_type (str): Type of ansatz ('tensor_ring' or 'mps_ttn_combined')
    
    Returns:
        TorchConnector wrapped quantum circuit, circuit with measurements
    """
    logger.info(f"Creating {feature_map_type} re-uploading quantum circuit with {reps} repetitions")
    
    # Build complete circuit with repetitions
    full_circuit = QuantumCircuit(num_qubits)

    if feature_map_type=='standard' or feature_map_type=='partial':
        feature_params = ParameterVector('x', num_qubits)
    elif feature_map_type=='partial_RA':
        feature_params = ParameterVector('x', num_qubits*3)
    
    # Variational parameters (different for each repetition)
    all_var_params = []
    
    for rep in range(reps):
        # 1. Feature map (data re-uploading) - SAME features each time
        if feature_map_type == 'standard':
            fm = standard_reuploading_feature_map(num_qubits=num_qubits, hadamard_init=(True if rep == 0 else False), squared_transform=(True if rep > 0 else False))
        elif feature_map_type == 'partial':
            fm = partial_reuploading_feature_map(num_qubits=num_qubits, hadamard_init=(True if rep == 0 else False), squared_transform=(True if rep > 0 else False))
        elif feature_map_type == 'partial_RA':
            fm = partial_encoding_RA_feature_map(num_qubits=num_qubits, hadamard_init=(True if rep == 0 else False), squared_transform=(True if rep > 0 else False))
        else:
            raise ValueError(f"Unknown feature map type: {feature_map_type}")
        
        # Separate feature parameters (x) from variational parameters (w_L*) in feature map
        fm_feature_params = [p for p in fm.parameters if p.name.startswith('x[')]
        fm_var_params = [p for p in fm.parameters if not p.name.startswith('x[')]
        
        # Bind feature parameters (reuse same x₀, x₁, x₂, x₃ or x₀...x₁₁)
        param_dict = {fm_feature_params[i]: feature_params[i] for i in range(len(fm_feature_params))}
        
        # If feature map has variational parameters (e.g., from RealAmplitudes), handle them
        if fm_var_params:
            # Create unique parameters for this repetition's feature map variational params
            fm_var_params_vec = ParameterVector(f'θ_fm{rep}', len(fm_var_params))
            all_var_params.extend(fm_var_params_vec)
            # Add to param_dict
            for i, p in enumerate(fm_var_params):
                param_dict[p] = fm_var_params_vec[i]
        
        fm_bound = fm.assign_parameters(param_dict)
        
        # Compose the feature map circuit directly (don't convert to gate)
        full_circuit.compose(fm_bound, inplace=True)

        # Add barrier between feature map and VQC block for clarity
        full_circuit.barrier()
        
        # 2. Ansatz (variational circuit) - DIFFERENT parameters each time
        if ansatz_type == 'tensor_ring':
            ansatz = tensor_ring(num_qubits, reps=1).decompose()
        elif ansatz_type == 'mps_ttn_combined':
            ansatz = construct_tensor_ring_ansatz_circuit(num_qubits).decompose()
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")
        
        # Create unique parameters for this repetition
        var_params = ParameterVector(f'θ_rep{rep}', ansatz.num_parameters)
        all_var_params.extend(var_params)
        
        param_dict = {ansatz.parameters[i]: var_params[i] for i in range(len(ansatz.parameters))}
        ansatz_bound = ansatz.assign_parameters(param_dict)
        
        # Compose the ansatz circuit directly (don't convert to gate)
        full_circuit.compose(ansatz_bound, inplace=True)
        
        # Add barrier between repetitions for clarity
        if rep < reps - 1:
            full_circuit.barrier()
    
    logger.info(f"Circuit built: {num_qubits} qubits, {reps} repetitions")
    logger.info(f"Parameters: {len(feature_params)} feature params, {len(all_var_params)} variational params")

    # Create a copy of the circuit with measurements for visualization
    circuit_with_measurements = full_circuit.copy()
    circuit_with_measurements.measure_all()  # Add measurements to all qubits
    
    # Create SamplerV2 primitive with options
    sampler = SamplerV2()
    sampler.options.default_shots = 4096
    sampler.options.seed_simulator = 12345
    
    # Create SamplerQNN with the complete circuit
    # SamplerQNN returns probability distribution over all 2^n computational basis states
    # For 4 qubits: 16 output values (probabilities for |0000⟩, |0001⟩, ..., |1111⟩)
    # Specify which parameters are inputs (features) and which are weights (variational)
    qnn = SamplerQNN(
        circuit=full_circuit,
        input_params=feature_params,
        weight_params=all_var_params,
        sampler=sampler,
        input_gradients=True
    )
    
    logger.info(f"SamplerQNN created: returns {2**num_qubits} output values (probability distribution)")
    
    # Wrap with TorchConnector
    torch_connector = TorchConnector(qnn)
    
    # Return both the connector and the original circuit for plotting
    return torch_connector, circuit_with_measurements


class HybridCNNQNN(nn.Module):
    """
    Hybrid CNN-Quantum Neural Network.
    
    Architecture:
    1. CNN extracts 4 features from images
    2. Quantum circuit processes these 4 features
    3. Quantum output is used for classification
    """
    def __init__(self, qnn, num_classes=4, num_qubits=4):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.qnn = qnn  # TorchConnector wrapped quantum circuit
        # SamplerQNN returns 2^n probabilities (16 for 4 qubits)
        self.fc_final = nn.Linear(2**num_qubits, num_classes)  # Map 16 quantum outputs to classes
        self.softmax = nn.Softmax(dim=1)
        
    def to(self, device):
        """Override to() to handle quantum circuit properly"""
        # Only move CNN and linear layer to device (QNN stays on CPU)
        self.cnn = self.cnn.to(device)
        self.fc_final = self.fc_final.to(device)
        return self
        
    def forward(self, x, apply_softmax=False):
        # Extract features with CNN (on GPU if available)
        cnn_device = next(self.cnn.parameters()).device
        x = self.cnn(x)  # (batch, 4)
        
        # Process with quantum circuit (requires CPU)
        # Move to CPU for quantum circuit, then back to original device
        x_cpu = x.cpu()
        x_qnn = self.qnn(x_cpu)  # (batch, 16) - SamplerQNN returns probability distribution
        x = x_qnn.to(cnn_device)  # Move back to GPU if needed
        
        # Projection to class logits (on GPU if available)
        x = self.fc_final(x)  # (batch, num_classes) - map 16 quantum probs to num_classes

        # Final probabilities
        # Apply softmax ONLY during inference if needed
        if apply_softmax:
            x = F.softmax(x, dim=1)
        
        return x

class CNNOnly(nn.Module):
    """
    Classical CNN only (no quantum circuit) for baseline comparison.
    """
    # Conv layers
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 7x7 -> 7x7
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)
        
        # Calculate size after convolutions: 28 -> 14 -> 7 -> 3 (after 3 pools)
        # But we'll do 2 pools: 28 -> 14 -> 7
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)  # Output one for each MNIST class
        
    def forward(self, x):
        # Conv blocks
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # 28 -> 14
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 14 -> 7
        
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)  # Flatten: (batch, 128, 7, 7) -> (batch, 128*7*7)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output num_classes features

        return x


def load_mnist_dataset(num_classes=4, samples_per_class=100, batch_size=32, device='cpu', seed=12345):
    """
    Load MNIST data and create PyTorch DataLoaders.
    
    Args:
        num_classes (int): Number of classes to use (4, 6, or 8)
        samples_per_class (int): Number of samples per class
        batch_size (int): Batch size for DataLoaders
        seed (int): Random seed
    
    Returns:
        train_loader, test_loader, validation_loader: DataLoaders for training, testing, and validation
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load MNIST
    logger.info(f"Loading MNIST dataset with {num_classes} classes...")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    mnist_train = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root="../data", train=False, download=True, transform=transform)

    X_train = mnist_train.data / 255.0  # Normalize to [0,1]
    y_train = mnist_train.targets

    X_test = mnist_test.data / 255.0
    y_test = mnist_test.targets

    # Split a validation set from test set data (50%)
    val_size = len(X_test) // 2
    X_val = X_test[:val_size]
    y_val = y_test[:val_size]

    X_test = X_test[val_size:]
    y_test = y_test[val_size:]

    # Filter by number of classes and sample
    selected_indices = []
    for class_label in range(num_classes):
        # Get all samples of this class
        indices = [i for i in range(len(y_train)) if y_train[i] == class_label]
        
        # Random sample
        selected = np.random.choice(indices, samples_per_class, replace=False)
        
        selected_indices.extend(selected)

    # Convert to tensors
    X_train_subset = X_train[selected_indices]
    y_train_subset = y_train[selected_indices]
    
    # Shuffle
    shuffle_idx = torch.randperm(len(X_train_subset))
    X_train = X_train_subset[shuffle_idx]
    y_train = y_train_subset[shuffle_idx]

    # Preserve only num_classes in test and validation sets
    test_indices = [i for i in range(len(y_test)) if y_test[i] < num_classes]
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]

    val_indices = [i for i in range(len(y_val)) if y_val[i] < num_classes]
    X_val = X_val[val_indices]
    y_val = y_val[val_indices]
    
    logger.info(f"Train dataset shape: {X_train.shape}, Labels shape: {y_train.shape}")
    logger.info(f"Test dataset shape: {X_test.shape}, Labels shape: {y_test.shape}")
    logger.info(f"Validation dataset shape: {X_val.shape}, Labels shape: {y_val.shape}")
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train.unsqueeze(1), y_train)
    test_dataset = TensorDataset(X_test.unsqueeze(1), y_test)
    validation_dataset = TensorDataset(X_val.unsqueeze(1), y_val)

    # Use pinned memory for faster CPU-GPU transfers (if using CUDA)
    use_pinned = (device == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            pin_memory=use_pinned, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            pin_memory=use_pinned, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                         pin_memory=use_pinned, num_workers=0)

    return train_loader, test_loader, validation_loader


def train_hybrid_model(
    model,
    train_loader,
    test_loader,
    validation_loader,
    early_stop=2,
    epochs=20,
    learning_rate=0.001,
    device='cpu',
    log_dir='logs',
):
    """
    Train the hybrid CNN-QNN model with comprehensive logging.
    
    Args:
        model: HybridCNNQNN model
        train_loader: Training data loader
        test_loader: Test data loader
        validation_loader: Validation data loader
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        device (str): 'cpu' or 'cuda'
        log_dir (str): Directory to save logs
        early_stop (int): Early stopping patience
    Returns:
        dict: Training history (loss, accuracy) and best results
    """
    model = model.to(device)
    model.train()
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'best_epoch': 0,
        'best_train': {},
        'best_test': {},
        'best_val': {'f1': 0.0},
        'epoch_times': [],
    }
    
    # Track best model
    patient = 0
    
    # Log training configuration
    config_log = {
        'device': device,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': train_loader.batch_size,
        'train_size': len(train_loader.dataset),
        'test_size': len(test_loader.dataset),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    logger.info(f"Configuration: {json.dumps(config_log, indent=2)}")
    
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        epoch_start = time.time()
        
        # Training
        model.train()
        total_loss = 0
        preds = []
        targets = []
        batch_losses = []
        
        for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training Batches"):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            batch_losses.append(loss.item())
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
        
        # Epoch statistics
        model.eval()
        avg_loss = total_loss / len(train_loader)
        train_metrics = compute_metrics(preds, targets)
        train_metrics['loss'] = avg_loss

        # Validation evaluation
        val_metrics = evaluate_model(model, validation_loader, device)

        # Test evaluation
        test_metrics = evaluate_model(model, test_loader, device)
        
        # Save history
        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)
        
        # Track best model
        if val_metrics["f1"] > history['best_val']['f1']:
            # Track best model performance
            history['best_train'] = train_metrics
            history['best_test'] = test_metrics
            history['best_val'] = val_metrics
            history['best_epoch'] = epoch + 1
            logger.info(f"✓ New best model! Test F1: {test_metrics['f1']:.4f}")

            # Save best model to file
            torch.save(model.state_dict(), os.path.join(log_dir, f'best_model_epoch{epoch+1}.pth'))
            patient = 0
        else:
            patient += 1
            if patient > early_stop:
                break
        
        # Log to console
        logger.info(f"Epoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s")
        logger.info(f"Train Loss: {avg_loss:.4f}, Train Acc: {train_metrics['acc']:.4f}, Train F1: {train_metrics['f1']:.2f}%, Train Precision: {train_metrics['precision']:.2f}%, Train Recall: {train_metrics['recall']:.2f}%")
        logger.info(f"Test Acc: {test_metrics['acc']:.4f}, Test F1: {test_metrics['f1']:.2f}%, Test Precision: {test_metrics['precision']:.2f}%, Test Recall: {test_metrics['recall']:.2f}%")
        logger.info("-" * 60)
    
    # Final comprehensive log
    history['total_training_time'] = sum(history['epoch_times'])
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best Results:")
    logger.info(f"  Epoch: {history['best_epoch']}/{epochs}")
    logger.info(f"  Train: {history['best_train']}")
    logger.info(f"  Test: {history['best_test']}")
    logger.info(f"  Validation: {history['best_val']}")
    logger.info(f"Total Training Time: {history['total_training_time']:.2f}s")
    logger.info("=" * 60)

    return history


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device (str): 'cpu' or 'cuda'
    
    Returns:
        float: Test accuracy (percentage)
    """
    model.eval()
    preds = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    return compute_metrics(preds, targets)

def compute_metrics(y_pred, y_true):
    """
    Compute accuracy, precision, recall, and F1-score.
    
    Args:
        y_pred (list or np.array): Predicted labels
        y_true (list or np.array): True labels
    Returns:
        dict: Dictionary with accuracy, precision, recall, F1-score
    """
    
    accuracy = accuracy_score(y_true, y_pred) * 100.0
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    
    return {
        'acc': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_expressivity(circuit, num_qubits, trained_params, num_samples=1000, num_bins=75, shots=4096):
    """
    Calculate the expressivity of a trained quantum circuit as KL-divergence from the Haar distribution.
    
    Following the method from the paper:
    1. Sample random parameter configurations using trained parameters
    2. Execute circuit with measurements for each configuration
    3. Normalize probability distribution: N * p
    4. Create buckets for the normalized probabilities
    5. Re-arrange occurrences over buckets and re-count distribution
    6. Compute Haar measure for each bucket
    7. Compute KL-divergence between circuit distribution and Haar distribution
    
    Args:
        circuit (QuantumCircuit): The parameterized quantum circuit to evaluate
        num_qubits (int): Number of qubits in the circuit
        trained_params (np.array or torch.Tensor): Trained variational parameters from the model
        num_samples (int): Number of random parameter samples to draw
        num_bins (int): Number of bins for discretizing the probability distribution
        shots (int): Number of shots for each circuit execution
    
    Returns:
        float: The expressivity value (KL-divergence from Haar distribution)
    """
    logger.info(f"Calculating expressivity with {num_samples} samples and {num_bins} bins...")
    logger.info("Using trained parameters from the model")
    
    # Convert trained params to numpy if needed
    if hasattr(trained_params, 'detach'):
        trained_params = trained_params.detach().cpu().numpy()
    trained_params = np.array(trained_params)
    
    # Step 1: Get all parameters from the circuit
    params = list(circuit.parameters)
    num_params = len(params)
    logger.info(f"Circuit has {num_params} parameters")
    
    # Separate feature parameters and variational parameters
    # Feature params: 'x[0]', 'x[1]', ... (first num_qubits params)
    # Variational params: 'θ_rep0[0]', 'θ_rep0[1]', ... (remaining params)
    num_feature_params = num_qubits
    num_var_params = num_params - num_feature_params
    
    if len(trained_params) != num_var_params:
        logger.warning(f"Trained params length ({len(trained_params)}) doesn't match expected ({num_var_params})")
        # Pad or truncate as needed
        if len(trained_params) < num_var_params:
            trained_params = np.pad(trained_params, (0, num_var_params - len(trained_params)), mode='constant')
        else:
            trained_params = trained_params[:num_var_params]
    
    logger.info(f"Feature parameters: {num_feature_params}, Variational parameters: {num_var_params}")
    
    # Add measurements if not present
    circuit_with_meas = circuit.copy()
    if circuit_with_meas.num_clbits == 0:
        circuit_with_meas.measure_all()
    
    # Step 2: Sample random parameter configurations and execute circuits
    sampler = SamplerV2()
    sampler.options.default_shots = shots
    sampler.options.seed_simulator = 12345
    
    # Collect all probability distributions from samples
    all_probs = []
    
    logger.info("Sampling random input features with trained variational parameters...")
    for sample_idx in range(num_samples):
        # Generate random INPUT features uniformly in [0, 2π]
        # Keep trained VARIATIONAL parameters fixed
        random_features = np.random.uniform(0, 2 * np.pi, num_feature_params)
        
        # Combine: random features + trained variational parameters
        all_params = np.concatenate([random_features, trained_params])
        
        # Bind parameters to circuit
        param_dict = {params[i]: all_params[i] for i in range(num_params)}
        bound_circuit = circuit_with_meas.assign_parameters(param_dict)
        
        # Execute circuit
        job = sampler.run([bound_circuit])
        result = job.result()
        
        # Get measurement counts
        counts = result[0].data.meas.get_counts()
        
        # Convert counts to probability distribution
        # Build probability vector for all 2^n possible outcomes
        num_states = 2 ** num_qubits
        prob_dist = np.zeros(num_states)
        
        for bitstring, count in counts.items():
            # Convert bitstring to integer index
            state_idx = int(bitstring, 2)
            prob_dist[state_idx] = count / shots
        
        all_probs.append(prob_dist)
    
    # Convert to array: shape (num_samples, 2^n)
    all_probs = np.array(all_probs)
    logger.info(f"Collected {num_samples} probability distributions")
    
    # Step 3: Normalization - N * p
    # For each sample, normalize probabilities by multiplying by number of states
    N = 2 ** num_qubits
    normalized_probs = N * all_probs  # Shape: (num_samples, N)
    
    # Step 4 & 5: Create buckets and re-arrange occurrences
    # Flatten all normalized probabilities and histogram them
    all_normalized_values = normalized_probs.flatten()
    
    # Create bins from 0 to N (the maximum possible value of N*p is N when p=1)
    bin_edges = np.linspace(0, N, num_bins + 1)
    
    # Count how many normalized probability values fall in each bin
    hist, _ = np.histogram(all_normalized_values, bins=bin_edges)
    
    # Normalize to get probability distribution over bins
    circuit_distribution = hist / hist.sum()
    
    # Step 6: Compute the Haar measure for each bucket
    # The Haar measure gives the expected distribution for a random quantum state
    # For a bucket [k, k+1], the Haar measure is approximately:
    # P_Haar(k) = (N-1) * (1 - k/N)^(N-2) / N for k in [0, N]
    
    # Use bin centers for Haar calculation
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute Haar measure for each bin
    # Formula: (N-1) * (1 - x/N)^(N-2) / N where x is the bin center
    haar_distribution = np.zeros(num_bins)
    for i, x in enumerate(bin_centers):
        if x < N:
            # Haar distribution formula from Porter-Thomas distribution
            haar_distribution[i] = (N - 1) * ((1 - x / N) ** (N - 2)) / N
        else:
            haar_distribution[i] = 0
    
    # Normalize Haar distribution
    haar_distribution = haar_distribution / haar_distribution.sum()
    
    # Step 7: Compute KL-divergence
    # KL(P || Q) = sum(P(i) * log(P(i) / Q(i)))
    # Add small epsilon to avoid division by zero and log(0)
    epsilon = 1e-10
    circuit_distribution = circuit_distribution + epsilon
    haar_distribution = haar_distribution + epsilon
    
    # Renormalize after adding epsilon
    circuit_distribution = circuit_distribution / circuit_distribution.sum()
    haar_distribution = haar_distribution / haar_distribution.sum()
    
    # Calculate KL-divergence using scipy
    kl_divergence = entropy(circuit_distribution, haar_distribution)
    
    logger.info(f"Expressivity (KL-divergence from Haar): {kl_divergence:.6f}")
    
    return kl_divergence

def plot_circuit(circuit, filename='quantum_circuit.png', decompose_depth=0):
    """
    Plot and save the quantum circuit diagram.
    
    Args:
        circuit (QuantumCircuit): Quantum circuit to plot
        filename (str): Filename to save the plot
        decompose_depth (int): Number of decomposition levels (0=show blocks, 1+=show gates)
    """
    # Show abstracted view with blocks (decompose_depth=0) or gates (decompose_depth>0)
    if decompose_depth > 0:
        circuit_to_plot = circuit.decompose(reps=decompose_depth)
    else:
        circuit_to_plot = circuit
    
    # Draw with better visualization parameters
    fig = circuit_to_plot.draw(
        output='mpl',
        fold=100,  # Fold for readability
        scale=0.8
    )
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Quantum circuit diagram saved to {filename}")
    plt.close(fig)


def main():
    """
    Main execution function.
    """
    parser = argparse.ArgumentParser(description="Hybrid CNN-QNN Classifier for MNIST")
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes to classify (4, 6, or 8)')
    parser.add_argument('--samples_per_class', type=int, default=200, help='Number of samples per class for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for optimizer')
    parser.add_argument('--feature_map_type', type=str, default='standard', choices=['standard', 'partial', 'partial_RA'], help='Type of feature map for data re-uploading')
    parser.add_argument('--ansatz_type', type=str, default='tensor_ring', choices=['tensor_ring', 'mps_ttn_combined'], help='Type of ansatz for variational circuit')
    parser.add_argument('--reps', type=int, default=2, help='Number of [feature_map → ansatz] repetitions')
    parser.add_argument('--early_stop', type=int, default=2, help='Early stopping patience')
    args = parser.parse_args()

    # Set mlflow experiment or create if it doesn't exist
    mlflow.set_experiment("QuantumCNN_SamplerV2")

    with mlflow.start_run():
        # Configuration
        NUM_CLASSES = args.num_classes
        SAMPLES_PER_CLASS = args.samples_per_class
        BATCH_SIZE = args.batch_size
        EPOCHS = args.epochs
        LEARNING_RATE = args.learning_rate
        FEATURE_MAP_TYPE = args.feature_map_type
        ANSZATZ_TYPE = args.ansatz_type
        REPS = args.reps  # Number of [feature_map → ansatz] repetitions (standard=2, partial=3, partial_RA=3)
        EARLY_STOP = args.early_stop

        # Log configuration to mlflow
        mlflow.log_params({
            'num_classes': NUM_CLASSES,
            'samples_per_class': SAMPLES_PER_CLASS,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'feature_map_type': FEATURE_MAP_TYPE,
            'reps': REPS,
            'ansatz_type': ANSZATZ_TYPE,
            'early_stop': EARLY_STOP
        })

        # Make result directory
        results_dir = f'../results/{FEATURE_MAP_TYPE}_re-uploading_{ANSZATZ_TYPE}_MNIST_{NUM_CLASSES}_classes_{SAMPLES_PER_CLASS}_samples_x_class'
        os.makedirs(results_dir, exist_ok=True)
        
        # Configure logging to write to both file and console
        log_file = os.path.join(results_dir, 'hybrid_cnn_qnn.log')
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        
        # Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Load data
        logger.info(f"Loading MNIST data: {NUM_CLASSES} classes, {SAMPLES_PER_CLASS} samples/class")
        train_loader, test_loader, validation_loader = load_mnist_dataset(
            num_classes=NUM_CLASSES,
            samples_per_class=SAMPLES_PER_CLASS,
            batch_size=BATCH_SIZE,
            device=device
        )
        
        # Create quantum circuit with repetitions
        logger.info(f"Creating quantum circuit: {FEATURE_MAP_TYPE} re-uploading with {REPS} repetitions")
        qnn, circuit = create_quantum_circuit(
            feature_map_type=FEATURE_MAP_TYPE, 
            num_qubits=4,
            reps=REPS,
            ansatz_type=ANSZATZ_TYPE
        )

        # Plot and save quantum circuit diagram showing architecture
        # decompose_depth=0 shows high-level blocks (feature maps + ansatz)
        # decompose_depth=1+ shows individual gates (more detailed)
        """circuit_plot_file = os.path.join(results_dir, f'quantum_circuit_{FEATURE_MAP_TYPE}_{ANSZATZ_TYPE}_{NUM_CLASSES}classes_reps{REPS}.png')
        plot_circuit(circuit, filename=circuit_plot_file, decompose_depth=1)"""
        
        # Train the model    
        logger.info("Building hybrid CNN-QNN model...")
        model = HybridCNNQNN(qnn, num_classes=NUM_CLASSES, num_qubits=4).to(device)
        logger.info(f"Model created successfully")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Train model with logging
        res = train_hybrid_model(
            model,
            train_loader,
            test_loader,
            validation_loader,
            epochs=EPOCHS,
            early_stop=EARLY_STOP,
            learning_rate=LEARNING_RATE,
            device=device,
            log_dir=os.path.join(results_dir)
        )

        # Calculate expressivity of the quantum circuit AFTER training
        # Expressivity measures how the trained circuit explores the state space
        # using the trained variational parameters with random input features
        logger.info("=" * 60)
        logger.info("Calculating circuit expressivity (after training)...")
        logger.info("=" * 60)
        
        # Extract trained variational parameters from the QNN
        trained_weights = model.qnn.weight.detach().cpu().numpy()
        logger.info(f"Extracted {len(trained_weights)} trained parameters from QNN")
        
        expressivity = calculate_expressivity(
            circuit=circuit,
            num_qubits=4,
            trained_params=trained_weights,
            num_samples=1000,  # Number of random input feature samples
            num_bins=75,       # Number of bins for distribution
            shots=4096         # Shots per circuit execution
        )
        logger.info(f"Circuit Expressivity (KL-divergence): {expressivity:.6f}")
        logger.info("=" * 60)

        # Log metrics to mlflow
        mlflow.log_metrics({
            'expressivity': expressivity,
            'best_train_acc': res['best_train']['acc'],
            'best_train_f1': res['best_train']['f1'],
            'best_train_precision': res['best_train']['precision'],
            'best_train_recall': res['best_train']['recall'],
            'best_test_acc': res['best_test']['acc'],
            'best_test_f1': res['best_test']['f1'],
            'best_test_precision': res['best_test']['precision'],
            'best_test_recall': res['best_test']['recall'],
            'best_val_acc': res['best_val']['acc'],
            'best_val_f1': res['best_val']['f1'],
            'best_val_precision': res['best_val']['precision'],
            'best_val_recall': res['best_val']['recall'],
            'best_epoch': res['best_epoch'],
            'total_training_time_sec': res['total_training_time']
        })

if __name__ == '__main__':
    main()
