"""
Hybrid CNN-Quantum Neural Network Classifier for MNIST with GPU Support
Project 10: Quantum Classifiers with Data Re-uploading on MNIST

GPU-Enabled Version:
- CNN runs on PyTorch GPU (CUDA)
- Quantum simulation runs on Qiskit Aer GPU (requires qiskit-aer-gpu)

Architecture:
1. Classical CNN extracts features from MNIST images
2. CNN outputs 4 features
3. VQC (Variational Quantum Circuit) with data re-uploading processes these 4 features
4. Quantum circuit is measured for classification (on GPU if available)
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import dagshub
import mlflow
from tqdm import tqdm
from sklearn.model_selection import KFold

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.quantum_info import SparsePauliOp
import json
import os
from datetime import datetime

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.feature_maps import (
    standard_reuploading_feature_map,
    partial_reuploading_feature_map
)
from src.utils.ansatz import tensor_ring

logger = logging.getLogger(__name__)

dagshub.init(repo_owner='mattiacolucci', repo_name='quantum_CNN', mlflow=True)


def check_qiskit_gpu_support():
    """
    Check if Qiskit Aer has GPU support available.
    
    Returns:
        tuple: (has_gpu, device_name, available_devices)
    """
    try:
        sim = AerSimulator()
        available_devices = sim.available_devices()
        has_gpu = 'GPU' in available_devices
        device_name = 'GPU' if has_gpu else 'CPU'
        return has_gpu, device_name, available_devices
    except Exception as e:
        logger.error(f"Error checking GPU support: {e}")
        return False, 'CPU', ['CPU']


class CNNFeatureExtractor(nn.Module):
    """
    Classical CNN that extracts 4 features from MNIST images (28x28).
    These 4 features will be fed to the quantum circuit.
    """
    def __init__(self):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)
        
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 4)  # Output 4 features for quantum circuit
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Normalize to [0, π] range for quantum encoding
        x = torch.sigmoid(x) * np.pi
        
        return x


def create_quantum_circuit_gpu(feature_map_type='standard', num_qubits=4, reps=2, use_gpu=True):
    """
    Create GPU-accelerated quantum neural network with data re-uploading.
    
    Args:
        feature_map_type (str): 'standard' or 'partial' re-uploading
        num_qubits (int): Number of qubits (must be 4)
        reps (int): Number of [feature_map → ansatz] repetitions
        use_gpu (bool): Whether to use GPU for quantum simulation
    
    Returns:
        TorchConnector wrapped quantum circuit
    """
    logger.info(f"Creating {feature_map_type} re-uploading quantum circuit with {reps} repetitions")
    
    # Check GPU availability
    has_gpu, device_name, available_devices = check_qiskit_gpu_support()
    
    if use_gpu and has_gpu:
        logger.info(f"✓ Qiskit Aer GPU support detected! Using GPU for quantum simulation")
        qiskit_device = 'GPU'
    else:
        if use_gpu and not has_gpu:
            logger.warning(f"⚠️  GPU requested but not available. Available devices: {available_devices}")
            logger.warning(f"⚠️  Falling back to CPU for quantum simulation")
            logger.warning(f"⚠️  To enable GPU: install qiskit-aer-gpu or build from source with GPU support")
        qiskit_device = 'CPU'
    
    # Build complete circuit with repetitions
    full_circuit = QuantumCircuit(num_qubits)
    
    # Feature parameters (same features re-uploaded each time)
    feature_params = ParameterVector('x', num_qubits)
    
    # Variational parameters (different for each repetition)
    all_var_params = []
    
    for rep in range(reps):
        # 1. Feature map (data re-uploading) - SAME features each time
        if feature_map_type == 'standard':
            fm = standard_reuploading_feature_map(num_qubits=num_qubits)
        elif feature_map_type == 'partial':
            fm = partial_reuploading_feature_map(num_qubits=num_qubits)
        else:
            raise ValueError(f"Unknown feature map type: {feature_map_type}")
        
        # Bind feature parameters
        param_dict = {fm.parameters[i]: feature_params[i] for i in range(len(fm.parameters))}
        fm_bound = fm.assign_parameters(param_dict)
        full_circuit.compose(fm_bound, inplace=True)
        full_circuit.barrier()
        
        # 2. Ansatz (variational circuit) - DIFFERENT parameters each time
        ansatz = tensor_ring(num_qubits, reps=1).decompose()
        var_params = ParameterVector(f'θ_rep{rep}', ansatz.num_parameters)
        all_var_params.extend(var_params)
        
        param_dict = {ansatz.parameters[i]: var_params[i] for i in range(len(ansatz.parameters))}
        ansatz_bound = ansatz.assign_parameters(param_dict)
        full_circuit.compose(ansatz_bound, inplace=True)
        
        if rep < reps - 1:
            full_circuit.barrier()
    
    logger.info(f"Circuit built: {num_qubits} qubits, {reps} repetitions")
    logger.info(f"Parameters: {len(feature_params)} feature params, {len(all_var_params)} variational params")
    logger.info(f"Quantum simulation device: {qiskit_device}")

    # Create a copy for visualization
    circuit_with_measurements = full_circuit.copy()
    circuit_with_measurements.measure_all()
    
    # Create GPU-enabled or CPU backend
    if qiskit_device == 'GPU':
        # Configure GPU backend
        backend = AerSimulator(method='statevector', device='GPU')
        logger.info(f"✓ AerSimulator configured with GPU acceleration")
    else:
        # Fallback to CPU
        backend = AerSimulator(method='statevector', device='CPU')
        logger.info(f"AerSimulator configured with CPU")
    
    # Create EstimatorV2 with GPU-enabled backend
    estimator = EstimatorV2(backend=backend)
    estimator.options.default_shots = 4096
    estimator.options.seed_simulator = 12345
    
    # Define observable
    observable = SparsePauliOp.from_list([("Z" * num_qubits, 1.0)])
    
    # Create EstimatorQNN
    qnn = EstimatorQNN(
        circuit=full_circuit,
        observables=observable,
        input_params=feature_params,
        weight_params=all_var_params,
        estimator=estimator,
        input_gradients=True
    )
    
    # Wrap with TorchConnector
    torch_connector = TorchConnector(qnn)
    
    return torch_connector, circuit_with_measurements, qiskit_device


class HybridCNNQNN_GPU(nn.Module):
    """
    GPU-Accelerated Hybrid CNN-Quantum Neural Network.
    
    Architecture:
    1. CNN extracts 4 features from images (GPU)
    2. Quantum circuit processes these 4 features (GPU if available)
    3. Quantum output is used for classification (GPU)
    """
    def __init__(self, qnn, num_classes=4, qiskit_on_gpu=False):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.qnn = qnn
        self.linear = nn.Linear(1, num_classes)
        self.qiskit_on_gpu = qiskit_on_gpu
        
    def forward(self, x, apply_softmax=False):
        # Extract features with CNN
        x = self.cnn(x)  # (batch, 4)
        
        # Process with quantum circuit
        if self.qiskit_on_gpu:
            # Quantum circuit runs on GPU - no need to move to CPU
            x = self.qnn(x)  # (batch, 1)
        else:
            # Quantum circuit runs on CPU - need CPU-GPU transfers
            device = x.device
            x_cpu = x.cpu()
            x = self.qnn(x_cpu)
            x = x.to(device)
        
        # Projection to class logits
        x = self.linear(x)  # (batch, num_classes)

        if apply_softmax:
            x = F.softmax(x, dim=1)
        
        return x


def load_mnist_dataset(num_classes=4, samples_per_class=100, k_folds=5, seed=12345):
    """Load MNIST data and create PyTorch DataLoaders."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.info(f"Loading MNIST dataset with {num_classes} classes...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    mnist_train = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root="../data", train=False, download=True, transform=transform)

    all_data = torch.cat([mnist_train.data, mnist_test.data], dim=0)
    all_targets = torch.cat([mnist_train.targets, mnist_test.targets], dim=0)

    kfold_datasets = []
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(all_data)):
        val_size = int(0.5 * len(test_ids))
        val_ids = test_ids[:val_size]
        test_ids = test_ids[val_size:]

        X_train = all_data[train_ids] / 255.0
        y_train = all_targets[train_ids]
        X_test = all_data[test_ids] / 255.0
        y_test = all_targets[test_ids]
        X_val = all_data[val_ids] / 255.0
        y_val = all_targets[val_ids]
    
        selected_indices = []
        for class_label in range(num_classes):
            indices = [i for i in range(len(y_train)) if y_train[i] == class_label]
            selected = np.random.choice(indices, samples_per_class, replace=False)
            selected_indices.extend(selected)
    
        X_train_subset = X_train[selected_indices]
        y_train_subset = y_train[selected_indices]
        
        shuffle_idx = torch.randperm(len(X_train_subset))
        X_train = X_train_subset[shuffle_idx]
        y_train = y_train_subset[shuffle_idx]
        
        logger.info(f"Train dataset shape: {X_train.shape}, Labels shape: {y_train.shape}")
        logger.info(f"Test dataset shape: {X_test.shape}, Labels shape: {y_test.shape}")
        
        train_dataset = TensorDataset(X_train.unsqueeze(1), y_train)
        test_dataset = TensorDataset(X_test.unsqueeze(1), y_test)
        validation_dataset = TensorDataset(X_val.unsqueeze(1), y_val)

        kfold_datasets.append({"train": train_dataset, "test": test_dataset, "validation": validation_dataset})
    
    logger.info(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}, Validation size: {len(validation_dataset)}")
    
    return kfold_datasets


def train_hybrid_model(
    model,
    train_loader,
    test_loader,
    validation_loader,
    epochs=20,
    learning_rate=0.001,
    device='cpu',
    log_dir='logs'
):
    """Train the hybrid CNN-QNN model with comprehensive logging."""
    model = model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'best_epoch': 0,
        'best_train': {},
        'best_test': {},
        'best_val': {'f1': 0.0},
        'epoch_times': [],
    }
    
    patient = 0
    early_stop = 10
    
    config_log = {
        'device': device,
        'qiskit_device': 'GPU' if model.qiskit_on_gpu else 'CPU',
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
        
        model.train()
        total_loss = 0
        preds = []
        targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
        
        model.eval()
        avg_loss = total_loss / len(train_loader)
        train_metrics = compute_metrics(preds, targets)
        train_metrics['loss'] = avg_loss

        val_metrics = evaluate_model(model, validation_loader, device)
        test_metrics = evaluate_model(model, test_loader, device)
        
        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)
        
        if val_metrics["f1"] > history['best_val']['f1']:
            history['best_train'] = train_metrics
            history['best_test'] = test_metrics
            history['best_val'] = val_metrics
            history['best_epoch'] = epoch + 1
            logger.info(f"✓ New best model! Val F1: {val_metrics['f1']:.4f}, Test F1: {test_metrics['f1']:.4f}")
            torch.save(model.state_dict(), os.path.join(log_dir, f'best_model_epoch{epoch+1}.pth'))
            patient = 0
        else:
            patient += 1
            logger.info(f"No improvement for {patient}/{early_stop} epochs")
            if patient >= early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s")
        logger.info(f"Train Loss: {avg_loss:.4f}, Acc: {train_metrics['acc']:.2f}%, F1: {train_metrics['f1']:.2f}%")
        logger.info(f"Val   Acc: {val_metrics['acc']:.2f}%, F1: {val_metrics['f1']:.2f}%")
        logger.info(f"Test  Acc: {test_metrics['acc']:.2f}%, F1: {test_metrics['f1']:.2f}%")
        logger.info("-" * 60)
    
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
    """Evaluate model on test set."""
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
    """Compute accuracy, precision, recall, and F1-score."""
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


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="GPU-Accelerated Hybrid CNN-QNN Classifier for MNIST")
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes to classify')
    parser.add_argument('--samples_per_class', type=int, default=400, help='Number of samples per class')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--feature_map_type', type=str, default='standard', choices=['standard', 'partial'])
    parser.add_argument('--reps', type=int, default=2, help='Number of repetitions')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for quantum simulation if available')
    args = parser.parse_args()

    with mlflow.start_run():
        NUM_CLASSES = args.num_classes
        SAMPLES_PER_CLASS = args.samples_per_class
        BATCH_SIZE = args.batch_size
        EPOCHS = args.epochs
        LEARNING_RATE = args.learning_rate
        FEATURE_MAP_TYPE = args.feature_map_type
        REPS = args.reps
        USE_GPU = args.use_gpu

        mlflow.log_params({
            'num_classes': NUM_CLASSES,
            'samples_per_class': SAMPLES_PER_CLASS,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'feature_map_type': FEATURE_MAP_TYPE,
            'reps': REPS,
            'use_qiskit_gpu': USE_GPU
        })

        results_dir = f'../results/{FEATURE_MAP_TYPE}_re-uploading_MNIST_{NUM_CLASSES}_classes_reps{REPS}_gpu'
        os.makedirs(results_dir, exist_ok=True)
        
        log_file = os.path.join(results_dir, 'hybrid_cnn_qnn_gpu.log')
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        
        # Device configuration
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info("="*60)
        logger.info("Device Configuration:")
        logger.info(f"  PyTorch device: {device}")
        if device == 'cuda':
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Check Qiskit GPU support
        has_qiskit_gpu, qiskit_device, qiskit_devices = check_qiskit_gpu_support()
        logger.info(f"  Qiskit Aer available devices: {qiskit_devices}")
        logger.info(f"  Qiskit GPU support: {'✓ Available' if has_qiskit_gpu else '✗ Not available'}")
        if USE_GPU and not has_qiskit_gpu:
            logger.warning("  ⚠️  GPU requested but Qiskit GPU not available - will use CPU for quantum")
        logger.info("="*60)
        
        # Load data
        logger.info(f"Loading MNIST data: {NUM_CLASSES} classes, {SAMPLES_PER_CLASS} samples/class")
        kfold_datasets = load_mnist_dataset(
            num_classes=NUM_CLASSES,
            samples_per_class=SAMPLES_PER_CLASS
        )
        
        # Create quantum circuit with GPU support
        logger.info(f"Creating quantum circuit: {FEATURE_MAP_TYPE} re-uploading with {REPS} repetitions")
        qnn, circuit, qiskit_device = create_quantum_circuit_gpu(
            feature_map_type=FEATURE_MAP_TYPE, 
            num_qubits=4,
            reps=REPS,
            use_gpu=USE_GPU
        )

        # K-fold training
        results = []
        for fold_idx, fold in enumerate(kfold_datasets):
            logger.info("="*60)
            logger.info(f"FOLD {fold_idx+1}/{len(kfold_datasets)} Training")
            logger.info("="*60)
            
            model = HybridCNNQNN_GPU(qnn, num_classes=NUM_CLASSES, qiskit_on_gpu=(qiskit_device=='GPU')).to(device)
            logger.info(f"Model created successfully")
            logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

            use_pinned = (device == 'cuda')
            train_loader = DataLoader(fold["train"], batch_size=BATCH_SIZE, shuffle=True, 
                                    pin_memory=use_pinned, num_workers=0)
            test_loader = DataLoader(fold["test"], batch_size=BATCH_SIZE, shuffle=False, 
                                   pin_memory=use_pinned, num_workers=0)
            validation_loader = DataLoader(fold["validation"], batch_size=BATCH_SIZE, shuffle=False,
                                         pin_memory=use_pinned, num_workers=0)

            os.makedirs(os.path.join(results_dir, f"fold_{fold_idx+1}"), exist_ok=True)
        
            res = train_hybrid_model(
                model,
                train_loader,
                test_loader,
                validation_loader,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                device=device,
                log_dir=os.path.join(results_dir, f"fold_{fold_idx+1}")
            )
            results.append(res)

        # Average results over folds
        avg_results = {}
        for key in results[0].keys():
            if key == 'epoch_times':
                continue
            avg_results[key] = {}
            for metric in results[0][key].keys():
                avg_results[key][metric] = np.mean([r[key][metric] for r in results])

        # Log metrics to mlflow
        mlflow.log_metrics({
            'best_train_acc': avg_results['best_train']['acc'],
            'best_train_f1': avg_results['best_train']['f1'],
            'best_train_precision': avg_results['best_train']['precision'],
            'best_train_recall': avg_results['best_train']['recall'],
            'best_test_acc': avg_results['best_test']['acc'],
            'best_test_f1': avg_results['best_test']['f1'],
            'best_test_precision': avg_results['best_test']['precision'],
            'best_test_recall': avg_results['best_test']['recall'],
            'best_val_acc': avg_results['best_val']['acc'],
            'best_val_f1': avg_results['best_val']['f1'],
            'best_val_precision': avg_results['best_val']['precision'],
            'best_val_recall': avg_results['best_val']['recall'],
            'total_training_time_sec': avg_results['total_training_time']
        })

        logger.info("="*60)
        logger.info("GPU-Accelerated Training Complete!")
        logger.info(f"CNN Device: {device}")
        logger.info(f"Quantum Device: {qiskit_device}")
        logger.info("="*60)


if __name__ == '__main__':
    main()
