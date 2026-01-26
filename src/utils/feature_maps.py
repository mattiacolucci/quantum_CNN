"""
Feature maps for data re-uploading quantum classifiers.
Project 10: Quantum Classifiers with Data Re-uploading on MNIST
"""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np


def standard_reuploading_feature_map(num_qubits=4, hadamard_init=True, squared_transform=False):
    """
    Standard re-uploading: features encoded on DIFFERENT rotation gates with DIFFERENT transformations.
    
    Block 1: H + Ry(x) - Initialize with Hadamard and encode original features
    Block 2: Rx(x²) - Re-upload squared features (NO Hadamard to preserve information)
    
    This increases expressivity by encoding x and x² in different bases (Y and X rotation).
    Hadamard is ONLY in first block to avoid destroying processed information.
    
    Args:
        num_qubits (int): Number of qubits (should be 4)
    
    Returns:
        QuantumCircuit: Feature map circuit
    """
    assert num_qubits == 4, "Standard re-uploading uses 4 qubits"
    
    qc = QuantumCircuit(num_qubits)
    features = ParameterVector('x', num_qubits)

    if squared_transform:
        features = [features[i]**2 / np.pi for i in range(num_qubits)]
    
    # Block 1: Hadamard + Ry(x) - Initialize and encode original features
    if hadamard_init:
        for i in range(num_qubits):
            qc.h(i)  # Hadamard ONLY in first block for initialization
    for i in range(num_qubits):
        qc.ry(features[i], i)  # Encode x_i
    qc.barrier()
    
    for i in range(num_qubits):
        qc.rx(features[i], i)  # Encode x_i
    qc.barrier()
    
    return qc


def partial_reuploading_feature_map(num_qubits=4, hadamard_init=True, squared_transform=False):
    """
    Partial re-uploading block: features encoded SEQUENTIALLY at different moments
    
    Block 1: H + Ry(x₀, x₁) - Initialize and encode first 2 features
    Block 2: Ry(x₂) - upload third feature
    Block 3: Ry(x₃) - upload fourth feature
    
    Args:
        num_qubits (int): Number of qubits (should be 4)
    
    Returns:
        QuantumCircuit: Feature map circuit
    """
    assert num_qubits == 4, "Partial re-uploading uses 4 qubits"
    
    qc = QuantumCircuit(num_qubits)
    features = ParameterVector('x', num_qubits)

    if squared_transform:
        features = [features[i]**2 / np.pi for i in range(num_qubits)]
    
    # Block 1: Initialize with Hadamard + encode first 2 features (original)
    if hadamard_init:
        for i in range(num_qubits):
            qc.h(i)  # Hadamard ONLY in first block for initialization
        qc.barrier()
    
    # Encode first 2 features with original values
    qc.ry(features[0], 0)  # x₀
    qc.ry(features[1], 1)  # x₁
    qc.barrier()
    
    # Block 2: upload third feature
    qc.ry(features[2], 2)  # x₂
    qc.barrier()
    
    # Block 3: upload fourth feature
    qc.ry(features[3], 3)  # x₃
    qc.barrier()
    
    return qc


def simple_feature_map(num_qubits=4):
    """
    Simple feature map: H + Ry for baseline comparison.
    
    Args:
        num_qubits (int): Number of qubits
    
    Returns:
        QuantumCircuit: Feature map circuit
    """
    qc = QuantumCircuit(num_qubits)
    features = ParameterVector('x', num_qubits)
    
    for i in range(num_qubits):
        qc.h(i)
        qc.ry(features[i], i)
    
    return qc
