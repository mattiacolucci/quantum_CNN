"""
Feature maps for data re-uploading quantum classifiers.
Project 10: Quantum Classifiers with Data Re-uploading on MNIST
"""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np


def standard_reuploading_feature_map(num_qubits=4, num_blocks=2):
    """
    Standard re-uploading: features encoded on DIFFERENT rotation gates.
    Uses Ry and Rx gates alternating between blocks.
    
    Args:
        num_qubits (int): Number of qubits (should be 4)
        num_blocks (int): Number of re-uploading blocks (should be 2)
    
    Returns:
        QuantumCircuit: Feature map circuit
    """
    assert num_qubits == 4, "Standard re-uploading uses 4 qubits"
    
    qc = QuantumCircuit(num_qubits)
    features = ParameterVector('x', num_qubits)
    
    # Block 1: Hadamard + Ry rotations
    for i in range(num_qubits):
        qc.h(i)
    for i in range(num_qubits):
        qc.ry(features[i], i)
    qc.barrier()
    
    # Block 2: Rx rotations (re-uploading same features, different gates)
    for i in range(num_qubits):
        qc.rx(features[i], i)
    qc.barrier()
    
    return qc


def partial_reuploading_feature_map(num_qubits=4, num_blocks=3):
    """
    Partial re-uploading: features encoded SEQUENTIALLY at different moments.
    Uses same rotation gates (Ry) but features loaded in stages.
    
    Block 1: First 2 features (x[0], x[1])
    Block 2: Third feature (x[2])
    Block 3: Fourth feature (x[3])
    
    Args:
        num_qubits (int): Number of qubits (should be 4)
        num_blocks (int): Number of re-uploading blocks (should be 3)
    
    Returns:
        QuantumCircuit: Feature map circuit
    """
    assert num_qubits == 4, "Partial re-uploading uses 4 qubits"
    
    qc = QuantumCircuit(num_qubits)
    features = ParameterVector('x', num_qubits)
    
    # Initialize with Hadamard
    for i in range(num_qubits):
        qc.h(i)
    qc.barrier()
    
    # Block 1: First 2 features
    qc.ry(features[0], 0)
    qc.ry(features[1], 1)
    qc.barrier()
    
    # Block 2: Third feature
    qc.ry(features[2], 2)
    qc.barrier()
    
    # Block 3: Fourth feature
    qc.ry(features[3], 3)
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
