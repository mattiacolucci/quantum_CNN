"""
Feature maps for data re-uploading quantum classifiers.
Project 10: Quantum Classifiers with Data Re-uploading on MNIST
"""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
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


def partial_encoding_RA_feature_map(num_qubits=4, hadamard_init=True, squared_transform=False):
    """
    Partial encoding with only 2 qubits at a time receiving features, alternating with RealAmplitudes.
    
    Pattern (3 blocks):
    1. Encode on q0, q1: Rx(x[0])+Rz(x[1]) on q0, Rx(x[2])+Rz(x[3]) on q1
       → RealAmplitudes on all 4 qubits
    2. Encode on q1, q2: Rx(x[4])+Rz(x[5]) on q1, Rx(x[6])+Rz(x[7]) on q2
       → RealAmplitudes on all 4 qubits
    3. Encode on q2, q3: Rx(x[8])+Rz(x[9]) on q2, Rx(x[10])+Rz(x[11]) on q3
       → RealAmplitudes on all 4 qubits
    
    Total: 12 input features + RealAmplitudes variational parameters
    
    Args:
        num_qubits (int): Number of qubits (should be 4)
        hadamard_init (bool): Whether to apply Hadamard gates for initialization
        squared_transform (bool): Whether to apply squared transformation to features
    
    Returns:
        QuantumCircuit: Feature map circuit with embedded RealAmplitudes blocks
    """
    assert num_qubits == 4, "Partial encoding with 2 qubits uses 4 qubits total"
    
    qc = QuantumCircuit(num_qubits)
    features = ParameterVector('x', num_qubits*3)
    
    if squared_transform:
        features = [features[i]**2 / np.pi for i in range(num_qubits*3)]
    
    # Optional Hadamard initialization on all qubits
    if hadamard_init:
        for i in range(num_qubits):
            qc.h(i)
        qc.barrier()
    
    # Block 1: Encode on q0, q1
    qc.rx(features[0], 0)
    qc.rz(features[1], 0)
    qc.rx(features[2], 1)
    qc.rz(features[3], 1)
    qc.barrier()
    
    # RealAmplitudes block 1 on qubits 0, 1
    ra1 = RealAmplitudes(num_qubits=2, reps=1, parameter_prefix='w_L0', insert_barriers=False)
    qc.compose(ra1, qubits=[0, 1], inplace=True)
    qc.cx(0, 1)  # CNOT with control=0, target=1
    qc.barrier()
    
    # Block 2: Encode on q1, q2
    qc.rx(features[4], 1)
    qc.rz(features[5], 1)
    qc.rx(features[6], 2)
    qc.rz(features[7], 2)
    qc.barrier()
    
    # RealAmplitudes block 2 on qubits 1, 2
    ra2 = RealAmplitudes(num_qubits=2, reps=1, parameter_prefix='w_L1', insert_barriers=False)
    qc.compose(ra2, qubits=[1, 2], inplace=True)
    qc.cx(1, 2)  # CNOT with control=1, target=2
    qc.barrier()

    # Block 3: Encode on q2, q3
    qc.rx(features[8], 2)
    qc.rz(features[9], 2)
    qc.rx(features[10], 3)
    qc.rz(features[11], 3)
    qc.barrier()
    
    # RealAmplitudes block 3 on qubits 2, 3
    ra3 = RealAmplitudes(num_qubits=2, reps=1, parameter_prefix='w_L2', insert_barriers=False)
    qc.compose(ra3, qubits=[2, 3], inplace=True)
    qc.cx(2, 3)  # CNOT with control=2, target=3
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
