# Project 10: Quantum Classifiers with Data Re-uploading on MNIST

## Project Description

This project implements quantum classifiers for MNIST digit classification using data re-uploading techniques. The implementation now includes a **hybrid CNN-Quantum approach** where:

1. **Classical CNN** extracts 4 features from MNIST images
2. **Quantum Circuit (VQC)** processes these 4 features using data re-uploading
3. **Classical Measurement** produces the final classification

### Two Re-uploading Strategies:

1. **Standard Re-uploading**: Features encoded on different rotation gates (Ry, Rx)
2. **Partial Re-uploading**: Features encoded sequentially at different circuit stages

## Requirements

Based on the project notes:
- ✅ Use **4 qubits** with **4 features** (one per qubit)
- ✅ Use **Tensor Ring** as the variational ansatz
- ✅ Test on **4, 6, and 8 MNIST classes**
- ✅ **No dimensionality reduction** (no autoencoder, PCA, or complex preprocessing)
- ✅ Features reduced to 4 by spatial averaging (image quadrants)

## Files Structure

### Main Implementation Files

**Hybrid CNN-Quantum Approach (RECOMMENDED):**
- **`hybrid_cnn_qnn.py`**: Hybrid CNN-Quantum classifier (PyTorch + Qiskit)
  - CNN extracts 4 features from images
  - Quantum circuit processes features with data re-uploading
  - Final classification via classical measurement

**Pure Quantum Approach:**
- **`mnist_classifier.py`**: Pure quantum classifier with manual feature reduction
  - Uses spatial averaging to reduce images to 4 features
  - Direct VQC classification

**Shared Components:**
- **`feature_maps.py`**: Feature map implementations (standard/partial re-uploading)
- **`ansatz.py`**: Quantum circuit ansatzes (Tensor Ring, TTN, MPS)
# Quantum computing
pip install qiskit qiskit-machine-learning

# Deep learning
pip install torch torchvision

# Scientific computing and utilities
pip install numpy scikit-learn matplotlib
pip installuirements and notes (Italian)
- **`README.md`**: This file

### Old/Unused Files
- **`data_pipeline.py`**: Contains autoencoder (not used as per requirements)
- **`EsophagusCancerClassifier.ipynb`**: Unrelated old project

## Installation

```bash
pip install qiskit qiskit-machine-learning
pip install tor - Hybrid CNN-QNN (Recommended)

Run the hybrid CNN-Quantum classifier:

```bash
python hybrid_cnn_qnn.py
```

This will:
1. Train a CNN to extract 4 features from MNIST images
2. Feed features to quantum circuit with data re-uploading
3. Perform classification using quantum measurements
4. Save model, results, and training plots

### Alternative - Pure Quantum Approach

Run the pure quantum classifier:

```bash
python mnist_classifier.py
```
#### Hybrid CNN-QNN

```python
from hybrid_cnn_qnn import (
    load_mnist_dataloader, 
    create_quantum_circuit, 
    HybridCNNQNN, 
    train_hybrid_model
)
import torch

# Load data
train_loader, test_loader, _ = load_mnist_dataloader(
    num_classes=4,
    samples_per_class=100,
    batch_size=16
)

# Create quantum circuit
qnn = create_quantum_circuit(feature_map_type='standard', num_qubits=4)

# Build hybrid model
model = HybridCNNQNN(qnn, num_classes=4)

# Train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
history = train_hybrid_model(model, train_loader, test_loader, epochs=20, device=device)
```

#### Pure Quantum


This will:
1. Load MNIST data for 4, 6, and 8 classes
2. Use spatial averaging to reduce to 4 features
3. Train classifiers with both standard and partial re-uploading
4ython mnist_classifier.py
```

This will:
1. Load MNIST data for 4, 6, and 8 classes
2. Train classifiers with both standard and partial re-uploading
3. Evaluate and save results
### Hybrid CNN-QNN Approach (Recommended)
1. **Load MNIST**: Select first N classes (4, 6, or 8)
2. **CNN Feature Extraction**: CNN learns to extract 4 optimal features
3. **Automatic Normalization**: CNN outputs are scaled to [0, π] for quantum encoding
4. **End-to-End Training**: CNN and quantum circuit trained together

### Pure Quantum Approach
1. **Load MNIST**: Select first N classes (4, 6, or 8)
2. **Sample**: Take equal samples per class
3. **Feature Reduction**: Divide 28x28 image into 4 quadrants, average each
4. **Normalization**: Scale to [0, π] for quantum encoding

**Note: Both approaches avoid
# Load data
X_train, X_test, y_train, y_test = load_mnist_data(
    num_classes=4,
    samples_per_class=100
)

# Build classifier (standard re-uploading)
classifier = build_quantum_classifier(
    feature_map_type='standard',  # or 'partial'
    ansatz_type='tensor_ring',
    num_qubits=4,
    num_classes=4,
    maxiter=100
)

# Train and evaluate
results = train_and_evaluate(classifier, X_train, X_test, y_train, y_test)
```
### Hybrid CNN-QNN
- Training history plot (loss and accuracy curves)
- Trained model (`.pth` file)
- Results dictionary (`.pkl` file)
- Console logs with batch-by-batch progress
- Final test accuracy

### Pure Quantum
## Feature Map Architectures

### Standard Re-uploading (2 blocks)
```
Block 1: H gates + Ry(x[i]) for all qubits
Block 2: Rx(x[i]) for all qubits (same features, different gates)
```
- Uses **different rotation gates** (Ry, Rx)
- All features loaded simultaneously

### Partial Re-uploading (3 blocks)
```
Block 1: H gates + Ry(x[0]), Ry(x[1])
Block 2: Ry(x[2])
Block 3: Ry(x[3])
```
- Uses **same rotation gate** (Ry only)
- Features loaded **sequentially** at different times

## Data Preprocessing

1. **Load MNIST**: Select first N classes (4, 6, or 8)
2. **Sample**: Take equal samples per class
3. **Feature Reduction**: Divide 28x28 image into 4 quadrants, average each
4. **Normalization**: Scale to [0, π] for quantum encoding

**No PCA, autoencoder, or complex feature engineering as per project requirements.**

## Ansatz

Uses **Tensor Ring** (circular entanglement):
- Connects adjacent qubits: (0,1), (1,2), (2,3)
- Closes the loop: (3,0)
- Provides full entanglement in a ring topology

## Expected Output

The classifier will output:
- Training accuracy
- Test accuracy
- Training time
- Saved models (`.pkl` files)
- Summary of all experiments
### Hybrid CNN-QNN
- Start with **50 samples per class** and small batch size (8-16)
- Use GPU if available: `device = 'cuda'`
- Reduce `epochs` to 10-20 for initial testing
- Monitor training: CNN should learn quickly, quantum part adds refinement

### Pure Quantum

## Notes from Requirements

From `notes.txt` (translated):
- Output dimension must be > number of qubits
- Fixed VQC (using Tensor Ring)
- Standard re-uploading uses **different** rotation gates
- Partial re-uploading uses **same** rotation gates
- Don't use Rz alone, prefer Ry and Rx
- Final code should be packaged in provided XML format
- If issues arise, check Qiskit version compatibility

## Performance Tips

- Start with **50-100 samples per class** for faster testing
- Increase `maxiter` (iterations) for better accuracy
- Use `COBYLA` optimizer (gradient-free, good for noisy circuits)
- Consider reducing shot count for faster but noisier results

## Troubleshooting

1. **Memory issues**: Reduce `samples_per_class` or `maxiter`
2. **Slow training**: Decrease `shots` in Sampler options
3. **Poor accuracy**: Increase training samples and iterations
4. **Qiskit errors**: Verify version compatibility with course materials

## Future Work

- Implement XML packaging as specified in project requirements
- Test additional ansatzes (TTN comparison)
- Hyperparameter optimization
- Cross-validation for robust evaluation
