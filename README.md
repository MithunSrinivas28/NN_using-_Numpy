# Building a Neural Network From Scratch using NumPy

A simple implementation of a neural network for handwritten digit classification using only NumPy and linear algebra.

Dataset used: MNIST

---

# Overview

This project builds a basic neural network without using frameworks like TensorFlow or PyTorch.

The network learns to classify handwritten digits from 0–9 by performing:

1. Forward Propagation  
2. Backpropagation  
3. Gradient Descent Updates  

Everything is implemented manually using NumPy.

---

# Dataset

MNIST dataset contains:

- 28 × 28 grayscale digit images
- 784 pixels per image
- Labels from 0 to 9

Each image is flattened into a vector of size:

```python
784 x 1
```

---

# Neural Network Architecture

```text
Input Layer      → 784 neurons
Hidden Layer     → 10 neurons
Output Layer     → 10 neurons
```

Output layer represents digit probabilities:

```text
[0,1,2,3,4,5,6,7,8,9]
```

---

# Training Process

The model trains using three major steps.

---

## 1. Forward Propagation

Input moves through the network to generate predictions.

### Hidden Layer

```math
Z1 = W1X + b1
A1 = ReLU(Z1)
```

### Output Layer

```math
Z2 = W2A1 + b2
A2 = Softmax(Z2)
```

---

# ReLU Activation

```math
ReLU(x) = max(0, x)
```

Used to introduce non-linearity.

---

# Softmax Function

Converts output values into probabilities.

```math
Softmax(z_i) = \frac{e^{z_i}}{\sum e^{z_j}}
```

The output probabilities always sum to 1.

---

# Backpropagation

Calculates how wrong the predictions are and computes gradients.

### Output Error

```math
dZ2 = A2 - Y
```

### Gradients

```math
dW2 = \frac{1}{m} dZ2 A1^T
db2 = \frac{1}{m} \sum dZ2
```

### Hidden Layer Error

```math
dZ1 = W2^T dZ2 * ReLU'(Z1)
```

---

# Gradient Descent

Updates weights and biases to reduce error.

```math
W = W - \alpha dW
b = b - \alpha db
```

Where:

- `α` = learning rate

---

# Parameter Initialization

Weights are initialized randomly between:

```python
-0.5 to 0.5
```

Shapes:

```python
W1 -> (10, 784)
b1 -> (10, 1)

W2 -> (10, 10)
b2 -> (10, 1)
```

---

# One Hot Encoding

Labels are converted into binary vectors.

Example:

```text
Digit: 3

[0,0,0,1,0,0,0,0,0,0]
```

---

# Training Loop

```python
for iteration in range(iterations):

    forward_propagation()
    backpropagation()
    update_parameters()
```

Accuracy is printed every few iterations.

---

# Results

After fixing initialization and propagation issues:

| Iterations | Accuracy |
|---|---|
| 100 | ~75% |
| 250 | ~84% |

Development set accuracy:

```text
~85.5%
```

---

# Key Concepts Learned

- Matrix multiplication in neural networks
- Forward propagation
- Backpropagation
- ReLU activation
- Softmax probabilities
- Gradient descent
- One-hot encoding
- Weight updates

---

# Future Improvements

Possible upgrades:

- More hidden layers
- More neurons
- Better optimizers
  - Adam
  - RMSProp
  - Momentum
- Regularization
- Improved accuracy tuning

---

# Technologies Used

- Python
- NumPy
- Pandas
- Jupyter Notebook
- Kaggle

---

# Final Note

This project is useful for understanding how neural networks work internally instead of relying entirely on high-level ML frameworks.
