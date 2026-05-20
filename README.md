# Building a Neural Network From Scratch using NumPy

A simple implementation of a neural network for handwritten digit classification using only NumPy and linear algebra.

Dataset used: MNIST

---

# Overview

This project builds a neural network completely from scratch without using frameworks like TensorFlow or PyTorch.

Instead of calling prebuilt training functions, every major operation is manually implemented using:
- NumPy
- matrix multiplication
- activation functions
- backpropagation
- gradient descent

The goal of this project is to understand what actually happens inside a neural network during training.

---

# Dataset

The project uses the MNIST handwritten digit dataset.

Each image:
- is grayscale
- has dimensions `28 × 28`
- contains `784` total pixels

Before training, every image is flattened into a vector:

```python
784 x 1
```

This vector becomes the input to the neural network.

<img width="1371" height="1148" alt="ChatGPT Image May 20, 2026, 06_33_43 PM" src="https://github.com/user-attachments/assets/71ce46df-61ae-4bf1-af20-c3c3da2374f4" />

---

# Neural Network Architecture

The model contains three layers:

```text
Input Layer      → 784 neurons
Hidden Layer     → 10 neurons
Output Layer     → 10 neurons
```

The output layer represents probabilities for digits:

```text
[0,1,2,3,4,5,6,7,8,9]
```

The neuron with the highest probability becomes the predicted digit.

---

# Working of the Neural Network

The neural network learns using three main stages:

```text
1. Forward Propagation
2. Backpropagation
3. Parameter Updates
```

These steps repeat continuously during training.

---

# 1. Forward Propagation

Forward propagation is the prediction phase.

The input image moves through the network layer by layer until the final prediction is generated.

---

## Hidden Layer Computation

The hidden layer performs:
- weighted multiplication
- bias addition
- activation

Mathematically:

```math
Z1 = W1X + b1
```

Where:
- `X` → input image
- `W1` → weights
- `b1` → bias
- `Z1` → raw hidden layer output

---

## ReLU Activation

The hidden layer uses the ReLU activation function:

```math
ReLU(x) = max(0, x)
```

Meaning:

```text
x > 0  → keep value
x <= 0 → convert to 0
```

Activated output:

```math
A1 = ReLU(Z1)
```

ReLU introduces non-linearity so the network can learn complex patterns instead of behaving like a simple linear equation.

---

## Output Layer

The activated hidden layer values move into the output layer.

```math
Z2 = W2A1 + b2
```

Then softmax converts the output into probabilities:

```math
A2 = Softmax(Z2)
```

---

# Softmax Function

Softmax converts raw outputs into probability values.

```math
Softmax(z_i) = \frac{e^{z_i}}{\sum e^{z_j}}
```

Example output:

```text
[0.01, 0.02, 0.91, 0.03, ...]
```

The highest probability becomes the predicted digit.

---

# 2. Backpropagation

Backpropagation is the learning phase.

The network compares:
- predicted output
- actual output

and calculates the error.

---

## Error Calculation

```math
dZ2 = A2 - Y
```

Where:
- `A2` → predicted output
- `Y` → actual label

---

## Gradient Calculation

Gradients are calculated to determine:
- which weights caused the error
- how much they should change

```math
dW2 = \frac{1}{m} dZ2 A1^T
```

```math
db2 = \frac{1}{m} \sum dZ2
```

The error is then propagated backward through the hidden layer.

---

# 3. Parameter Updates

Weights and biases are updated using gradient descent.

```math
W = W - \alpha dW
```

```math
b = b - \alpha db
```

Where:
- `α` = learning rate

This step slowly reduces prediction error over multiple iterations.

---

# Complete Training Flow

```text
Input Image
    ↓
Forward Propagation
    ↓
Prediction
    ↓
Loss Calculation
    ↓
Backpropagation
    ↓
Gradient Calculation
    ↓
Weight Update
    ↓
Improved Prediction
```

The cycle repeats until the network reaches good accuracy.

---

# One Hot Encoding

Labels are converted into binary vectors before training.

Example:

```text
Digit: 3

[0,0,0,1,0,0,0,0,0,0]
```

This format helps compare predictions with actual labels during backpropagation.

---

# Training Loop

The neural network repeatedly performs:

```python
for iteration in range(iterations):

    forward_propagation()
    backpropagation()
    update_parameters()
```

Accuracy is printed during training to monitor learning progress.

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

This project helps understand:

- Matrix multiplication in neural networks
- Forward propagation
- Backpropagation
- ReLU activation
- Softmax probabilities
- Gradient descent
- One-hot encoding
- Weight updates
- Learning through error correction

---

# Future Improvements

Possible improvements include:

- More hidden layers
- More neurons
- Better optimizers
  - Adam
  - RMSProp
  - Momentum
- Regularization
- Better weight initialization
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

Building a neural network from scratch gives a much deeper understanding of how machine learning models actually work internally.

Instead of treating neural networks like a black box, this project explains:
- how predictions are generated
- how errors are calculated
- how weights learn over time
- how mathematical operations power deep learning systems
