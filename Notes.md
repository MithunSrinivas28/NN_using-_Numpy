# Neural Network From Scratch — NumPy Study Notes

> These notes are written so you can come back after 2 months, read top to bottom, and fully understand what the project does, why every decision was made, and how the math connects to the code.

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [The Problem We Are Solving](#2-the-problem-we-are-solving)
3. [The Dataset — MNIST](#3-the-dataset--mnist)
4. [Network Architecture](#4-network-architecture)
5. [NumPy Fundamentals Used Here](#5-numpy-fundamentals-used-here)
6. [Data Preprocessing — Step by Step](#6-data-preprocessing--step-by-step)
7. [Parameter Initialization](#7-parameter-initialization)
8. [Forward Propagation — Making a Prediction](#8-forward-propagation--making-a-prediction)
9. [Activation Functions — Why They Exist](#9-activation-functions--why-they-exist)
10. [Backpropagation — How the Network Learns](#10-backpropagation--how-the-network-learns)
11. [Gradient Descent — Updating the Weights](#11-gradient-descent--updating-the-weights)
12. [One-Hot Encoding](#12-one-hot-encoding)
13. [The Full Training Loop](#13-the-full-training-loop)
14. [Making Predictions After Training](#14-making-predictions-after-training)
15. [Results and What They Mean](#15-results-and-what-they-mean)
16. [Complete Code With Inline Explanations](#16-complete-code-with-inline-explanations)
17. [Mental Model — The Big Picture](#17-mental-model--the-big-picture)
18. [Common Mistakes and Why They Happen](#18-common-mistakes-and-why-they-happen)

---

## 1. What This Project Is

Most people learn machine learning like this:

```python
model.fit(X, y)
```

One line. Magic happens. Accuracy appears.

But that hides **everything important**.

This project removes the magic. You implement a neural network using only:

- `NumPy` — matrix math
- `pandas` — loading the CSV
- `matplotlib` — visualizing images

No TensorFlow. No PyTorch. No Keras.

**Why does this matter for placements/interviews?**

When an interviewer asks "what is backpropagation" or "how does gradient descent work" — people who built it from scratch give better answers than people who called `.fit()`. You can point to actual equations and code you wrote yourself.

---

## 2. The Problem We Are Solving

**Task:** Given an image of a handwritten digit (0 through 9), predict which digit it is.

```
Input: image of handwritten "7"
Output: 7
```

This is a **multi-class classification** problem with 10 classes (digits 0–9).

---

## 3. The Dataset — MNIST

**MNIST** is the "hello world" of machine learning.

- Tens of thousands of grayscale images
- Each image is **28 × 28 pixels**
- That equals **784 pixels** per image
- Each pixel is a number between **0 and 255** (0 = black, 255 = white)
- Each image has a **label** (the actual digit it represents)

**How the CSV looks:**

```
label | pixel0 | pixel1 | pixel2 | ... | pixel783
  1   |   0    |   0    |   0    | ... |    0
  0   |   0    |   0    |   0    | ... |    0
  4   |   0    |   0    |   0    | ... |    0
```

- First column: the true digit (label)
- Remaining 784 columns: pixel intensity values

**Train set:** 42,000 images
**Test set:** 28,000 images (no labels — used for Kaggle submission)

---

## 4. Network Architecture

```
Input Layer        Hidden Layer       Output Layer
(784 neurons)  →   (10 neurons)   →   (10 neurons)
```

**Layer 0 (Input):** 784 nodes, one per pixel.

**Layer 1 (Hidden):** 10 neurons. These learn internal features — edges, curves, shapes.

**Layer 2 (Output):** 10 neurons. One per digit (0 through 9). The neuron with the highest value wins.

**Why only 10 hidden neurons?**

This is a small network intentionally. More neurons = more parameters = more complexity. For MNIST, 10 hidden neurons already gives ~85% accuracy. It's enough to understand the mechanics.

**How the output works:**

After the output layer, we apply softmax which turns 10 raw numbers into 10 probabilities that sum to 1.

```
Raw output:    [2.1, 0.3, 0.1, ..., 5.4, ...]
After softmax: [0.02, 0.01, ..., 0.87, ...]   ← probabilities
```

The index with the highest probability is the predicted digit.

---

## 5. NumPy Fundamentals Used Here

Before touching the neural net code, understand the NumPy operations it depends on.

### 5.1 Arrays and Shapes

```python
import numpy as np

a = np.array([1, 2, 3])        # 1D array, shape (3,)
b = np.array([[1, 2], [3, 4]]) # 2D array, shape (2, 2)

print(a.shape)  # (3,)
print(b.shape)  # (2, 2)
```

**Shape is everything.** In neural networks, you constantly need to make sure matrix dimensions are compatible.

### 5.2 Transpose

```python
X = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape (2, 3)

X.T                          # shape (3, 2)
```

We use `.T` constantly because data usually comes as `(samples, features)` but neural nets want `(features, samples)`.

### 5.3 Matrix Multiplication (dot product)

```python
A = np.random.rand(10, 784)  # weights, shape (10, 784)
X = np.random.rand(784, 100) # input, shape (784, 100) — 100 examples

Z = A.dot(X)                 # result shape = (10, 100)
```

Rule: `(m × k) dot (k × n)` = `(m × n)`. Inner dimensions must match.

This is the core operation of forward propagation.

### 5.4 Broadcasting

```python
Z = np.random.rand(10, 100)  # shape (10, 100)
b = np.random.rand(10, 1)    # shape (10, 1)

result = Z + b               # NumPy broadcasts b across all 100 columns
# result shape = (10, 100)
```

Broadcasting means NumPy automatically extends smaller arrays to match larger ones along compatible dimensions. This is how we add bias to all training examples at once.

### 5.5 Element-wise Operations

```python
a = np.array([1, -2, 3, -4])

np.maximum(a, 0)   # [1, 0, 3, 0]  — this is ReLU
np.exp(a)          # e^1, e^-2, e^3, e^-4
```

### 5.6 Axis-based Operations

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

np.sum(A)           # 21   — sum of everything
np.sum(A, axis=0)   # [5, 7, 9]   — sum down each column
np.sum(A, axis=1)   # [6, 15]     — sum across each row
np.argmax(A, axis=0) # index of max value in each column
```

`argmax` is used to pick the predicted digit from the output layer.

### 5.7 Boolean Masks

```python
Z = np.array([3, -1, 0, 5, -2])

mask = Z > 0        # [True, False, False, True, False]
Z * mask            # [3, 0, 0, 5, 0]
```

This is exactly how the derivative of ReLU is computed in backpropagation.

### 5.8 Random Initialization

```python
np.random.rand(10, 784)       # values between 0 and 1
np.random.rand(10, 784) - 0.5 # values between -0.5 and 0.5
```

Subtracting 0.5 centers the values around zero. This is important for symmetric initialization.

---

## 6. Data Preprocessing — Step by Step

```python
import numpy as np
import pandas as pd

data = pd.read_csv("train.csv")

# Convert DataFrame to NumPy array
data = np.array(data)           # shape: (42000, 785)
m, n = data.shape               # m = 42000 samples, n = 785 columns

# Shuffle to randomize order — prevents the model from learning order patterns
np.random.shuffle(data)
```

**Why shuffle?**

If the data is ordered (all 0s first, then all 1s, etc.), early batches would only teach the network about certain digits. Shuffling ensures it sees all digits randomly.

```python
# Split into dev set (validation) and training set
data_dev = data[0:1000].T        # first 1000 rows, then transposed
Y_dev = data_dev[0]              # labels — shape (1000,)
X_dev = data_dev[1:n]            # pixels — shape (784, 1000)
X_dev = X_dev / 255.             # normalize

data_train = data[1000:m].T      # remaining rows, transposed
Y_train = data_train[0]          # labels — shape (41000,)
X_train = data_train[1:n]        # pixels — shape (784, 41000)
X_train = X_train / 255.         # normalize

_, m_train = X_train.shape       # m_train = 41000
```

**Key things happening here:**

**Transpose (`.T`):**
- Original data shape after slicing: `(samples, 785)`
- After `.T`: `(785, samples)`
- Now `data_dev[0]` = first row = all labels
- `data_dev[1:n]` = rows 1 to 784 = all pixel columns
- This makes the pixel matrix shape `(784, samples)` — each column is one image

**Normalization (`/ 255.`):**
- Pixel values originally range from 0 to 255
- Dividing by 255 puts them in range [0, 1]
- Neural networks train much better on small numbers
- Prevents weights from growing extremely large during training

---

## 7. Parameter Initialization

```python
def init_params():
    W1 = np.random.rand(10, 784) - 0.5   # shape (10, 784)
    b1 = np.random.rand(10, 1) - 0.5     # shape (10, 1)
    W2 = np.random.rand(10, 10) - 0.5    # shape (10, 10)
    b2 = np.random.rand(10, 1) - 0.5     # shape (10, 1)
    return W1, b1, W2, b2
```

**Understanding the shapes:**

| Parameter | Shape | Why |
|-----------|-------|-----|
| `W1` | `(10, 784)` | 10 hidden neurons, each connected to 784 inputs |
| `b1` | `(10, 1)` | one bias per hidden neuron |
| `W2` | `(10, 10)` | 10 output neurons, each connected to 10 hidden neurons |
| `b2` | `(10, 1)` | one bias per output neuron |

**Why random and not zeros?**

If all weights are zero, every neuron computes the same thing. All gradients become identical. All neurons update identically. The network never learns different features. This is called the **symmetry problem**.

Random initialization **breaks symmetry** — each neuron starts at a different value and learns different things.

**Why subtract 0.5?**

`np.random.rand` gives values in `[0, 1]`. Subtracting 0.5 centers them around 0 in range `[-0.5, 0.5]`. Small random values near zero are a good starting point.

---

## 8. Forward Propagation — Making a Prediction

Forward propagation = passing input through the network to produce a prediction.

```
Input X  →  [W1, b1]  →  ReLU  →  [W2, b2]  →  Softmax  →  Prediction
```

### The Math

**Layer 1 (hidden):**

```
Z1 = W1 · X + b1
A1 = ReLU(Z1)
```

**Layer 2 (output):**

```
Z2 = W2 · A1 + b2
A2 = softmax(Z2)
```

### The Code

```python
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1   # (10, 784) · (784, m) + (10, 1) = (10, m)
    A1 = ReLU(Z1)          # (10, m) — element-wise
    Z2 = W2.dot(A1) + b2   # (10, 10) · (10, m) + (10, 1) = (10, m)
    A2 = softmax(Z2)        # (10, m) — probabilities
    return Z1, A1, Z2, A2
```

**Why return Z1, A1, Z2, A2?**

We need all of these during backpropagation. Z1 is needed for the ReLU derivative. A1 is needed for gradient calculations.

### Shape Walkthrough (for 41000 training examples)

| Variable | Shape | Meaning |
|----------|-------|---------|
| `X` | `(784, 41000)` | 784 pixels, 41000 images |
| `W1` | `(10, 784)` | weights layer 1 |
| `Z1 = W1 · X` | `(10, 41000)` | pre-activation hidden layer |
| `A1 = ReLU(Z1)` | `(10, 41000)` | activated hidden layer |
| `W2` | `(10, 10)` | weights layer 2 |
| `Z2 = W2 · A1` | `(10, 41000)` | pre-activation output |
| `A2 = softmax(Z2)` | `(10, 41000)` | output probabilities |

Each column of `A2` is the probability distribution over digits for one image. The column sums to 1.

---

## 9. Activation Functions — Why They Exist

### The Problem Without Activation Functions

If every layer just did `Z = W · X + b`, then chaining layers gives:

```
Z2 = W2 · (W1 · X + b1) + b2
   = (W2 · W1) · X + (W2 · b1 + b2)
   = W_combined · X + b_combined
```

Two layers collapsed into one. No matter how many layers you stack, it's still just linear. A linear model cannot learn complex patterns like recognizing handwritten digits.

**Activation functions introduce non-linearity** — the ability to model complex, curved decision boundaries.

### ReLU (Rectified Linear Unit)

```python
def ReLU(Z):
    return np.maximum(Z, 0)
```

**Math:**
```
ReLU(x) = max(0, x)
```

**Behavior:**
- Positive values: kept as-is
- Negative values: set to zero

**Why ReLU for the hidden layer?**

- Simple and fast to compute
- Does not have the vanishing gradient problem that sigmoid has
- Works well in practice for hidden layers

**Derivative of ReLU:**

```python
def ReLU_deriv(Z):
    return Z > 0   # returns 1 where Z > 0, else 0 (boolean)
```

This boolean array acts as a mask. During backpropagation, gradients only flow through neurons that were active (positive) in the forward pass.

### Softmax (Output Layer)

```python
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
```

**Math:**
```
softmax(z_i) = e^(z_i) / sum(e^(z_j) for all j)
```

**What it does:**

Converts raw scores into probabilities.

**Example:**
```
Z2 column:  [2.1, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 5.4, 0.0, 0.0]
After exp:  [8.2, 1.3, 1.0, 1.0, 1.0, 1.0, 1.0, 221, 1.0, 1.0]
After norm: [0.03, 0.01, 0.004, ..., 0.87, ...]
```

Digit 7 has probability 0.87 — the network predicts 7.

**Why softmax for the output layer?**

- Outputs are probabilities (between 0 and 1)
- All outputs sum to 1
- Highest probability = predicted class

---

## 10. Backpropagation — How the Network Learns

Backpropagation answers: "by how much should each weight change to reduce the error?"

### The Concept

After forward propagation, we have a prediction `A2`. We compare it to the true label `Y`. The difference is the error. We need to propagate this error backwards through the network to calculate how much each weight contributed to it.

### The Math

**Step 1: Output layer error**

```
dZ2 = A2 - Y_one_hot
```

This is the difference between predicted probabilities and actual one-hot labels.

**Step 2: Output layer weight gradients**

```
dW2 = (1/m) * dZ2 · A1ᵀ
db2 = (1/m) * sum(dZ2)
```

**Step 3: Hidden layer error (chain rule)**

```
dZ1 = W2ᵀ · dZ2 * ReLU'(Z1)
```

The `*` here is element-wise multiplication (not dot product).

`ReLU'(Z1)` is the derivative of ReLU applied to Z1 — it's 1 where Z1 > 0, and 0 elsewhere.

**Step 4: Hidden layer weight gradients**

```
dW1 = (1/m) * dZ1 · Xᵀ
db1 = (1/m) * sum(dZ1)
```

### The Code

```python
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y               # (10, m) — output error
    dW2 = 1/m * dZ2.dot(A1.T)          # (10, 10) — W2 gradient
    db2 = 1/m * np.sum(dZ2)            # scalar — b2 gradient

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)  # (10, m) — hidden error
    dW1 = 1/m * dZ1.dot(X.T)              # (10, 784) — W1 gradient
    db1 = 1/m * np.sum(dZ1)               # scalar — b1 gradient

    return dW1, db1, dW2, db2
```

**Why divide by m?**

Dividing by the number of training examples `m` gives us the **average gradient** across all examples. Without this, the gradient magnitude would depend on dataset size, making the learning rate sensitive to how much data you have.

**Why `W2.T`?**

During backpropagation we "reverse" the forward pass. In forward prop, `Z2 = W2 · A1`. To propagate error backwards through W2, we use `W2ᵀ`. This is the chain rule through matrix multiplication.

---

## 11. Gradient Descent — Updating the Weights

```python
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2
```

**The update rule:**

```
W := W - α * dW
```

- `W` = current weight
- `α` = learning rate (hyperparameter, set to 0.10 here)
- `dW` = gradient (direction of steepest increase in error)
- We subtract because we want to go downhill (minimize error)

**Why 0.10 as learning rate?**

- Too small (e.g., 0.001): learning is very slow, needs many iterations
- Too large (e.g., 1.0): updates overshoot, loss oscillates or diverges
- 0.10 is a reasonable starting point for this problem

**The gradient tells us:**

"If we increase this weight slightly, does the error go up or down?"

The gradient points toward increasing error, so we subtract it to move toward decreasing error.

---

## 12. One-Hot Encoding

The labels `Y` are single numbers like `[0, 4, 7, 1, ...]`.

But our output layer produces 10 probabilities. We need a comparable target format.

**One-hot encoding** converts each label into a vector of length 10 where only the correct class position is 1.

```
Label: 7
One-hot: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                                ^
                           position 7 = 1
```

```python
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))  # shape (m, 10)
    one_hot_Y[np.arange(Y.size), Y] = 1          # set correct position to 1
    one_hot_Y = one_hot_Y.T                       # shape (10, m)
    return one_hot_Y
```

**NumPy trick used here:**

`one_hot_Y[np.arange(Y.size), Y] = 1`

`np.arange(Y.size)` = `[0, 1, 2, 3, ..., m-1]` — row indices

`Y` = `[0, 4, 7, 1, ...]` — column indices

This fancy indexing sets position `(i, Y[i])` to 1 for each example `i` — exactly the correct class.

---

## 13. The Full Training Loop

```python
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()        # random init

    for i in range(iterations):
        # Step 1: Forward pass — make prediction
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)

        # Step 2: Backward pass — compute gradients
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)

        # Step 3: Update weights
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        # Print accuracy every 10 iterations
        if i % 10 == 0:
            predictions = get_predictions(A2)
            print("Iteration:", i, " | Accuracy:", get_accuracy(predictions, Y))

    return W1, b1, W2, b2
```

**Run with:**

```python
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
```

**What happens across 500 iterations:**

| Iteration | Accuracy |
|-----------|----------|
| 0 | ~8.6% (random guessing = 10%) |
| 50 | ~58% |
| 100 | ~70% |
| 250 | ~80% |
| 490 | ~85% |

The network starts at near-random performance and improves as weights update to recognize digit patterns.

---

## 14. Making Predictions After Training

```python
def get_predictions(A2):
    return np.argmax(A2, 0)   # index of max value in each column
```

`A2` shape is `(10, m)`. Each column is one image's probability distribution.

`np.argmax(A2, 0)` finds the row index (0–9) with the highest value in each column. That index is the predicted digit.

```python
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size
```

Count how many predictions match the true labels. Divide by total to get accuracy.

```python
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions
```

```python
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]   # shape (784, 1) — one image
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = Y_train[index]

    print("Prediction:", prediction)
    print("Label:", label)

    # Reshape to 28x28 for display
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
```

**Dev set evaluation:**

```python
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)   # ~81%
```

The dev set was held out during training, so this accuracy reflects how well the model generalizes.

---

## 15. Results and What They Mean

**Training accuracy after 500 iterations: ~85%**

**Dev (validation) accuracy: ~81%**

The small gap between training and dev accuracy suggests minimal overfitting — the network generalizes reasonably.

**Why not 99%?**

This is a deliberately small network (784 → 10 → 10). Improvements would come from:

- More hidden neurons (e.g., 128 or 256)
- More hidden layers
- Better optimizers (Adam instead of vanilla gradient descent)
- Regularization (dropout, L2)
- Data augmentation

For a network built from scratch in pure NumPy with 10 hidden neurons — 85% is solid.

---

## 16. Complete Code With Inline Explanations

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ─── Load and Preprocess Data ───────────────────────────────────────────────

data = np.array(pd.read_csv("train.csv"))
m, n = data.shape        # m = number of samples, n = 785 (label + 784 pixels)
np.random.shuffle(data)  # shuffle to remove any ordering bias

# Dev set: first 1000 examples
# Transpose so shape is (785, 1000), then split label from pixels
data_dev = data[0:1000].T
Y_dev = data_dev[0]          # shape (1000,) — true labels
X_dev = data_dev[1:n] / 255. # shape (784, 1000) — normalized pixels

# Train set: remaining examples
data_train = data[1000:m].T
Y_train = data_train[0]           # shape (41000,)
X_train = data_train[1:n] / 255.  # shape (784, 41000)
_, m_train = X_train.shape        # number of training examples


# ─── Parameter Initialization ────────────────────────────────────────────────

def init_params():
    # Weights: small random values centered at 0
    W1 = np.random.rand(10, 784) - 0.5  # (hidden_neurons, input_features)
    b1 = np.random.rand(10, 1) - 0.5    # (hidden_neurons, 1)
    W2 = np.random.rand(10, 10) - 0.5   # (output_neurons, hidden_neurons)
    b2 = np.random.rand(10, 1) - 0.5    # (output_neurons, 1)
    return W1, b1, W2, b2


# ─── Activation Functions ────────────────────────────────────────────────────

def ReLU(Z):
    # Returns max(0, x) element-wise — kills negative values
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    # Derivative of ReLU: 1 if Z > 0, else 0
    # Boolean array used as a mask in backprop
    return Z > 0

def softmax(Z):
    # Converts raw scores to probabilities
    # Each column sums to 1
    A = np.exp(Z) / sum(np.exp(Z))
    return A


# ─── One-Hot Encoding ────────────────────────────────────────────────────────

def one_hot(Y):
    # Converts [0, 4, 7, ...] → 10-row binary matrix
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))  # (m, 10)
    one_hot_Y[np.arange(Y.size), Y] = 1          # set correct class to 1
    one_hot_Y = one_hot_Y.T                       # (10, m)
    return one_hot_Y


# ─── Forward Propagation ─────────────────────────────────────────────────────

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1   # linear transform: (10, 784)·(784, m) = (10, m)
    A1 = ReLU(Z1)          # non-linearity: (10, m)
    Z2 = W2.dot(A1) + b2   # second linear transform: (10, 10)·(10, m) = (10, m)
    A2 = softmax(Z2)        # output probabilities: (10, m), each column sums to 1
    return Z1, A1, Z2, A2


# ─── Backpropagation ─────────────────────────────────────────────────────────

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)

    # Output layer: error = prediction - truth
    dZ2 = A2 - one_hot_Y                  # (10, m)
    dW2 = 1/m * dZ2.dot(A1.T)            # (10, 10)
    db2 = 1/m * np.sum(dZ2)              # scalar

    # Hidden layer: propagate error backwards through W2, apply ReLU derivative
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1) # (10, m)
    dW1 = 1/m * dZ1.dot(X.T)             # (10, 784)
    db1 = 1/m * np.sum(dZ1)              # scalar

    return dW1, db1, dW2, db2


# ─── Parameter Update ────────────────────────────────────────────────────────

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    # Move each parameter in the direction that reduces error
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


# ─── Evaluation Helpers ──────────────────────────────────────────────────────

def get_predictions(A2):
    # Index of max probability = predicted digit
    return np.argmax(A2, 0)  # shape (m,)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


# ─── Training Loop ───────────────────────────────────────────────────────────

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 10 == 0:
            print("Iteration:", i)
            print("Accuracy:", get_accuracy(get_predictions(A2), Y))

    return W1, b1, W2, b2

# Train
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)


# ─── Testing ─────────────────────────────────────────────────────────────────

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction:", prediction, "| Label:", label)
    plt.imshow(current_image.reshape((28, 28)) * 255, cmap='gray')
    plt.show()

# Check predictions on dev set
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print("Dev accuracy:", get_accuracy(dev_predictions, Y_dev))
```

---

## 17. Mental Model — The Big Picture

### Think of the network as a student taking a test

| Neural Network | Student Analogy |
|----------------|-----------------|
| Weights | Student's understanding of each concept |
| Bias | Student's baseline tendency |
| Forward prop | Student reads question and writes answer |
| Loss/Error | How wrong the answer was |
| Backprop | Student figures out where reasoning failed |
| Weight update | Student corrects understanding |
| Next iteration | Student tries the next question |

After 500 iterations (practice problems), the student has gotten much better.

### The Information Flow

```
Image pixels (784 numbers)
        ↓
  [W1 · X + b1]       ← 10 neurons detect low-level patterns
        ↓
    ReLU(Z1)           ← suppress irrelevant signals
        ↓
  [W2 · A1 + b2]      ← 10 output neurons combine patterns
        ↓
   softmax(Z2)         ← convert to probabilities
        ↓
   argmax(A2)          ← pick most likely digit
```

### What Each Layer Learns (intuitively)

- **Hidden layer** (W1): Detects combinations of pixel patterns — curves, edges, strokes
- **Output layer** (W2): Combines these patterns to recognize full digits

With only 10 hidden neurons, the representations are limited. More neurons → richer internal representations → higher accuracy.

---

## 18. Common Mistakes and Why They Happen

### Shape errors during matrix multiplication

**Symptom:** `ValueError: shapes (10, 784) and (10, 41000) not aligned`

**Why:** You forgot to transpose. `W1.dot(X)` requires `W1` columns = `X` rows. `W1` is `(10, 784)`, `X` must be `(784, m)` not `(m, 784)`.

**Fix:** Make sure `X` is transposed: `X = data[1:n]` after `data = data.T`.

---

### Accuracy stuck at 10% (random guessing)

**Why:** Weights never update meaningfully. Usually caused by:
- A bug in backprop (wrong variable names)
- Wrong shape somewhere causing broadcasting to mask the real values
- Returning wrong variables from `forward_prop`

**Fix:** Print shapes of all intermediate variables. Confirm they match the table in Section 8.

---

### Loss explodes / NaN values

**Why:** Learning rate too high. Weights overshoot.

**Fix:** Reduce `alpha` (try 0.01 or 0.001).

---

### Softmax numerical instability

**Why:** `np.exp(Z)` can overflow if Z contains very large values.

**Fix (production-grade):** Subtract max before exponentiating:
```python
def softmax(Z):
    Z_stable = Z - np.max(Z, axis=0)
    return np.exp(Z_stable) / np.sum(np.exp(Z_stable), axis=0)
```

For MNIST with small weights this usually isn't a problem, but good to know.

---

### One-hot encoding wrong shape

**Why:** Forgetting to transpose. `one_hot_Y` must be `(10, m)` to subtract from `A2` which is also `(10, m)`.

**Fix:** The `.T` at the end of `one_hot()` is essential.

---

## Quick Reference Cheat Sheet

```
DATA FLOW
─────────────────────────────────────────────────────────────────
X              (784, m)   — input, one image per column
Y              (m,)       — true labels
one_hot_Y      (10, m)    — one-hot encoded labels

FORWARD PROPAGATION
─────────────────────────────────────────────────────────────────
Z1 = W1·X + b1   (10, m)   — hidden pre-activation
A1 = ReLU(Z1)    (10, m)   — hidden activation
Z2 = W2·A1 + b2  (10, m)   — output pre-activation
A2 = softmax(Z2) (10, m)   — output probabilities

BACKPROPAGATION
─────────────────────────────────────────────────────────────────
dZ2 = A2 - Y_oh           (10, m)
dW2 = (1/m) dZ2 · A1ᵀ    (10, 10)
db2 = (1/m) Σ dZ2         scalar
dZ1 = W2ᵀ·dZ2 * relu'(Z1) (10, m)
dW1 = (1/m) dZ1 · Xᵀ     (10, 784)
db1 = (1/m) Σ dZ1         scalar

UPDATE
─────────────────────────────────────────────────────────────────
W := W - α · dW
b := b - α · db
```

---

*Built from scratch. No frameworks. Just NumPy, math, and 500 iterations.*
