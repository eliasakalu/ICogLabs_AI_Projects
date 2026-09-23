import numpy as np
import numpy.typing as npt


def linear(
    W: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64] | np.float64,
) -> npt.NDArray[np.float64]:
    """Compute a dense-layer affine transformation: XW.T + b."""
    return X @ W.T + b


def sigmoid(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Apply the sigmoid activation element by element."""
    return 1 / (1 + np.exp(-z))


def relu(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Apply the ReLU activation element by element."""
    return np.maximum(0, z)


def softmax(z: npt.NDArray[np.float64], axis: int = -1) -> npt.NDArray[np.float64]:
    """Convert scores to probabilities along the specified axis."""
    z = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

def binary_cross_entropy(
    y_pred: npt.NDArray[np.float64], 
    y_true: npt.NDArray[np.float64]
) -> np.float64:
    """Compute Binary Cross-Entropy loss over batch."""
    eps = 1e-15  
    y_pred = np.clip(y_pred, eps, 1 - eps)
    cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.float64(cost)


def mse_loss(
    y_pred: npt.NDArray[np.float64], 
    y_true: npt.NDArray[np.float64]
) -> np.float64:
    """Compute Mean Squared Error loss over batch."""
    return np.float64(np.mean((y_pred - y_true) ** 2))


def cross_entropy_loss(
    y_pred: npt.NDArray[np.float64], y_true_onehot: npt.NDArray[np.float64]
) -> float:
    """Compute categorical cross-entropy loss over batch."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.sum(y_true_onehot * np.log(y_pred)) / y_pred.shape[0]
    return float(loss)


def mse_loss_backward(
    y_pred: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """dL/dA for mse_loss. Pairs with any final activation (chains through it normally)."""
    return 2 * (y_pred - y_true) / y_pred.size


def binary_cross_entropy_backward(
    y_pred: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """dL/dA for binary_cross_entropy. Pairs with a sigmoid output layer."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / y_pred.size


def softmax_cross_entropy_backward(
    y_pred: npt.NDArray[np.float64], y_true_onehot: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Combined dL/dZ for a softmax output layer + categorical cross-entropy.

    Only correct when the model's final layer uses softmax activation --
    Layer.activation_backward passes softmax gradients straight through,
    assuming this combined form was already applied here.
    """
    return (y_pred - y_true_onehot) / y_pred.shape[0]


def accuracy(
    y_pred: npt.NDArray[np.float64], y_true: npt.NDArray[np.int64]
) -> float:
    """Compute classification accuracy percentage."""
    predictions = np.argmax(y_pred, axis=-1)
    return float(np.mean(predictions == y_true) * 100)