import numpy as np
import numpy.typing as npt

from functions import linear, sigmoid, relu, softmax


class Layer:
    """Base class for all network layers."""

    def __init__(self, activation: str = "linear") -> None:
        self.activation = activation.lower()

    def apply_activation(self, Z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.activation == "linear":
            return Z
        elif self.activation == "sigmoid":
            return sigmoid(Z)
        elif self.activation == "relu":
            return relu(Z)
        elif self.activation == "softmax":
            return softmax(Z)
        else:
            raise ValueError(
                f"Currently we dont have activation func you want '{self.activation}'. "
                "Choose from linear, sigmoid, relu, softmax."
            )

    def activation_backward(
        self, dA: npt.NDArray[np.float64], Z: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        if self.activation == "linear":
            return dA
        elif self.activation == "relu":
            dZ = np.array(dA, copy=True)
            dZ[Z <= 0] = 0
            return dZ
        elif self.activation == "sigmoid":
            s = sigmoid(Z)
            return dA * s * (1 - s)
        elif self.activation == "softmax":
            return dA
        else:
            raise ValueError(f"Unsupported activation '{self.activation}'.")

    def forward(
        self, X: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], tuple]:
        raise NotImplementedError

    def backward(
        self, dA: npt.NDArray[np.float64], cache: tuple
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]:
        raise NotImplementedError

    def update_params(
        self, grads: dict[str, npt.NDArray[np.float64]], lr: float
    ) -> None:
        pass


class Flatten(Layer):
    """Changes (batch, ...) inputs into (batch, features)."""

    def __init__(self) -> None:
        super().__init__(activation="linear")

    def forward(
        self, X: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], tuple]:
        # Save shape for backward pass
        cache = (X.shape,)
        A = X.reshape(X.shape[0], -1)
        return A, cache

    def backward(
        self, dA: npt.NDArray[np.float64], cache: tuple
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]:
        # Reshape incoming gradient back to input shape
        orig_shape = cache[0]
        dA_prev = dA.reshape(orig_shape)
        return dA_prev, {}


class Dense(Layer):
    """Fully Connected (Dense) Layer."""

    def __init__(
        self,
        units: int,
        input_shape: int | None = None,
        activation: str = "linear",
    ) -> None:
        super().__init__(activation=activation)
        self.units = units
        self.input_shape = input_shape
        self.W: npt.NDArray[np.float64] | None = None
        self.b: npt.NDArray[np.float64] | None = None

        if input_shape is not None:
            self._initialize_weights(input_shape)

    def _initialize_weights(self, n_in: int) -> None:
        # He initialization for ReLU, small random weights otherwise
        scale = np.sqrt(2.0 / n_in) if self.activation == "relu" else 0.01
        self.W = np.random.randn(self.units, n_in) * scale
        self.b = np.zeros((1, self.units), dtype=np.float64)

    def forward(
        self, X: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], tuple]:
        if self.W is None or self.b is None:
            self.input_shape = X.shape[-1]
            self._initialize_weights(self.input_shape)

        # Z = XW^T + b
        Z = linear(self.W, X, self.b)
        A = self.apply_activation(Z)
        
        cache = (X, Z)
        return A, cache

    def backward(
        self, dA: npt.NDArray[np.float64], cache: tuple
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]:
        A_prev, Z = cache

        # Compute gradients
        dZ = self.activation_backward(dA, Z)
        dW = dZ.T @ A_prev
        db = np.sum(dZ, axis=0, keepdims=True)
        dA_prev = dZ @ self.W

        grads = {"dW": dW, "db": db}
        return dA_prev, grads

    def update_params(
        self, grads: dict[str, npt.NDArray[np.float64]], lr: float
    ) -> None:
        if self.W is not None and self.b is not None:
            self.W -= lr * grads["dW"]
            self.b -= lr * grads["db"]


class MaxPool2D(Layer):
    """Max pooling with non-overlapping windows that do downsampling.
       No learnable weights. Each window outputs its maximum value.
    """

    def __init__(self, pool_size: int = 2) -> None:
        super().__init__(activation="linear")
        self.pool_size = pool_size

    def forward(
        self, X: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], tuple]:
        batch, H, W, C = X.shape
        p = self.pool_size
        out_h, out_w = H // p, W // p

        # Drop any leftover rows/cols that don't fill a full window
        X_crop = X[:, :out_h * p, :out_w * p, :]
        # Split H and W into (out_h, p) and (out_w, p), then max over each p-sized window
        X_reshaped = X_crop.reshape(batch, out_h, p, out_w, p, C)
        A = X_reshaped.max(axis=(2, 4))

        cache = (X.shape, X_crop, A)
        return A, cache

    def backward(
        self, dA: npt.NDArray[np.float64], cache: tuple
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]:
        X_shape, X_crop, A = cache
        p = self.pool_size

        # Broadcast A and dA back up to full window size to find/route the max
        A_up = np.repeat(np.repeat(A, p, axis=1), p, axis=2)
        dA_up = np.repeat(np.repeat(dA, p, axis=1), p, axis=2)

        # Gradient flows only to the position(s) that were the max in their window
        mask = (X_crop == A_up)
        dX_crop = dA_up * mask

        dA_prev = np.zeros(X_shape)
        dA_prev[:, :X_crop.shape[1], :X_crop.shape[2], :] = dX_crop
        return dA_prev, {}


class Conv2D(Layer):
    """2D convolution over spatial input of shape (batch, H, W, C).
    weight sharing and receptive field increases per layers
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int | tuple[int, int],
        input_shape: tuple[int, int, int] | None = None,
        stride: int = 1,
        padding: str = "valid",
        activation: str = "linear",
    ) -> None:
        super().__init__(activation=activation)
        self.filters = filters
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding.lower()
        self.input_shape = input_shape
        self.W: npt.NDArray[np.float64] | None = None
        self.b: npt.NDArray[np.float64] | None = None

        if input_shape is not None:
            self._initialize_weights(input_shape)

    def _initialize_weights(self, input_shape: tuple[int, int, int]) -> None:
        _, _, C = input_shape
        kh, kw = self.kernel_size
        n_in = kh * kw * C
        # He initialization for ReLU, small random weights otherwise
        scale = np.sqrt(2.0 / n_in) if self.activation == "relu" else 0.01
        # W has shape (filters, kh, kw, C)
        self.W = np.random.randn(self.filters, kh, kw, C) * scale
        # b has shape (1, filters)
        self.b = np.zeros((1, self.filters), dtype=np.float64)

    def _pad_amount(self) -> tuple[int, int]:
        if self.padding != "same":
            return 0, 0
        kh, kw = self.kernel_size
        return kh // 2, kw // 2

    def _pad(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pad_h, pad_w = self._pad_amount()
        if pad_h == 0 and pad_w == 0:
            return X
        return np.pad(X, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))

    def _extract_patches(
        self, Xp: npt.NDArray[np.float64], out_h: int, out_w: int
    ) -> npt.NDArray[np.float64]:
        batch = Xp.shape[0]
        kh, kw = self.kernel_size
        C = Xp.shape[-1]
        patches = np.zeros((batch, out_h, out_w, kh, kw, C))
        for i in range(out_h):
            for j in range(out_w):
                hs, ws = i * self.stride, j * self.stride
                patches[:, i, j] = Xp[:, hs:hs + kh, ws:ws + kw, :]
        return patches

    def forward(
        self, X: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], tuple]:
        # Initialize weights using the input size (H, W, C)
        if self.W is None or self.b is None:
            self.input_shape = X.shape[1:]
            self._initialize_weights(self.input_shape)

        Xp = self._pad(X)
        batch, H, W, C = Xp.shape
        kh, kw = self.kernel_size
        out_h = (H - kh) // self.stride + 1
        out_w = (W - kw) // self.stride + 1

        patches = self._extract_patches(Xp, out_h, out_w)

        # Flatten each patch and each filter, then one matmul does every convolution
        patches_flat = patches.reshape(batch, out_h, out_w, -1)
        W_flat = self.W.reshape(self.filters, -1)
        Z = patches_flat @ W_flat.T + self.b
        A = self.apply_activation(Z)

        cache = (X.shape, patches, Z)
        return A, cache

    def backward(
        self, dA: npt.NDArray[np.float64], cache: tuple
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]:
        X_shape, patches, Z = cache
        batch, out_h, out_w = Z.shape[0], Z.shape[1], Z.shape[2]
        kh, kw = self.kernel_size

        dZ = self.activation_backward(dA, Z)

        # Treat every (batch, out_h, out_w) position as one row for a big matmul,
        # same trick as Dense.backward, just with an im2col-flattened patch as "X".
        N = batch * out_h * out_w
        K = kh * kw * self.W.shape[-1]
        P = patches.reshape(N, K)
        dZ_flat = dZ.reshape(N, self.filters)
        W_flat = self.W.reshape(self.filters, K)

        dW = (dZ_flat.T @ P).reshape(self.W.shape)
        db = np.sum(dZ_flat, axis=0, keepdims=True)

        # Scatter each patch's gradient back to the (padded) input positions it came from,
        # accumulating where patches overlap (stride < kernel size).
        dP = (dZ_flat @ W_flat).reshape(batch, out_h, out_w, kh, kw, self.W.shape[-1])
        pad_h, pad_w = self._pad_amount()
        H_p, W_p = X_shape[1] + 2 * pad_h, X_shape[2] + 2 * pad_w
        dXp = np.zeros((batch, H_p, W_p, X_shape[3]))
        for i in range(out_h):
            for j in range(out_w):
                hs, ws = i * self.stride, j * self.stride
                dXp[:, hs:hs + kh, ws:ws + kw, :] += dP[:, i, j]

        # Undo padding to get the gradient wrt the original (unpadded) input
        if pad_h == 0 and pad_w == 0:
            dA_prev = dXp
        else:
            dA_prev = dXp[:, pad_h:pad_h + X_shape[1], pad_w:pad_w + X_shape[2], :]

        grads = {"dW": dW, "db": db}
        return dA_prev, grads

    def update_params(
        self, grads: dict[str, npt.NDArray[np.float64]], lr: float
    ) -> None:
        if self.W is not None and self.b is not None:
            self.W -= lr * grads["dW"]
            self.b -= lr * grads["db"]


if __name__ == "__main__":
    # Batch of 4 grayscale images, each 28 x 28
    X = np.random.randn(4, 28, 28)

    flatten = Flatten()
    X_flat, _ = flatten.forward(X)

    dense = Dense(units=128, activation="relu")
    A1, _ = dense.forward(X_flat)

    dense2 = Dense(units=64, activation="relu")
    A2, _ = dense2.forward(A1)

    print(f"Input shape:  {X_flat.shape}")
    print(f"W1 shape:     {dense.W.shape}")
    print(f"b1 shape:     {dense.b.shape}")
    print(f"Output 1:     {A1.shape}")
    print(f"W2 shape:     {dense2.W.shape}")
    print(f"b2 shape:     {dense2.b.shape}")
    print(f"Output 2:     {A2.shape}")