import numpy as np
import numpy.typing as npt

from layers import Layer, Dense, Conv2D
from functions import (
    mse_loss,
    mse_loss_backward,
    binary_cross_entropy,
    binary_cross_entropy_backward,
    cross_entropy_loss,
    softmax_cross_entropy_backward,
)

# Each loss maps to (loss value)
LOSSES: dict[str, tuple] = {
    "mse": (mse_loss, mse_loss_backward, None),
    "binary_crossentropy": (binary_cross_entropy, binary_cross_entropy_backward, "sigmoid"),
    "categorical_crossentropy": (cross_entropy_loss, softmax_cross_entropy_backward, "softmax"),
}


class Model:
    """Sequential container: runs input through layers in order."""

    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers
        self.loss_name: str | None = None
        self.loss_fn = None
        self.loss_grad_fn = None

    def compile(self, loss: str) -> None:
        """Select the loss function used by fit()."""
        loss_key = loss.lower()
        if loss_key not in LOSSES:
            raise ValueError(f"Unsupported loss '{loss}'. Choose from {list(LOSSES)}.")

        loss_fn, loss_grad_fn, required_activation = LOSSES[loss_key]
        if required_activation is not None:
            last = self.layers[-1]
            if getattr(last, "activation", None) != required_activation:
                raise ValueError(
                    f"Loss '{loss}' requires the final layer's activation to be "
                    f"'{required_activation}', but it is '{getattr(last, 'activation', None)}'."
                )

        self.loss_name = loss_key
        self.loss_fn = loss_fn
        self.loss_grad_fn = loss_grad_fn

    def forward(
        self, X: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], list[tuple]]:
        """Run X through every layer, keeping each layer's cache for backward()."""
        caches = []
        A = X
        for layer in self.layers:
            A, cache = layer.forward(A)
            caches.append(cache)
        return A, caches

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Inference: forward pass only, no cache kept."""
        A = X
        for layer in self.layers:
            A, _ = layer.forward(A)
        return A

    def backward(
        self, dA: npt.NDArray[np.float64], caches: list[tuple], lr: float
    ) -> None:
        """Backprop dA through every layer in reverse, updating weights as we go."""
        for layer, cache in zip(reversed(self.layers), reversed(caches)):
            dA, grads = layer.backward(dA, cache)
            layer.update_params(grads, lr)

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        epochs: int = 10,
        lr: float = 0.01,
        batch_size: int | None = None,
        verbose: bool = True,
    ) -> list[float]:
        """Mini-batch gradient descent. Returns the per-epoch average loss."""
        if self.loss_fn is None:
            raise RuntimeError("Call model.compile(loss=...) before fit().")

        m = X.shape[0]
        batch_size = batch_size or m
        history = []

        for epoch in range(epochs):
            perm = np.random.permutation(m)
            X_shuf, y_shuf = X[perm], y[perm]

            epoch_loss, n_batches = 0.0, 0
            for start in range(0, m, batch_size):
                X_batch = X_shuf[start:start + batch_size]
                y_batch = y_shuf[start:start + batch_size]

                A, caches = self.forward(X_batch)
                epoch_loss += self.loss_fn(A, y_batch)
                n_batches += 1

                dA = self.loss_grad_fn(A, y_batch)
                self.backward(dA, caches, lr)

            avg_loss = epoch_loss / n_batches
            history.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

        return history

    def summary(self, input_shape: tuple[int, ...] | None = None) -> None:
        """Print a layer-by-layer summary with output shapes and param counts.
        layers first, e.g. model.summary(input_shape=(32, 32, 3)).
        """
        if input_shape is not None:
            self.predict(np.zeros((1, *input_shape)))

        header = f"{'Layer':<22}{'Output Shape':<20}{'Params':<12}"
        print(header)
        print("=" * len(header))

        total_params = 0
        for i, layer in enumerate(self.layers):
            name = f"{layer.__class__.__name__}_{i}"
            params = 0
            out_shape = "?  (call summary with input_shape, or predict() once)"
            if isinstance(layer, Dense):
                if layer.W is not None:
                    params = layer.W.size + layer.b.size
                    out_shape = f"(None, {layer.units})"
            elif isinstance(layer, Conv2D):
                if layer.W is not None:
                    params = layer.W.size + layer.b.size
                    out_shape = f"(None, *, *, {layer.filters})"
            else:
                out_shape = "(None, *)"
            total_params += params
            print(f"{name:<22}{out_shape:<20}{params:<12}")

        print("=" * len(header))
        print(f"Total params: {total_params}")