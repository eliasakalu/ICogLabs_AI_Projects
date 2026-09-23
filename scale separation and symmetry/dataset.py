import os
import tarfile
import urllib.request
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from PIL import Image


class DownloadProgressBar(tqdm):
    """To display download with tdqm bar."""
    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _load_images_from_folder(
    folder: str,
    classes: list[str],
    img_size: tuple[int, int] = (160, 160),
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """find directory and load all images as a NumPy array."""
    class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
    X, y = [], []
    for cls in sorted(classes):
        cls_dir = os.path.join(folder, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            fpath = os.path.join(cls_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB").resize(img_size)
            except Exception:
                continue
            X.append(np.asarray(img, dtype=np.uint8))
            y.append(class_to_idx[cls])
    return np.stack(X), np.array(y, dtype=np.int64)


def load_imagenette(
    data_dir: str = "./data",
    img_size: int = 160,
    subset_classes: list[str] | None = None,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.int64],
    npt.NDArray[np.float32],
    npt.NDArray[np.int64],
]:
    """Download and load ImageNette (10-class ImageNet subset) into NumPy arrays."""

    # Pick 160px or 320px image size
    variant = "320" if img_size > 160 else "160"
    url = f"https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-{variant}.tgz"
    filepath = os.path.join(data_dir, f"imagenette2-{variant}.tgz")
    extracted_dir = os.path.join(data_dir, f"imagenette2-{variant}")

    os.makedirs(data_dir, exist_ok=True)

    # Download if data doesnt exist
    if not os.path.exists(filepath):
        print(f"Downloading ImageNette-{variant}...")
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=filepath
        ) as t:
            urllib.request.urlretrieve(url, filepath, reporthook=t.update_to)

    # Extract tar file
    if not os.path.exists(extracted_dir):
        print("Extracting ImageNette...")
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=data_dir)

    # Discover classes (folders inside train/)
    train_dir = os.path.join(extracted_dir, "train")
    val_dir   = os.path.join(extracted_dir, "val")
    all_classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    classes = subset_classes if subset_classes is not None else all_classes

    # Load images into arrays
    print(f"Loading train images ({len(classes)} classes)...")
    x_train, y_train = _load_images_from_folder(
        train_dir, classes, (img_size, img_size)
    )
    print("Loading val images...")
    x_test, y_test = _load_images_from_folder(
        val_dir, classes, (img_size, img_size)
    )

    # Normalize to [0, 1] and transpose to (Batch, H, W, C)
    x_train = (x_train).astype(np.float32) / 255.0
    x_test  = (x_test ).astype(np.float32) / 255.0

    return x_train, y_train, x_test, y_test


def to_one_hot(
    y: npt.NDArray[np.int64], num_classes: int = 10
) -> npt.NDArray[np.float64]:
    """Convert integer class indices to one-hot binary matrix."""
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes), dtype=np.float64)
    one_hot[np.arange(m), y] = 1.0
    return one_hot


if __name__ == "__main__":
    x_tr, y_tr, x_te, y_te = load_imagenette(img_size=160)
    print(f"X_train shape: {x_tr.shape}")
    print(f"y_train shape: {y_tr.shape}")
    print(f"X_test shape:  {x_te.shape}")
    print(f"y_test shape:  {y_te.shape}")