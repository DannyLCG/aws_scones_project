"""Filter CIFAR-100 to bicycle/motorcycle classes and write PNGs + .lst
manifests. Ports starter.ipynb cells 29-45 and 51-53.
"""


def filter_dataset(dataset: dict, filter_labels: list[int]) -> dict:
    """Keep only rows whose fine label is in `filter_labels`."""
    raise NotImplementedError


def save_images(dataset: dict, dir: str) -> None:
    """Reshape each row's pixel data to 32x32x3 and save it as a PNG in `dir`."""
    raise NotImplementedError


def to_metadata_file(df, prefix: str) -> None:
    """Write a SageMaker image-classification .lst manifest for `df`."""
    raise NotImplementedError
