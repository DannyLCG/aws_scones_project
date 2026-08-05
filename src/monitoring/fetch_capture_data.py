"""Pull Model Monitor's captured JSONLines endpoint I/O down from S3.
Ports starter.ipynb cells 82-88.
"""


def download_capture_data(capture_s3_uri: str, local_dir: str = "captured_data") -> list[str]:
    """Download all capture files under `capture_s3_uri` into `local_dir`."""
    raise NotImplementedError


def load_capture_records(local_dir: str) -> list[dict]:
    """Parse every JSONLines capture file in `local_dir` into records."""
    raise NotImplementedError
