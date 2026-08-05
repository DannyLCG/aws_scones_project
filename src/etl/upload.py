"""Sync staged images and manifests to S3. Ports starter.ipynb cells 47-55."""


def sync_to_s3(local_dir: str, bucket: str, prefix: str) -> None:
    """Sync `local_dir` to s3://{bucket}/{prefix}/."""
    raise NotImplementedError


def upload_manifest(bucket: str, key: str, local_path: str) -> None:
    """Upload a single .lst manifest file to S3."""
    raise NotImplementedError
