"""Fit the SageMaker image-classification estimator and deploy a real-time
endpoint with Data Capture enabled. Ports starter.ipynb cells 56-73.
"""


def build_estimator(role: str, output_path: str, instance_type: str = "ml.p3.2xlarge"):
    """Construct the image-classification Estimator (image_shape=3,32,32, num_classes=2)."""
    raise NotImplementedError


def deploy_endpoint(estimator, instance_type: str = "ml.m5.xlarge", capture_uri: str | None = None):
    """Deploy `estimator` to a real-time endpoint with DataCaptureConfig attached."""
    raise NotImplementedError
