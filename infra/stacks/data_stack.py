"""S3 buckets: staged train/test images + .lst manifests, model artifacts,
and the Model Monitor data capture destination.
"""
from aws_cdk import Stack
from constructs import Construct


class DataStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # TODO: S3 bucket for staged train/test images + .lst manifests
        # TODO: S3 prefix/bucket for Model Monitor data capture (data_capture_config)
