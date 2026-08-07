"""S3 bucket for staged train/test data, .lst manifests, and the Model
Monitor data capture destination — one bucket, mirroring the prefixes
(data/, data_capture/) the original odlo-scones-bucket used.
"""
from aws_cdk import RemovalPolicy, Stack
from aws_cdk import aws_s3 as s3
from constructs import Construct


class DataStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Let CDK generate a globally-unique name rather than hardcoding one:
        # nothing downstream depends on the literal bucket name, since Lambda
        # code reads s3_bucket from the event payload, not from config
        self.bucket = s3.Bucket(
            self,
            "DataBucket",
            # Treat this as disposable, matching the "ephemeral infra" story:
            # the data can always be re-synced from data/train and data/test
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )
