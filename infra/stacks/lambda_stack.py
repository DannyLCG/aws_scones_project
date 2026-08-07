"""The three chained Lambdas (serialize -> predict -> filter) and their
execution roles. Replaces manual console deploys of scripts/LAMBDA/*.
"""
from pathlib import Path

from aws_cdk import Stack
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_s3 as s3
from constructs import Construct

# Resolve src/lambdas/ relative to this file, not the directory cdk synth runs from
REPO_ROOT = Path(__file__).resolve().parents[2]
LAMBDA_SRC = REPO_ROOT / "src" / "lambdas"


class LambdaStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        data_bucket: s3.IBucket,
        confidence_threshold: str = "0.8",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # TODO: predictImageLabel function (src/lambdas/predict_image_label), sagemaker:InvokeEndpoint + ENDPOINT_NAME env var

        # Deploy serialize_image_data as-is: stdlib + boto3 only, nothing to bundle
        self.serialize_image_data = _lambda.Function(
            self,
            "SerializeImageData",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="handler.lambda_handler",
            code=_lambda.Code.from_asset(str(LAMBDA_SRC / "serialize_image_data")),
        )
        # Grant read on exactly this bucket, not blanket S3 access
        data_bucket.grant_read(self.serialize_image_data)

        # Deploy filter_predictions as-is: pure Python, nothing to bundle
        self.filter_predictions = _lambda.Function(
            self,
            "FilterPredictions",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="handler.lambda_handler",
            code=_lambda.Code.from_asset(str(LAMBDA_SRC / "filter_predictions")),
            environment={
                "CONFIDENCE_THRESHOLD": confidence_threshold,
            },
        )
        # Skip extra IAM grants: this function makes no AWS calls of its own,
        # so the default CloudWatch Logs execution role CDK attaches is enough
