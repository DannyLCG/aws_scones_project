"""The three chained Lambdas (serialize -> predict -> filter) and their
execution roles. Replaces manual console deploys of scripts/LAMBDA/*.
"""
from pathlib import Path

from aws_cdk import Duration, Stack
from aws_cdk import aws_iam as iam
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
        endpoint_name: str,
        confidence_threshold: str = "0.8",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

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

        # Deploy predict_image_label, pointed at the endpoint the notebook stood up
        self.predict_image_label = _lambda.Function(
            self,
            "PredictImageLabel",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="handler.lambda_handler",
            code=_lambda.Code.from_asset(str(LAMBDA_SRC / "predict_image_label")),
            environment={
                # Inject the endpoint rather than hardcoding it, so re-training
                # is a redeploy with new context, not a code edit
                "ENDPOINT_NAME": endpoint_name,
            },
            # Raise the 3s default: a cold endpoint can take several seconds to answer
            timeout=Duration.seconds(30),
        )
        # Allow invoking only this endpoint, not every endpoint in the account.
        # The endpoint lives outside CDK (created by the notebook), so its ARN is
        # built from the name instead of read off a construct.
        self.predict_image_label.add_to_role_policy(
            iam.PolicyStatement(
                actions=["sagemaker:InvokeEndpoint"],
                resources=[
                    self.format_arn(
                        service="sagemaker",
                        resource="endpoint",
                        resource_name=endpoint_name,
                    )
                ],
            )
        )

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
