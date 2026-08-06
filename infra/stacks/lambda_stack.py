"""The three chained Lambdas (serialize -> predict -> filter) and their
execution roles. Replaces manual console deploys of scripts/LAMBDA/*.
"""
from pathlib import Path

from aws_cdk import Stack
from aws_cdk import aws_lambda as _lambda
from constructs import Construct

# Resolve src/lambdas/ relative to this file, not the directory cdk synth runs from
REPO_ROOT = Path(__file__).resolve().parents[2]
LAMBDA_SRC = REPO_ROOT / "src" / "lambdas"


class LambdaStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        confidence_threshold: str = "0.8",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # TODO: serializeImageData function (src/lambdas/serialize_image_data), s3:GetObject on data bucket
        # TODO: predictImageLabel function (src/lambdas/predict_image_label), sagemaker:InvokeEndpoint + ENDPOINT_NAME env var

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
