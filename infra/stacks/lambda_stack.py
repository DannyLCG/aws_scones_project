"""The three chained Lambdas (serialize -> predict -> filter) and their
execution roles. Replaces manual console deploys of scripts/LAMBDA/*.
"""
from aws_cdk import Stack
from constructs import Construct


class LambdaStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # TODO: serializeImageData function (src/lambdas/serialize_image_data), s3:GetObject on data bucket
        # TODO: predictImageLabel function (src/lambdas/predict_image_label), sagemaker:InvokeEndpoint + ENDPOINT_NAME env var
        # TODO: filterPredictions function (src/lambdas/filter_predictions), CONFIDENCE_THRESHOLD env var
