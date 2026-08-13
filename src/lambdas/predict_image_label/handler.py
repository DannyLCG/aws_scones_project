"""
Lambda 2: invoke the SageMaker endpoint and predict probabilities.

Ported from scripts/LAMBDA/predictImageLabel/lambda_function.py — the
hardcoded ENDPOINT now reads from ENDPOINT_NAME (set in
infra/stacks/lambda_stack.py), so the same artifact can point at a
re-trained endpoint without a code change.
"""
import base64
import os

import boto3

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "")

runtime = boto3.client("runtime.sagemaker")


def lambda_handler(event: dict, context) -> dict:
    # Fail loudly at invocation rather than sending an empty EndpointName to SageMaker
    if not ENDPOINT_NAME:
        raise RuntimeError("ENDPOINT_NAME is not set")

    # Undo the base64 encoding serialize_image_data applied, back to raw PNG bytes
    image = base64.b64decode(event["body"]["image_data"])

    # Send the image to the endpoint for classification 
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="image/png",
        Body=image,
    )

    # Attach the probabilities as the raw JSON string the endpoint returns
    event["body"]["inferences"] = response["Body"].read().decode("utf-8")

    # Hand the enriched payload off to filter_predictions
    return {
        "statusCode": 200,
        "body": event["body"],
    }
