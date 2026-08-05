"""Lambda 2: invoke the SageMaker endpoint and attach inference probabilities.
Current logic: scripts/LAMBDA/predictImageLabel/lambda_function.py

Note: the endpoint name was hardcoded there (ENDPOINT = '...'); here it
should come from the ENDPOINT_NAME env var set by infra/stacks/lambda_stack.py.
"""
import os

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "")


def lambda_handler(event: dict, context) -> dict:
    raise NotImplementedError
