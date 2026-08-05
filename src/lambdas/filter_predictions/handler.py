"""Lambda 3: gate the Step Function execution on inference confidence.
Current logic: scripts/LAMBDA/filterPredictions/lambda_function.py

Note: THRESHOLD was hardcoded there (THRESHOLD = .8); here it should come
from the CONFIDENCE_THRESHOLD env var set by infra/stacks/lambda_stack.py.
"""
import os

THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.8"))


def lambda_handler(event: dict, context) -> dict:
    raise NotImplementedError
