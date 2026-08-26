"""Lambda 3: gate the Step Function execution on inference confidence.

Ported from scripts/LAMBDA/filterPredictions/lambda_function.py — the
hardcoded THRESHOLD there now reads from CONFIDENCE_THRESHOLD (set in
infra/stacks/lambda_stack.py).
"""
import json
import os

THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.8"))

def lambda_handler(event: dict, context) -> dict:
    # Decode the inferences from the event (produced by predict_image_label)
    inferences = json.loads(event["body"]["inferences"])

    # Stop the state machine if no inference is above our threshols
    meets_threshold = any(value >= THRESHOLD for value in inferences)
    if not meets_threshold:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    # Otherwise pass the unchanged payload
    return {
        "statusCode": 200,
        "body": event["body"],
    }
