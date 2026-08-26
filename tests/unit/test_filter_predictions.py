"""Unit tests for src/lambdas/filter_predictions/handler.py.

Runs against the captured predict_image_label output fixture, no AWS involved.
"""
import json
from pathlib import Path

import pytest

from src.lambdas.filter_predictions.handler import lambda_handler

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def load_fixture(name: str) -> dict:
    # Read a captured Lambda payload from disk
    with open(FIXTURES / name) as f:
        return json.load(f)


def test_lambda_handler_passes_through_when_confidence_meets_threshold():
    # Load a real predict_image_label output: bicycle confidence is 0.969
    event = load_fixture("predictImageLabelOutput.json")

    # Run the handler against it
    result = lambda_handler(event, context=None)

    # Confirm the payload passes through untouched
    assert result["statusCode"] == 200
    assert result["body"] == event["body"]


def test_lambda_handler_raises_when_confidence_below_threshold():
    # Start from the same fixture, then push both class probabilities down
    event = load_fixture("predictImageLabelOutput.json")
    event["body"]["inferences"] = json.dumps([0.3, 0.4])

    # Expect the state machine to stop instead of passing a weak prediction along
    with pytest.raises(Exception, match="THRESHOLD_CONFIDENCE_NOT_MET"):
        lambda_handler(event, context=None)
