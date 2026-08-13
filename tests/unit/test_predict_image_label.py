"""
Unit tests for src/lambdas/predict_image_label/handler.py.

moto has no useful canned-response support for runtime.sagemaker, so the
endpoint is stubbed with botocore's Stubber instead: it asserts the exact
API params the handler sends and returns a scripted response body.
"""
import base64
import io
import json
import os
from pathlib import Path

# Set the endpoint name before import: the handler reads it at module level,
# mirroring how Lambda resolves env vars once per cold start.
# Fake AWS credentials come from tests/conftest.py.
os.environ["ENDPOINT_NAME"] = "test-image-classification-endpoint"

import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber

from src.lambdas.predict_image_label import handler as predict_handler

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def load_fixture(name: str) -> dict:
    # Read a captured Lambda payload from disk
    with open(FIXTURES / name) as f:
        return json.load(f)


def streaming_body(payload: str) -> StreamingBody:
    """Wrap a string the way invoke_endpoint returns it, so the handler's
    response["Body"].read().decode() path is exercised for real"""
    raw = payload.encode("utf-8")
    return StreamingBody(io.BytesIO(raw), len(raw))


@pytest.fixture
def stub():
    # Intercept calls on the handler's module-level client
    with Stubber(predict_handler.runtime) as stubber:
        yield stubber


def test_lambda_handler_attaches_endpoint_inferences(stub):
    # Load the payload serialize_image_data hands over
    event = load_fixture("imageSerializerOutput.json")
    inferences = "[0.03067375347018242, 0.9693263173103333]"

    # Expect the decoded PNG bytes to reach the endpoint, not the base64 string
    stub.add_response(
        "invoke_endpoint",
        {"Body": streaming_body(inferences)},
        {
            "EndpointName": "test-image-classification-endpoint",
            "ContentType": "image/png",
            "Body": base64.b64decode(event["body"]["image_data"]),
        },
    )

    result = predict_handler.lambda_handler(event, context=None)

    # Confirm the endpoint's raw JSON string is attached verbatim, since
    # filter_predictions is what parses it back into a list
    assert result["statusCode"] == 200
    assert result["body"]["inferences"] == inferences
    assert json.loads(result["body"]["inferences"]) == [
        0.03067375347018242,
        0.9693263173103333,
    ]

    # Confirm the rest of the payload rides through untouched
    assert result["body"]["s3_bucket"] == event["body"]["s3_bucket"]
    assert result["body"]["s3_key"] == event["body"]["s3_key"]

    # Confirm no expected call went unused
    stub.assert_no_pending_responses()


def test_lambda_handler_output_feeds_filter_predictions(stub):
    # Prove the two Lambdas actually compose: this output is what
    # filter_predictions parses, so a shape mismatch fails here, not in prod
    from src.lambdas.filter_predictions.handler import lambda_handler as filter_handler

    event = load_fixture("imageSerializerOutput.json")
    stub.add_response(
        "invoke_endpoint",
        {"Body": streaming_body("[0.0306, 0.9693]")},
        {
            "EndpointName": "test-image-classification-endpoint",
            "ContentType": "image/png",
            "Body": base64.b64decode(event["body"]["image_data"]),
        },
    )

    predicted = predict_handler.lambda_handler(event, context=None)

    # Feed it straight into the next Lambda; 0.9693 clears the 0.8 gate
    filtered = filter_handler(predicted, context=None)
    assert filtered["statusCode"] == 200


def test_lambda_handler_requires_an_endpoint_name(monkeypatch):
    """Guard against deploying with the env var unset: better a clear error
    than an opaque SageMaker ValidationException on an empty EndpointName"""
    monkeypatch.setattr(predict_handler, "ENDPOINT_NAME", "")

    with pytest.raises(RuntimeError, match="ENDPOINT_NAME is not set"):
        predict_handler.lambda_handler(load_fixture("imageSerializerOutput.json"), None)
