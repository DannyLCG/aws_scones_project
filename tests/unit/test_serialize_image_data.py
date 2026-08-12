"""
Unit tests for the 'serialize_image_data' lambda function
`src/lambdas/serialize_image_data/handler.py.`

Runs against a mocked S3 bucket via the moto lib
"""
import base64
import json
from pathlib import Path

# Fake AWS credentials and moto's botocore hook come from tests/conftest.py
import boto3
from moto import mock_aws

from src.lambdas.serialize_image_data.handler import lambda_handler

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_fixture(name: str) -> dict:
    # Read a captured Lambda payload from disk
    with open(FIXTURES / name) as f:
        return json.load(f)


@mock_aws
def test_lambda_handler_downloads_and_encodes_the_target_image():
    # Load the event that kicks off the state machine
    event = load_fixture("s3_input_image.json")

    # Recreate the source bucket and upload the same real test image it points to
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=event["s3_bucket"])
    image_path = REPO_ROOT / "data" / "test" / "bicycle_s_000513.png"
    s3.upload_file(str(image_path), event["s3_bucket"], event["s3_key"])

    # Run the handler against the mocked bucket
    result = lambda_handler(event, context=None)

    # Confirm the bucket/key are echoed back and inferences start empty
    assert result["statusCode"] == 200
    assert result["body"]["s3_bucket"] == event["s3_bucket"]
    assert result["body"]["s3_key"] == event["s3_key"]
    assert result["body"]["inferences"] == []

    # Confirm the image wa base64-encoded and is not corrupted
    expected = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    assert result["body"]["image_data"] == expected
