"""Lambda 1: download the target image from S3 and base64-encode it.

Ported from scripts/LAMBDA/serializeImageData/lambda_function.py — image_data
is decoded to a UTF-8 string here (raw bytes aren't JSON-serializable) so the
payload can flow through Step Functions as-is.
"""
import base64
import boto3

s3 = boto3.client("s3")

# Define the lambda function
def lambda_handler(event: dict, context) -> dict:
    """Lambda function to serialize target data from S3"""
    bucket = event["s3_bucket"]
    key = event["s3_key"]

    # Pull the target image from S3 down to /tmp, 
    # the only writable path in Lambda
    filename = "/tmp/image.png"
    s3.download_file(bucket, key, filename)

    # Read it back and base64-encode it for the Step Function payload
    with open(filename, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Pass the image data back to the Step Function, with an empty slot for its inferences
    # This is handed off to predict_image_label
    return {
        "statusCode": 200,
        "body": {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": [],
        },
    }
