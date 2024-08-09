import json
import boto3
import base64
#NOTE: JSON strictly requires double quotes for strings.
#When handling JSON data in Python, always use double quotes for keys and string values.

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    bucket = event["s3_bucket"]
    key = event["s3_key"]

    # Download the data from s3 to /tmp/image.png
    filename = '/tmp/image.png'
    s3.download_file(bucket, key, filename)
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        "statusCode": 200,
        "body": {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


