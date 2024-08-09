import json
import base64
import boto3
#NOTE: JSON strictly requires double quotes for strings.
#When handling JSON data in Python, always use double quotes for keys and string values.

ENDPOINT = 'image-classification-2024-08-07-15-52-22-460' #Different from each trained model
runtime = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    # Decode the image from the last lambda event
    image = base64.b64decode(event["body"]["image_data"])

    # Make a prediction from our endpoint
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType='image/png', Body=image)

    event["body"]["inferences"] = response["Body"].read().decode('utf-8')

    return {
        "statusCode": 200,
        "body": event["body"]
    }