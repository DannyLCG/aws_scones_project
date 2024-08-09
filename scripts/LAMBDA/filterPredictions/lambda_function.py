import json

THRESHOLD = .8

def lambda_handler(event, context):
    # Grab the inferences from the event
    #inferences = event["body"]["inferences"]
    inferences = json.loads(event["body"]["inferences"])
    
    # Filter inference if any label probability is above THRESHOLD
    meets_threshold = any(value >= THRESHOLD for value in inferences)
    #meets_threshold = any(value >= THRESHOLD for row in inferences for value in row)
    #meets_threshold = any(max(row) >= THRESHOLD for row in inferences)
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        return {
        "statusCode": 200,
        "body": event["body"]
    }
    else:
        raise Exception('THRESHOLD_CONFIDENCE_NOT_MET')
    