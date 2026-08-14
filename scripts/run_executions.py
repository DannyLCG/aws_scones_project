"""
Find test images that pass and fail the confidence gate, then run both.

We cannot pick a failing image by eye: whether an execution fails depends on
what the model's top probability is, not on what the picture shows. This
script passes sample to the the endpoint until it has one confident and one
unconfident image, then, it and starts a state machine tp docuemnt
the passing and threshold-failing runs in the README

Usage:
    uv run python scripts/run_executions.py              # one passing, one failing
    uv run python scripts/run_executions.py --sample 40  # widen the search
"""
import argparse
import json
import logging
import sys
import time

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
DATA_STACK = "SconesDataStack"
ENDPOINT_STACK = "SconesEndpointStack"
SFN_STACK = "SconesStepFunctionsStack"

# Must match CONFIDENCE_THRESHOLD in infra/stacks/lambda_stack.py
THRESHOLD = 0.8

log = logging.getLogger("executions")


def configure_logging() -> None:
    """Log to stderr so stdout stays clean for the execution ARNs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    for noisy in ("botocore", "boto3", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def stack_output(cfn, stack: str, key: str) -> str:
    """Pull one named output off a deployed stack, or exit explaining why not."""
    try:
        outputs = cfn.describe_stacks(StackName=stack)["Stacks"][0].get("Outputs", [])
    except ClientError as exc:
        log.error("could not read %s: %s", stack, exc)
        raise SystemExit(1) from exc
    for out in outputs:
        if out["OutputKey"] == key:
            return out["OutputValue"]
    log.error("%s has no output %s — is it deployed?", stack, key)
    raise SystemExit(1)


def list_test_keys(s3, bucket: str, limit: int) -> list[str]:
    """List staged test images, newest listing order, capped at `limit`."""
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix="data/test/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".png"):
                keys.append(obj["Key"])
    if not keys:
        log.error("no test images under s3://%s/data/test/", bucket)
        raise SystemExit(1)
    # Spread the sample across the listing instead of taking one alphabetical
    # cluster, which would skew toward a single class
    stride = max(1, len(keys) // limit)
    return keys[::stride][:limit]


def top_confidence(runtime, s3, endpoint: str, bucket: str, key: str) -> float:
    """Return the endpoint's highest class probability for one image."""
    image = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    response = runtime.invoke_endpoint(
        EndpointName=endpoint, ContentType="image/png", Body=image
    )
    return max(json.loads(response["Body"].read().decode("utf-8")))


def find_candidates(runtime, s3, endpoint: str, bucket: str, keys: list[str]):
    """Scan images until one clears the threshold and one falls below it."""
    passing = failing = None
    for key in keys:
        confidence = top_confidence(runtime, s3, endpoint, bucket, key)
        verdict = "PASS" if confidence >= THRESHOLD else "FAIL"
        log.info("  %-38s %.4f  %s", key.rsplit("/", 1)[-1], confidence, verdict)

        if confidence >= THRESHOLD and passing is None:
            passing = (key, confidence)
        elif confidence < THRESHOLD and failing is None:
            failing = (key, confidence)

        # Stop as soon as both examples exist — each probe is a billed inference
        if passing and failing:
            break
    return passing, failing


def start_execution(sfn, arn: str, bucket: str, key: str, label: str) -> str:
    """Start one execution and return its ARN."""
    name = f"{label}-{time.strftime('%Y%m%d-%H%M%S')}"
    response = sfn.start_execution(
        stateMachineArn=arn,
        name=name,
        input=json.dumps({"s3_bucket": bucket, "s3_key": key}),
    )
    return response["executionArn"]


def wait_for(sfn, execution_arn: str) -> str:
    """Poll an execution until it leaves the RUNNING state."""
    while True:
        status = sfn.describe_execution(executionArn=execution_arn)["status"]
        if status != "RUNNING":
            return status
        time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int, default=20,
                        help="how many images to probe (default 20)")
    args = parser.parse_args()
    configure_logging()

    session = boto3.Session(region_name=REGION)
    cfn = session.client("cloudformation")
    s3 = session.client("s3")
    runtime = session.client("runtime.sagemaker")
    sfn = session.client("stepfunctions")

    # Resolve everything off deployed stacks, so no ARN is hardcoded
    bucket = stack_output(cfn, DATA_STACK, "DataBucketName")
    endpoint = stack_output(cfn, ENDPOINT_STACK, "EndpointName")
    state_machine = stack_output(cfn, SFN_STACK, "StateMachineArn")
    log.info("endpoint: %s", endpoint)

    # Probe the endpoint directly to classify images by confidence, which is
    # cheaper and faster than starting executions and hoping for a failure
    log.info("probing up to %d test images for a confident / unconfident pair", args.sample)
    keys = list_test_keys(s3, bucket, args.sample)
    passing, failing = find_candidates(runtime, s3, endpoint, bucket, keys)

    if not passing:
        log.error("no image cleared %.2f — try --sample %d", THRESHOLD, args.sample * 2)
        raise SystemExit(1)
    if not failing:
        log.error(
            "no image fell below %.2f in %d samples; the model is confident on this "
            "set — retry with --sample %d to search wider",
            THRESHOLD, len(keys), args.sample * 2,
        )
        raise SystemExit(1)

    # Run both so the console shows one green and one red execution
    for label, (key, confidence) in (("passing", passing), ("failing", failing)):
        log.info("starting %s execution with %s (confidence %.4f)", label, key, confidence)
        arn = start_execution(sfn, state_machine, bucket, key, label)
        status = wait_for(sfn, arn)
        log.info("  %s execution finished: %s", label, status)
        print(f"{label}: {status}\n  {arn}")


if __name__ == "__main__":
    main()
