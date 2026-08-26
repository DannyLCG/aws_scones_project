"""
CDK entry point.

Stacks are split by lifetime, not by service:

  SconesDataStack          persistent  bucket + SageMaker role       ~$0
  SconesEndpointStack      ephemeral   model + endpoint              $$$ while up
  SconesLambdaStack        ephemeral   the three chained Lambdas     ~$0
  SconesStepFunctionsStack ephemeral   state machine                 ~$0

Deploy the data stack once and leave it up; train once against it; then the
demo cycle is a single `cdk deploy` of the ephemeral stacks followed by
`cdk destroy`. Training itself cannot be a stack: AWS::SageMaker::TrainingJob
is NON_PROVISIONABLE in CloudFormation, so scripts/train.py sits between the
two deploys.
"""
import os
import sys

import aws_cdk as cdk
from stacks.data_stack import DataStack
from stacks.endpoint_stack import EndpointStack
from stacks.lambda_stack import LambdaStack
from stacks.monitor_stack import MonitorStack
from stacks.stepfunctions_stack import StepFunctionsStack


def detect_training_user() -> str | None:
    """
    Return the caller's own IAM user name, or None if there isn't one

    scripts/train.py calls CreateTrainingJob as the developer, so that identity
    needs iam:PassRole on the SageMaker execution role. Deriving the name from
    the current credentials means a bare `cdk deploy SconesDataStack` keeps the
    grant instead of silently deleting it — and anyone cloning this repo gets
    their own identity without editing anything or remembering a flag
    """
    try:
        import boto3

        arn = boto3.client("sts").get_caller_identity()["Arn"]
    except Exception:
        # When no credentials, or no permission to ask: skip the grant rather than
        # breaking synth. Deploys that need it can still pass -c training_user
        return None
    # arn:aws:iam::<account>:user/<name> is the only shape with a user to
    # attach a policy to; assumed roles and SSO sessions have none.
    return arn.rsplit("/", 1)[-1] if ":user/" in arn else None


app = cdk.App()

# Pin the region deliberately instead of inheriting it from the AWS profile:
# the built-in image-classification container is published to only 37 regions,
# and the profile's default (mx-central-1) is not one of them. Override with
# -c region=<name> only for another region that carries the algorithm.
REGION = app.node.try_get_context("region") or "us-east-1"

# ECR accounts that host the built-in algorithm, per AWS's published registry
# list (verified with sagemaker.core.image_uris.retrieve)
ALGORITHM_REGISTRY = {
    "us-east-1": "811284229777",
    "us-west-2": "433757028032",
}
if REGION not in ALGORITHM_REGISTRY:
    sys.exit(
        f"region {REGION} has no image-classification container registered here; "
        f"known: {', '.join(sorted(ALGORITHM_REGISTRY))}"
    )
ALGORITHM_IMAGE = (
    f"{ALGORITHM_REGISTRY[REGION]}.dkr.ecr.{REGION}.amazonaws.com/image-classification:1"
)

env = cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"), region=REGION)

# Stand up the persistent foundation first so its bucket and role can be
# referenced by everything downstream. training_user names the IAM user that
# runs scripts/train.py, which needs iam:PassRole on the execution role.
# -c training_user=<name> overrides; otherwise it is whoever is deploying.
training_user = app.node.try_get_context("training_user") or detect_training_user()

data_stack = DataStack(
    app,
    "SconesDataStack",
    training_user=training_user,
    env=env,
)

# The endpoint can only be described once a training job has produced weights,
# so this stack exists only when the artifact is supplied:
#   cdk deploy SconesEndpointStack -c model_artifact=s3://.../model.tar.gz
model_artifact = app.node.try_get_context("model_artifact") or os.environ.get(
    "SCONES_MODEL_ARTIFACT"
)

endpoint_stack = None
if model_artifact:
    endpoint_stack = EndpointStack(
        app,
        "SconesEndpointStack",
        data_bucket=data_stack.bucket,
        execution_role=data_stack.sagemaker_role,
        model_artifact=model_artifact,
        algorithm_image=ALGORITHM_IMAGE,
        env=env,
    )
    # Take the endpoint name straight off the construct — a real cross-stack
    # reference, so the Lambda's env var and IAM policy can never drift from
    # the endpoint that actually got deployed
    endpoint_name = endpoint_stack.endpoint_name
else:
    # Warn rather than fail, so the data stack can still be deployed alone
    # (which is the prerequisite for training in the first place).
    # predict_image_label refuses to run with this value.
    endpoint_name = "ENDPOINT-NAME-NOT-SET"
    print(
        "WARNING: no model_artifact given, so SconesEndpointStack is skipped and "
        "predict_image_label will not work. Run scripts/train.py, then pass "
        "-c model_artifact=s3://.../model.tar.gz",
        file=sys.stderr,
    )

lambda_stack = LambdaStack(
    app,
    "SconesLambdaStack",
    data_bucket=data_stack.bucket,
    endpoint_name=endpoint_name,
    env=env,
)

# Chain the Lambdas by construct reference, so the state machine can never
# point at a stale ARN the way the committed ASL definition did
StepFunctionsStack(
    app,
    "SconesStepFunctionsStack",
    serialize_image_data=lambda_stack.serialize_image_data,
    predict_image_label=lambda_stack.predict_image_label,
    filter_predictions=lambda_stack.filter_predictions,
    env=env,
)
MonitorStack(app, "SconesMonitorStack", env=env)

app.synth()
