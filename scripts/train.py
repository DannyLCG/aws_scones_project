"""
Stage the dataset in S3 and run the image-classification training job.

This script is plain boto3 rather than SageMaker SDK given: 
1) SDK v3 removed the Estimator, Model and Predictor API's, which is what the original project used.
2) A training job is a one-shot task, not a resource, so there is nothing for the
SDK's object model to manage. boto3's create_training_job maps 1:1 onto
the same API vocabulary the CDK stacks use. It also does not break when the
SDK ships a major version 

Note: Training cannot live in CDK at all: AWS::SageMaker::TrainingJob is registered
with CloudFormation but marked NON_PROVISIONABLE, so no stack can create one.
That is why this step sits between `cdk deploy SconesDataStack` (which makes
the bucket and role) and `cdk deploy SconesEndpointStack` (which needs the
model.tar.gz this produces).

Usage:
    uv run python scripts/train.py                 # train, wait, print artifact URI
    uv run python scripts/train.py -v              # include boto3/debug detail
"""
import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DATA_STACK = "SconesDataStack"

# The built-in algorithm only exists in a subset of regions, NOT in mx-central-1 
REGION = "us-east-1"

# The algorithm supports only P2/P3/G4dn/G5 for training — CPU instances are
# inference-only. Of those, g5.xlarge is the one with quota on this account
# (g4dn is 0, p2/p3 unavailable), so it is effectively the only choice.
INSTANCE_TYPE = "ml.g5.xlarge"

log = logging.getLogger("train")


def configure_logging(verbose: bool) -> None:
    """Send timestamped logs to stderr, keeping stdout free for results."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Mute the AWS libraries' own chatter unless --verbose asked for it
    for noisy in ("botocore", "boto3", "urllib3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@contextmanager
def step(name: str):
    """Name a stage so any failure inside it reports where things broke."""
    log.info("%s ...", name)
    started = time.monotonic()
    try:
        yield
    except SystemExit:
        # fail() already logged the specific reason; just attach the stage
        log.error("STEP FAILED -> %s", name)
        raise
    except Exception as exc:
        # Turn an unexpected exception into the same "which step" report
        log.error("STEP FAILED -> %s (%s: %s)", name, type(exc).__name__, exc)
        log.debug("traceback follows", exc_info=True)
        raise SystemExit(1) from exc
    log.info("%s — done in %.1fs", name, time.monotonic() - started)


def fail(message: str) -> None:
    """Stop with a specific reason; the enclosing step() adds the stage name."""
    log.error(message)
    raise SystemExit(1)


def stack_outputs(cfn, stack_name: str) -> dict[str, str]:
    """Return the deployed stack's outputs, keyed by output name."""
    # Read the bucket and role straight off the deployed stack, so nothing
    # here hardcodes a name that CDK generated
    try:
        stacks = cfn.describe_stacks(StackName=stack_name)["Stacks"]
    except ClientError as exc:
        fail(
            f"could not read {stack_name}: {exc}\n"
            f"Deploy it first:  cd infra && cdk deploy {stack_name}"
        )
    return {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}


def upload_split(s3, bucket: str, split: str) -> int:
    """Mirror data/<split>/*.png to S3, returning how many images exist."""
    images = sorted(DATA_DIR.joinpath(split).glob("*.png"))
    if not images:
        fail(f"no images in data/{split}")

    # Skip re-uploading on repeat runs: the images never change, and 1,200
    # PUTs per attempt is pure waste while iterating on the job config
    prefix = f"data/{split}/"
    existing = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("KeyCount", 0)
    if existing >= len(images):
        log.info("  data/%s/: %d objects already staged, skipping", split, existing)
        return len(images)

    # Upload in parallel — 1,200 tiny files are latency-bound, not bandwidth-bound
    def put(path: Path) -> None:
        s3.upload_file(str(path), bucket, f"{prefix}{path.name}")

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(put, images))
    log.info("  data/%s/: uploaded %d images", split, len(images))
    return len(images)


def upload_manifests(s3, bucket: str) -> None:
    """Upload the .lst files that tell the algorithm each image's label."""
    for split in ("train", "test"):
        manifest = DATA_DIR / f"{split}.lst"
        # Refuse to train on missing labels rather than let SageMaker fail later
        if not manifest.exists():
            fail(
                f"missing {manifest.relative_to(REPO_ROOT)} — "
                "run:  uv run python scripts/prepare_data.py"
            )
        s3.upload_file(str(manifest), bucket, f"data/{split}.lst")
    log.info("  manifests: uploaded train.lst and test.lst")


def channel(s3_uri: str, content_type: str = "application/x-image") -> dict:
    """Build one InputDataConfig channel pointing at an S3 prefix."""
    # Every channel is a plain S3 prefix in File mode, matching the notebook
    return {
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": s3_uri,
                "S3DataDistributionType": "FullyReplicated",
            }
        },
        "ContentType": content_type,
    }


def resolve_algorithm_image() -> str:
    """Look up the built-in algorithm's container URI for this region."""
    # Import here rather than at module load: the SDK is slow to import and is
    # needed only for this one lookup. Resolving beats hardcoding an ECR
    # account number, which differs per region.
    from sagemaker.core import image_uris  # noqa: PLC0415

    return image_uris.retrieve("image-classification", REGION)


def submit_training_job(sm, job_name: str, bucket: str, role_arn: str,
                        algo_image: str, n_train: int) -> None:
    """Submit the training job that writes model.tar.gz back to the bucket."""
    sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": algo_image,
            "TrainingInputMode": "File",
        },
        RoleArn=role_arn,
        # Derive num_training_samples rather than hardcoding it: it must match
        # the number of images actually staged or the algorithm miscounts epochs
        HyperParameters={
            "image_shape": "3,32,32",
            "num_classes": "2",
            "num_training_samples": str(n_train),
        },
        # Feed all four channels the algorithm's image format requires: the
        # images themselves, plus a .lst per split carrying the labels
        InputDataConfig=[
            {"ChannelName": "train", **channel(f"s3://{bucket}/data/train/")},
            {"ChannelName": "validation", **channel(f"s3://{bucket}/data/test/")},
            {"ChannelName": "train_lst", **channel(f"s3://{bucket}/data/train.lst")},
            {"ChannelName": "validation_lst", **channel(f"s3://{bucket}/data/test.lst")},
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/models/"},
        ResourceConfig={
            "InstanceType": INSTANCE_TYPE,
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        # Cap the run so a hung job cannot quietly bill GPU hours; the notebook's
        # 360000s (100h) ceiling was far more than this 1,000-image job needs
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
    )


def wait_for_job(sm, job_name: str) -> dict:
    """Block until the job settles, then return its description."""
    # Poll every 30s for up to an hour, matching MaxRuntimeInSeconds
    sm.get_waiter("training_job_completed_or_stopped").wait(
        TrainingJobName=job_name,
        WaiterConfig={"Delay": 30, "MaxAttempts": 120},
    )

    # The waiter returns for Stopped and Failed too, so check the real status
    job = sm.describe_training_job(TrainingJobName=job_name)
    status = job["TrainingJobStatus"]
    if status != "Completed":
        fail(
            f"training job {status}: {job.get('FailureReason', 'no reason given')}\n"
            f"Logs: aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix "
            f"{job_name} --region {REGION}"
        )
    return job


def main() -> None:
    """Run the full path: read stack -> stage data -> train -> report artifact."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-wait", action="store_true", help="don't block on completion")
    parser.add_argument("-v", "--verbose", action="store_true", help="include debug detail")
    args = parser.parse_args()

    configure_logging(args.verbose)

    session = boto3.Session(region_name=REGION)
    cfn = session.client("cloudformation")
    s3 = session.client("s3")
    sm = session.client("sagemaker")

    # Discover where to read and write, and which role the job runs as
    with step(f"reading {DATA_STACK} outputs"):
        outputs = stack_outputs(cfn, DATA_STACK)
        bucket = outputs.get("DataBucketName")
        role_arn = outputs.get("SageMakerRoleArn")
        if not bucket or not role_arn:
            fail(f"{DATA_STACK} is missing DataBucketName/SageMakerRoleArn outputs")
        log.info("  bucket: %s", bucket)
        log.info("  role:   %s", role_arn)

    # Put the images and their labels where the training job will look
    with step("staging dataset in S3"):
        n_train = upload_split(s3, bucket, "train")
        upload_split(s3, bucket, "test")
        upload_manifests(s3, bucket)

    with step("resolving algorithm container"):
        algo_image = resolve_algorithm_image()
        log.info("  image: %s", algo_image)

    # Name the job after the clock so repeat runs never collide
    job_name = f"scones-image-classification-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    with step(f"submitting training job {job_name}"):
        log.info("  instance: %s", INSTANCE_TYPE)
        log.info("  samples:  %d", n_train)
        submit_training_job(sm, job_name, bucket, role_arn, algo_image, n_train)

    # Let the caller skip the wait when they only want the job started
    if args.no_wait:
        log.info("submitted; not waiting")
        print(
            f"\nPoll with:\n  aws sagemaker describe-training-job "
            f"--training-job-name {job_name} --region {REGION}"
        )
        return

    with step("waiting for training to finish"):
        job = wait_for_job(sm, job_name)

    # Report the artifact on stdout so it can be piped or copied into the deploy
    artifact = job["ModelArtifacts"]["S3ModelArtifacts"]
    metrics = {m["MetricName"]: m["Value"] for m in job.get("FinalMetricDataList", [])}
    if metrics:
        log.info("final metrics: %s", ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    print(f"\nmodel artifact:\n  {artifact}\n")
    print("Deploy the endpoint with it:")
    print(f"  cd infra && cdk deploy SconesEndpointStack -c model_artifact={artifact}")


if __name__ == "__main__":
    main()
