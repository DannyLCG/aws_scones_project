"""
Plot inference confidence from the our test data

Data Capture is enabled on the endpoint config,
so every request/response pair is written to s3://<bucket>/data_capture/ as
JSONL — no logging code in the inference path. This script reads those records
and renders the plots that monitor inference confidence

`run_executions.py` only invokes the endpoint enough times to capture both a passing
and a failing inference, which is too few to say anything about the confidence distribution.
--generate sends the whole test set through (N=200)

Note: Since real-time endpoints bill per instance-hour, not per request, 
200 extra inferences cost almost nothing beyond the ~90s they take

Usage:
    uv run python scripts/analyze_capture.py                 # generate 200, then plot
    uv run python scripts/analyze_capture.py --generate 0    # plot existing capture only
    uv run python scripts/analyze_capture.py --wait 600      # allow longer for capture
"""
import argparse
import base64
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import matplotlib
import pandas as pd
from botocore.exceptions import ClientError

# Render without a display, so this works over SSH and in CI
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = REPO_ROOT / "docs" / "inference_confidence.png"

REGION = "us-east-1"
DATA_STACK = "SconesDataStack"
ENDPOINT_STACK = "SconesEndpointStack"

# Must match CONFIDENCE_THRESHOLD in infra/stacks/lambda_stack.py — the gate
# filter_predictions actually enforces, not a number picked for the chart
THRESHOLD = 0.8

log = logging.getLogger("capture")


def configure_logging() -> None:
    """Log to stderr so stdout stays free for the summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    for noisy in ("botocore", "boto3", "urllib3", "s3transfer", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def display_path(path: Path) -> str:
    """Show a repo-relative path when possible, absolute otherwise."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def stack_output(cfn, stack: str, key: str) -> str:
    """Read one named output off a deployed stack."""
    try:
        outputs = cfn.describe_stacks(StackName=stack)["Stacks"][0].get("Outputs", [])
    except ClientError as exc:
        log.error("could not read %s — is it deployed? (%s)", stack, exc)
        raise SystemExit(1) from exc
    for out in outputs:
        if out["OutputKey"] == key:
            return out["OutputValue"]
    log.error("%s has no output %s", stack, key)
    raise SystemExit(1)


def generate_traffic(runtime, s3, endpoint: str, bucket: str, count: int) -> None:
    """Invoke the endpoint on `count` staged test images to populate capture."""
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix="data/test/"
    ):
        keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".png")]
    keys = keys[:count]
    if not keys:
        log.error("no test images staged under s3://%s/data/test/", bucket)
        raise SystemExit(1)

    def invoke(key: str) -> None:
        image = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        runtime.invoke_endpoint(EndpointName=endpoint, ContentType="image/png", Body=image)

    # Modest concurrency: enough to finish quickly, not enough to queue the
    # single-instance endpoint into throttling
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(invoke, keys))
    log.info("sent %d inferences to %s", len(keys), endpoint)


def wait_for_capture(s3, bucket: str, endpoint: str, expected: int, timeout: int) -> list[str]:
    """Poll until captured objects stop arriving, since capture is buffered."""
    prefix = f"data_capture/{endpoint}/"
    deadline = time.monotonic() + timeout
    previous, stable = -1, 0

    while time.monotonic() < deadline:
        keys = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".jsonl")]

        # Stop once the object count has held steady twice in a row and we have
        # at least something to plot — capture lands minutes after the requests
        if keys and len(keys) == previous:
            stable += 1
            if stable >= 2:
                return keys
        else:
            stable = 0
        previous = len(keys)
        log.info("  %d capture file(s) so far; waiting ...", len(keys))
        time.sleep(20)

    if previous > 0:
        log.warning("timed out waiting for capture to settle; using what arrived")
        return keys
    log.error(
        "no capture files under s3://%s/%s after %ds. Capture is buffered and can "
        "take several minutes — retry with --generate 0 --wait 900",
        bucket, prefix, timeout,
    )
    raise SystemExit(1)


def decode(field: dict) -> str:
    """Return a capture field's payload, undoing base64 when it is encoded."""
    data = field["data"]
    if field.get("encoding") == "BASE64":
        return base64.b64decode(data).decode("utf-8")
    return data


def load_records(s3, bucket: str, keys: list[str]) -> pd.DataFrame:
    """Parse captured JSONL into one row per inference."""
    rows = []
    for key in keys:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        for line in body.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                probabilities = json.loads(decode(record["captureData"]["endpointOutput"]))
                timestamp = record["eventMetadata"]["inferenceTime"]
            except (KeyError, json.JSONDecodeError):
                # Skip anything that is not a well-formed inference record
                continue
            rows.append(
                {
                    "timestamp": pd.to_datetime(timestamp),
                    "confidence": max(probabilities),
                    # argmax over [bicycle, motorcycle] gives the routing decision
                    "label": "bicycle" if probabilities[0] >= probabilities[1] else "motorcycle",
                }
            )

    if not rows:
        log.error("capture files held no parseable inference records")
        raise SystemExit(1)

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df["passes_gate"] = df["confidence"] >= THRESHOLD
    return df


def plot(df: pd.DataFrame, out_path: Path) -> None:
    """Render confidence over time beside its distribution."""
    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax_time, ax_dist) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2, 1]}
    )
    passed = "#2A9D8F"
    failed = "#E76F51"

    # Left: every inference in the order it happened. Let the axis follow the
    # data instead of clipping it — the rejected points are the ones worth seeing
    sns.scatterplot(
        data=df, x="timestamp", y="confidence", hue="passes_gate",
        palette={True: passed, False: failed}, s=70, edgecolor="white",
        linewidth=0.6, ax=ax_time, legend=False,
    )
    # Shade the rejection band so the gate reads as a region, not just a line
    ax_time.axhspan(0, THRESHOLD, color=failed, alpha=0.07, zorder=0)
    ax_time.axhline(THRESHOLD, color=failed, linestyle="--", linewidth=1.6)
    ax_time.text(
        0.01, THRESHOLD - 0.02, f"  gate = {THRESHOLD}", color=failed,
        va="top", ha="left", fontsize=12, transform=ax_time.get_yaxis_transform(),
    )
    ax_time.set_title("Observed inferences over time", fontsize=15)
    ax_time.set_xlabel("")
    ax_time.set_ylabel("Top class probability")
    ax_time.tick_params(axis="x", rotation=30)

    # Right: the same values as a distribution, which is what shows how much
    # headroom the model has above the gate rather than just pass/fail counts
    sns.histplot(
        data=df, y="confidence", bins=24, kde=True, color="#264653",
        edgecolor="white", ax=ax_dist,
    )
    ax_dist.axhspan(0, THRESHOLD, color=failed, alpha=0.07, zorder=0)
    ax_dist.axhline(THRESHOLD, color=failed, linestyle="--", linewidth=1.6)
    ax_dist.set_title("Confidence distribution", fontsize=15)
    ax_dist.set_ylabel("")
    ax_dist.set_xlabel("Inferences")

    # Keep both panels on one scale so the gate line lines up across them
    low = min(df["confidence"].min(), THRESHOLD) - 0.05
    for ax in (ax_time, ax_dist):
        ax.set_ylim(max(0, low), 1.02)

    rejected = int((~df["passes_gate"]).sum())
    fig.suptitle(
        f"Endpoint confidence — {len(df)} captured inferences, "
        f"{rejected} below the {THRESHOLD} gate",
        fontsize=17, y=0.98,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("wrote %s", display_path(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--generate", type=int, default=200,
                        help="invoke the endpoint on N test images first (0 to skip)")
    parser.add_argument("--wait", type=int, default=420,
                        help="seconds to wait for capture files to settle")
    args = parser.parse_args()
    configure_logging()

    session = boto3.Session(region_name=REGION)
    cfn = session.client("cloudformation")
    s3 = session.client("s3")
    runtime = session.client("runtime.sagemaker")

    bucket = stack_output(cfn, DATA_STACK, "DataBucketName")
    endpoint = stack_output(cfn, ENDPOINT_STACK, "EndpointName")
    log.info("endpoint: %s", endpoint)

    if args.generate:
        log.info("generating %d inferences to give the distribution some substance", args.generate)
        generate_traffic(runtime, s3, endpoint, bucket, args.generate)

    log.info("waiting for captured records (buffered, usually a few minutes)")
    keys = wait_for_capture(s3, bucket, endpoint, args.generate, args.wait)

    df = load_records(s3, bucket, keys)
    plot(df, OUT_PNG)

    # Summary doubles as the caption material for the README
    rejected = int((~df["passes_gate"]).sum())
    print(f"\ncaptured inferences : {len(df)}")
    print(f"cleared {THRESHOLD} gate     : {len(df) - rejected}")
    print(f"rejected            : {rejected} ({rejected / len(df):.1%})")
    print(f"confidence range    : {df['confidence'].min():.4f} – {df['confidence'].max():.4f}")
    print(f"median confidence   : {df['confidence'].median():.4f}")
    print(f"predicted labels    : {df['label'].value_counts().to_dict()}")
    print(f"\nplot: {display_path(OUT_PNG)}")


if __name__ == "__main__":
    main()
