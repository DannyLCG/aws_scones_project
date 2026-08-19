"""
Shared test setup

Two things have to happen before the first boto3 client is ever built, or
tests leak out to real AWS depending on collection order:

1. Fake credentials, so a mocking gap fails loudly (bad credentials)
   instead of authenticating against a real account
2. moto must be imported in order to register its interceptor in botocore's builtin
   handlers. Note: Clients built from a session created before that import never
   get the hook, so an unmocked S3 call would hit AWS for real
"""
import os

os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
# Avoid a crash when AWS_PROFILE is set to a profile name that doesn't exist locally
os.environ.pop("AWS_PROFILE", None)

# noqa is load-bearing: without it `ruff --fix` deletes this "unused" import
# and every mocked S3 call silently leaks to real AWS again
import moto  # noqa: E402,F401  (imported for its botocore handler registration)
