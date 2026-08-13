"""
The S3 bucket and the SageMaker execution role.

The bucket holds staged train/test
data, the .lst manifests, the trained model.tar.gz, and the Model Monitor
capture destination; the role is what training jobs and the endpoint assume.

This stack can be eployed once and leave it up since it costs almost nothing, and destroying it
would delete the model artifact, forcing a retrain before the next demo.
"""
from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from constructs import Construct


class DataStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        training_user: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Let CDK generate a globally-unique name rather than hardcoding one:
        # nothing downstream depends on the literal bucket name, since Lambda
        # code reads s3_bucket from the event payload, not from config
        self.bucket = s3.Bucket(
            self,
            "DataBucket",
            # Keep DESTROY so teardown is still one command if we need it
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # One role assumed by both the training job and the endpoint. SageMaker
        # requires this to exist before a job starts
        self.sagemaker_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Execution role for Scones training jobs and endpoint",
        )

        # Read the training data and write back model.tar.gz + captured traffic
        self.bucket.grant_read_write(self.sagemaker_role)

        # Pull the built-in image-classification containerfrom ECR
        # GetAuthorizationToken has no resource scope, so it must be granted on "*".
        self.sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                actions=["ecr:GetAuthorizationToken"],
                resources=["*"],
            )
        )
        self.sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                ],
                resources=["*"],
            )
        )

        # Ship training and inference logs plus algorithm metrics to CloudWatch
        self.sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                    "cloudwatch:PutMetricData",
                ],
                resources=["*"],
            )
        )

        # scripts/train.py calls CreateTrainingJob directly as the developer,
        # not through CloudFormation, so that identity needs iam:PassRole to
        # hand this role to SageMaker. Grant it here rather than by hand, so
        # the permission is version-controlled and disappears with the stack.
        #
        # This is safe to grant where broader IAM would not be: it names one
        # role ARN (no wildcard), and the condition means the role can only be
        # passed to SageMaker. The role itself only reads/writes this bucket,
        # so passing it confers nothing the developer cannot already do.
        if training_user:
            iam.ManagedPolicy(
                self,
                "PassTrainingRolePolicy",
                description="Allow the training user to pass the SageMaker execution role",
                users=[iam.User.from_user_name(self, "TrainingUser", training_user)],
                statements=[
                    iam.PolicyStatement(
                        actions=["iam:PassRole"],
                        resources=[self.sagemaker_role.role_arn],
                        conditions={
                            "StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}
                        },
                    )
                ],
            )

        # Surface both values so scripts/train.py can discover them from the
        # deployed stack instead of taking hardcoded arguments
        CfnOutput(self, "DataBucketName", value=self.bucket.bucket_name)
        CfnOutput(self, "SageMakerRoleArn", value=self.sagemaker_role.role_arn)
