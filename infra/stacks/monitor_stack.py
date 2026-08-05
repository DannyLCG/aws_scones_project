"""SageMaker Model Monitor schedule on the deployed endpoint, plus a
CloudWatch alarm on low-confidence rate / data drift.
"""
from aws_cdk import Stack
from constructs import Construct


class MonitorStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # TODO: MonitoringSchedule on the endpoint's captured data
        # TODO: CloudWatch alarm on low-confidence rate / drift metric
