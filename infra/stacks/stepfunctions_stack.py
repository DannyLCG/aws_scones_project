"""State machine chaining the three Lambdas. Replaces the hand-edited
SconestateMachine_definition.json (which hardcodes account ID / function ARNs).
"""
from aws_cdk import Stack
from constructs import Construct


class StepFunctionsStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # TODO: sfn.Chain of LambdaInvoke tasks (serialize -> predict -> filter),
        # retries matching the current ASL definition (Lambda service exceptions, 3 attempts, backoff 2)
