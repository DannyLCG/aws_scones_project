"""State machine chaining the three Lambdas.

Replaces the hand-edited SconestateMachine_definition.json, which pinned a
specific account ID and `:$LATEST` function ARNs into committed JSON — so it
only ever worked in the account it was exported from. Here the ARNs come from
construct references, and CDK generates the ASL at synth time.

The retry policy mirrors the original definition exactly: the four transient
Lambda service errors, 3 attempts, 1s interval, backoff rate 2. Note it does
NOT retry THRESHOLD_CONFIDENCE_NOT_MET — a low-confidence prediction is a
verdict, not a transient fault, so that failure must end the execution.
"""
from aws_cdk import CfnOutput, Duration, Stack
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from constructs import Construct

# The transient faults worth retrying, copied from the original ASL
RETRYABLE_LAMBDA_ERRORS = [
    "Lambda.ServiceException",
    "Lambda.AWSLambdaException",
    "Lambda.SdkClientException",
    "Lambda.TooManyRequestsException",
]


class StepFunctionsStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        serialize_image_data: _lambda.IFunction,
        predict_image_label: _lambda.IFunction,
        filter_predictions: _lambda.IFunction,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        def invoke(state_id: str, fn: _lambda.IFunction) -> tasks.LambdaInvoke:
            # Unwrap the Lambda response envelope so each step receives the
            # previous step's payload directly — the OutputPath: $.Payload the
            # original definition used, which is what lets the handlers pass
            # plain dicts to each other
            task = tasks.LambdaInvoke(
                self,
                state_id,
                lambda_function=fn,
                output_path="$.Payload",
            )
            # Ride out transient Lambda faults rather than failing the run
            task.add_retry(
                errors=RETRYABLE_LAMBDA_ERRORS,
                interval=Duration.seconds(1),
                max_attempts=3,
                backoff_rate=2,
            )
            return task

        # Chain the three steps in the order the payload flows
        definition = (
            invoke("serializeImageData", serialize_image_data)
            .next(invoke("predictImageLabel", predict_image_label))
            .next(invoke("filterPredictions", filter_predictions))
        )

        self.state_machine = sfn.StateMachine(
            self,
            "SconesStateMachine",
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            # Cap a run so a stuck endpoint cannot hold an execution open
            timeout=Duration.minutes(5),
        )

        # Print the ARN so executions can be started from the CLI without
        # looking the name up in the console
        CfnOutput(self, "StateMachineArn", value=self.state_machine.state_machine_arn)
