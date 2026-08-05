"""CDK entry point.

TODO: once each stack below owns real resources, wire cross-stack references
(data bucket -> Lambda env vars, Lambda ARNs -> state machine, endpoint name ->
monitor schedule) instead of instantiating them independently.
"""
import aws_cdk as cdk

from stacks.data_stack import DataStack
from stacks.lambda_stack import LambdaStack
from stacks.stepfunctions_stack import StepFunctionsStack
from stacks.monitor_stack import MonitorStack

app = cdk.App()

DataStack(app, "SconesDataStack")
LambdaStack(app, "SconesLambdaStack")
StepFunctionsStack(app, "SconesStepFunctionsStack")
MonitorStack(app, "SconesMonitorStack")

app.synth()
