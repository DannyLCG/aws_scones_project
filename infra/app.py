"""CDK entry point.

TODO: once the remaining stacks below own real resources, wire the rest of
the cross-stack references (Lambda ARNs -> state machine, endpoint name ->
monitor schedule) the same way data_stack -> lambda_stack is wired here.
"""
import aws_cdk as cdk

from stacks.data_stack import DataStack
from stacks.lambda_stack import LambdaStack
from stacks.stepfunctions_stack import StepFunctionsStack
from stacks.monitor_stack import MonitorStack

app = cdk.App()

# Stand up the data bucket first so its dependents can reference the real resource
data_stack = DataStack(app, "SconesDataStack")
LambdaStack(app, "SconesLambdaStack", data_bucket=data_stack.bucket)
StepFunctionsStack(app, "SconesStepFunctionsStack")
MonitorStack(app, "SconesMonitorStack")

app.synth()
