# VHS tape for the end-to-end demo:
#   data -> train -> deploy -> orchestrate -> monitor -> destroy
#
#   brew install vhs
#   vhs docs/demo.tape          # renders docs/demo.gif
#
# The point of scripting it: the recording is reproducible. Re-run this after
# changing the stacks and the GIF regenerates identically, with no editing.
#
# Hide/Show is what keeps it watchable. VHS keeps *executing* while hidden, so
# the ~10 min training job, the ~10 min endpoint creation and the ~8 min capture
# buffer all happen for real but never reach a frame. Roughly 45 minutes of wall
# time renders to a ~2 minute GIF of commands and their results.
#
# Prerequisites: the scones-dev profile, us-east-1 bootstrapped, and the .lst
# manifests generated (scripts/prepare_data.py). The tape builds the rest.
#
# Cost: one g5.xlarge training job (~$0.15) plus ~35 min of ml.m5.xlarge
# endpoint time (~$0.14). SconesDataStack is deployed but deliberately never
# destroyed — it holds the model artifact and the staged images for ~$0.005/mo.

Output docs/demo.gif

Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 760
Set TypingSpeed 35ms
Set Padding 20

# --- hidden setup: credentials and a deterministic prompt --------------------
# None of this is part of the story.
Hide
Type "export PS1='$ '"
Enter
Type "export AWS_PROFILE=scones-dev"
Enter
Type "cd /Users/momaz/Dannylcg/Obsidian/BRAIN/Projects/PProjects/aws_scones_project"
Enter
Type "clear"
Enter
Show

# --- 1. the persistent foundation --------------------------------------------
Type "# Bucket + SageMaker execution role. Deployed once, kept between demos."
Enter
Sleep 2s
Type "cd infra && cdk deploy SconesDataStack --require-approval never"
Enter
Sleep 5s
Hide
Wait@10m /\$$/
Show
Sleep 3s

# --- 2. train the model ------------------------------------------------------
# Training cannot be a CDK stack: AWS::SageMaker::TrainingJob is registered with
# CloudFormation but NON_PROVISIONABLE, so this step sits between two deploys.
Type "clear"
Enter
Type "cd .."
Enter
Type "# Training is a job, not a resource — CloudFormation cannot model one."
Enter
Sleep 2s
Type "uv run python scripts/train.py 2>&1 | tee /tmp/train.out"
Enter
Sleep 6s
Hide
Wait@25m /\$$/
Show
Sleep 4s

# Carry the freshly trained artifact into the deploy instead of pasting a URI,
# so the demo shows training and hosting actually connected
Type "clear"
Enter
Type 'export SCONES_MODEL_ARTIFACT=$(grep -oE "s3://[^ ]+model.tar.gz" /tmp/train.out | head -1)'
Enter
Type 'echo $SCONES_MODEL_ARTIFACT'
Enter
Wait@1m /\$$/
Sleep 3s

# --- 3. stand up the ephemeral stacks ----------------------------------------
Type "clear"
Enter
Type "# Endpoint, the three Lambdas, and the state machine that chains them."
Enter
Sleep 2s
Type "cd infra && cdk deploy SconesEndpointStack SconesLambdaStack SconesStepFunctionsStack --require-approval never"
Enter
Sleep 6s
Hide
Wait@25m /\$$/
Show
Sleep 3s

Type "clear"
Enter
Type 'EP=$(aws sagemaker list-endpoints --region us-east-1 --query "Endpoints[0].EndpointName" --output text)'
Enter
Wait@1m /\$$/
Type 'aws sagemaker wait endpoint-in-service --region us-east-1 --endpoint-name $EP && echo "$EP InService"'
Enter
Sleep 3s
Hide
Wait@20m /\$$/
Show
Sleep 3s

# --- 4. run the pipeline, both outcomes --------------------------------------
Type "clear"
Enter
Type "cd .."
Enter
Type "# One image the model is sure about, one it is not. The gate decides."
Enter
Sleep 2s
Type "uv run python scripts/run_executions.py"
Enter
Sleep 4s
Hide
Wait@10m /\$$/
Show
Sleep 6s

# --- 5. read back what the endpoint captured ---------------------------------
Type "clear"
Enter
Type "# Data Capture recorded every inference. No logging code in the handler."
Enter
Sleep 2s
Type "uv run python scripts/analyze_capture.py"
Enter
Sleep 6s
Hide
Wait@20m /\$$/
Show
Sleep 6s

# --- 6. tear down the billing resources --------------------------------------
Type "clear"
Enter
Type "# Destroy what costs money. The data stack keeps the model and the images."
Enter
Sleep 2s
Type "cd infra && cdk destroy SconesStepFunctionsStack SconesEndpointStack SconesLambdaStack --force"
Enter
Sleep 5s
Hide
Wait@15m /\$$/
Show
Sleep 3s

# --- 7. prove nothing is left running ----------------------------------------
Type "clear"
Enter
Type 'aws sagemaker list-endpoints --region us-east-1 --query "Endpoints[].EndpointName" --output text'
Enter
Wait@1m /\$$/
Sleep 2s
Type '# Empty: back to $0/hr, with the trained model still in S3.'
Enter
Sleep 4s
