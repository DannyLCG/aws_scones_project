"""
The real-time inference endpoint: Model -> EndpointConfig -> Endpoint.

Safe note: this stack holds the only always-billing resource in the project.

CDK has no stable L2 for SageMaker (only an alpha module), so these are L1
Cfn* constructs — a thin, typed mirror of the CloudFormation resources.

Data capture is configured here rather than in Python: it was
DataCaptureConfig(...) in the notebook, and is now a property of the endpoint
config, so Model Monitor's input is guaranteed by the deployed template.
"""
from aws_cdk import CfnOutput, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sagemaker as sagemaker
from constructs import Construct


class EndpointStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        data_bucket: s3.IBucket,
        execution_role: iam.IRole,
        model_artifact: str,
        algorithm_image: str,
        instance_type: str = "ml.m5.xlarge",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Point the built-in algorithm container at the weights scripts/train.py
        # produced. Both come in as plain strings because neither is a CDK
        # resource: the image lives in an AWS-owned ECR account, and the
        # artifact is written by a training job CloudFormation cannot model.
        model = sagemaker.CfnModel(
            self,
            "Model",
            execution_role_arn=execution_role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=algorithm_image,
                model_data_url=model_artifact,
            ),
        )

        endpoint_config = sagemaker.CfnEndpointConfig(
            self,
            "EndpointConfig",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    variant_name="AllTraffic",
                    model_name=model.attr_model_name,
                    initial_instance_count=1,
                    instance_type=instance_type,
                    initial_variant_weight=1.0,
                )
            ],
            # Capture 100% of request/response pairs so Model Monitor has data to work with. 
            # The endpoint writes them to S3 with no logging code
            data_capture_config=sagemaker.CfnEndpointConfig.DataCaptureConfigProperty(
                enable_capture=True,
                initial_sampling_percentage=100,
                destination_s3_uri=f"s3://{data_bucket.bucket_name}/data_capture",
                capture_options=[
                    sagemaker.CfnEndpointConfig.CaptureOptionProperty(capture_mode="Input"),
                    sagemaker.CfnEndpointConfig.CaptureOptionProperty(capture_mode="Output"),
                ],
            ),
        )

        self.endpoint = sagemaker.CfnEndpoint(
            self,
            "Endpoint",
            endpoint_config_name=endpoint_config.attr_endpoint_config_name,
        )

        # Expose the generated name so LambdaStack can inject it into
        # predict_image_label instead of taking it as a hand-type value
        self.endpoint_name = self.endpoint.attr_endpoint_name

        CfnOutput(self, "EndpointName", value=self.endpoint_name)
