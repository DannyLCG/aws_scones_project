# ML Workflow in AWS
This repo contanins code, and downloaded data from the Udacity's project "Build a Machine Learning Workflow for Scones unlimited on Amazon SageMaker". This project is a Computer Vision problem: I built an image classification model to autimatically detect the type of vehicle that delivery drivers use. The project used the CIFAR100 data and images corresponding to 'motorcycles' or 'bicycles' were retrieved by code. The goal was to deliver a scalable and reliable model. A really important point once the model is available to other teams is scalability so odel monitoring was implemented in order to prevent drift or degraded performance. I deployed an ML workflow by writing and chaining AWS Lambda functions with the help of AWS Step Functions. 

## Folder descriptions
*test*: contains the testing data already filtered from CIFAR100 (200 images).

*train*: contains the training data already filtered from CIFAR100 (1000 images).

*scripts*: Contains auxiliary code, mainly Lambda functions organized each in its own folder and with their corresponding json test file to facilitate function testing on the AWS Console.
