# Training the Llama2 7B Model with AWS Batch and Trainium

This example demonstrates how to train the Llama2 7B model using AWS Batch with Trainium. AWS Batch offers a scalable and cost-effective solution for executing batch computing workloads in the AWS Cloud. By seamlessly integrating Trainium with AWS Batch, you can achieve an efficient and cost-effective approach to training deep learning models at scale.
The following sample is an adoption of Llama2 - 7B tutorial originally published under the [Neuronx-Distributed Docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tutorials/training_llama2_7b.html#llama2-7b-tp-zero1-tutorial).


## AWS Batch with Trainium

For detailed instructions on setting up AWS Batch with Trainium, please refer to the official [Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/devflows/training/batch/batch-training.html#batch-training).

![image info](./images/aws-batch.png)

## Prerequisite infrastructure

### VPC Creation
For this example we would require a VPC that has two subnets and a Network Address Translation (NAT) gateway. Please follow the instructions mentioned [here](https://docs.aws.amazon.com/appstream2/latest/developerguide/managing-network-internet-NAT-gateway.html) on how to create a VPC with NAT Gateway.

### ECR Repo 
We would also need and ECR repo to store our docker container image. Please follow instructions [here](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html) on how to create an ECR repo.

### AWS CLI
AWS CLI should be installed and configured with permissions for Batch and ECR. You can follow the instructions mentioned [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) to install AWS CLI. 

### Other Tools
`Docker` and `jq` should also be installed.

## Training on AWS Batch with Trainium

### Configuration Update

Let's begin by updating the `config.txt` file to specify values for the following variables:

    REGION
    SUBNET
    SG
    ECR_REPO
    INSTANCE_ROLE
    FSX_DNS_NAME
    MOUNT_NAME

To tokenize the data, it is essential to acquire the tokenizer from HuggingFace and Meta. Follow the instructions outlined [here](https://huggingface.co/meta-llama/Llama-2-7b). It's important to note that the usage of the Llama 2 model is subject to the Meta license.

Before proceeding with downloading the model weights and tokenizer, please visit the aforementioned website and acknowledge their License terms. Once access is granted, utilize the download scripts provided by Meta to fetch the model weights and tokenizer. Once you have downloaded, you can copy the `tokenizer.model` to the root directory.

### Trigger LLama2-7B training job on AWS Batch
Execute the `llama_training_on_aws_batch.sh` script to initiate the training process. The script performs the following steps:

#### Docker Image Build:
Initially, it constructs a Docker container image tailored for our training job. This image encompasses instructions for executing Llama2 - 7B training on Trainium, utilizing the neuronx_distributed library with Tensor Parallelism and the ZeRO-1 Optimizer. This script uses a tensor-parallel size of 8 which will automatically set the zero-1 sharding degree to 16 (4 * 32 workers / tensor_parallel_size).

##### Note: 
All the training related hyperparameters are defined in `./docker/llama2/llama_batch_training.sh`. In case you want to make any amendments to any of those values, please do that before running the `llama_training_on_aws_batch.sh` script. `
#### Resource Provisioning:
The script orchestrates the provisioning of essential resources required for the training job. This includes the creation of a Placement Group, Launch Template, Compute Environment, Job Queue, and Job Definition. Notably, the tutorial is configured to leverage 4 Trn1.32xl nodes for computational tasks.

#### AWS Batch Job Submission:
Subsequently, it submits the AWS Batch job to commence the Llama2 model training. Upon submission, an ECS Cluster is dynamically established. Allow sufficient time for the cluster setup. Once operational, you can navigate through the cluster to monitor all tasks actively running on the Trn1.32xl instances, launched through this job.

To summarize, executing `llama_training_on_aws_batch.sh` script streamlines the entire process of setting up resources and initiating the training job on AWS Batch for the Llama2 model.

Also, For comprehensive log tracking, all logs are published to CloudWatch. Refer to the documentation [here](https://docs.aws.amazon.com/batch/latest/userguide/batch-eks-cloudwatch-logs.html) for additional details and insights into the logging process.