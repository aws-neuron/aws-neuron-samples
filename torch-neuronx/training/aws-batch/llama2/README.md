# Training the Llama2 7B Model with AWS Batch and Trainium

This example demonstrates how to train the Llama2 7B model using AWS Batch with Trainium. AWS Batch offers a scalable and cost-effective solution for executing batch computing workloads in the AWS Cloud. By seamlessly integrating Trainium with AWS Batch, you can achieve an efficient and cost-effective approach to training deep learning models at scale.
The following sample is an adoption of Llama2 - 7B tutorial originally published under the [Neuronx-Distributed Docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tutorials/training_llama2_7b.html#llama2-7b-tp-zero1-tutorial).


## AWS Batch with Trainium

As illustrated in the below diagram, running jobs on AWS Batch require a few resources to be prepared. To learn more about setting up AWS Batch with Trainium, please refer to the official [Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/devflows/training/batch/batch-training.html#batch-training).

![image info](./images/aws-batch.png)

## Prerequisite infrastructure
Below is the list of resources and tools you should have before getting started with the training on AWS Batch. 

### VPC Creation
For this example we would require a VPC that has two subnets(one public and one private), and a Network Address Translation (NAT) gateway. Please follow the instructions mentioned [here](https://docs.aws.amazon.com/appstream2/latest/developerguide/managing-network-internet-NAT-gateway.html) on how to create a VPC with NAT Gateway.

### ECR Repo 
We would also need and ECR repo to store our docker container image. Please follow instructions [here](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html) on how to create an ECR repo.

### AWS CLI
AWS CLI should be installed and configured with permissions for Batch and ECR. You can follow the instructions mentioned [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) to install AWS CLI. 

### Other Tools
`Docker` and `jq` should also be installed.

## Getting Started with LLama training on AWS Batch with Trainium

### Configuration Update

Let's begin by updating the `config.txt` file to specify values for the following variables:

    REGION
    SUBNET
    SG
    ECR_REPO
    INSTANCE_ROLE
    FSX_DNS_NAME
    MOUNT_NAME

Furthermore, to tokenize the data, it is essential to acquire the tokenizer from HuggingFace and Meta. Follow the instructions outlined [here](https://huggingface.co/meta-llama/Llama-2-7b). It's important to note that the usage of the Llama 2 model is subject to the Meta license.

Prior to downloading the model weights and tokenizer, kindly visit the specified website and acknowledge the associated License terms. Upon obtaining access, use the download scripts provided by Meta to acquire the model weights and tokenizer. Once the download is complete, ensure to place the `tokenizer.model` in the root directory (`\llama2`).

### Trigger LLama2-7B training job on AWS Batch
To kickstart the training process on Trainium with AWS Batch, run the `llama_training_on_aws_batch.sh` script. This script seamlessly abstracts and handles all the steps outlined below for you. Simply execute the script to initiate the training effortlessly. Here are all the steps carried out by this script. 

#### 1. Docker Image Build
Initially, it constructs a Docker container image tailored for our training job. This image encompasses instructions for executing Llama2 - 7B training on Trainium, utilizing the `neuronx_distributed` library with `Tensor Parallelism` and the `ZeRO-1 Optimizer`. This script uses a tensor-parallel size of 8 which will automatically set the zero-1 sharding degree to 16 (4 * 32 workers / tensor_parallel_size).

##### Note: 
The hyperparameters for training are specified in `./docker/llama2/llama_batch_training.sh`. If you need to modify any of these values, please make the necessary changes before executing the `llama_training_on_aws_batch.sh` script.

#### 2. Resource Provisioning
The script orchestrates the provisioning of essential resources required for the training job. This includes the creation of a Placement Group, Launch Template, Compute Environment, Job Queue, and Job Definition. Notably, the tutorial is configured to leverage 4 Trn1.32xl nodes for computational tasks.

#### 3. AWS Batch Job Submission
Subsequently, it submits the AWS Batch job to commence the Llama2 model training. Upon submission, an `ECS Cluster` is dynamically established. Allow sufficient time for the cluster setup. Once operational, you can navigate through the cluster to monitor all tasks actively running on the `Trn1.32xl` instances, launched through this job.

In summary, running the `llama_training_on_aws_batch.sh` script simplifies the entire workflow of configuring resources and launching the Llama2 model training job on AWS Batch with Trainium.

Once the job is submitted, you can use `Amazon CloudWatch` Logs to monitor, store, and view all your logs from AWS Batch. Refer to the documentation [here](https://docs.aws.amazon.com/batch/latest/userguide/batch-eks-cloudwatch-logs.html) for additional details and insights into the logging process.