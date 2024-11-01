# Overview

This project contains CDK code to provision :

* An ECS Cluster and one Inf2.xlarge EC2 instance joining the cluster. 
* An ECS Task Definition for Neruon Problem Detector and Recovery
* An ECS Service that run the containers as Daemon in all instances
* Related IAM roles and log groups  


This project is set up like a standard Python project.  The initialization
process also creates a virtualenv within this project, stored under the `.venv`
directory.  To create the virtualenv it assumes that there is a `python3`
(or `python` for Windows) executable in your path with access to the `venv`
package. If for any reason the automatic creation of the virtualenv fails,
you can create the virtualenv manually.

The `cdk.json` file tells the CDK Toolkit how to execute your app.

## Pre-requisites
Before you start, ensure that you have installed the latest version of the following tools on your machine:

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
2. [aws cdk](https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html)
3. [Session Manager Plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)


## Environment Setup 
To manually create a virtualenv on MacOS and Linux:

```
$ python3 -m venv .venv
```

After the init process completes and the virtualenv is created, you can use the following
step to activate your virtualenv.

```
$ source .venv/bin/activate
```

If you are a Windows platform, you would activate the virtualenv like this:

```
% .venv\Scripts\activate.bat
```

Once the virtualenv is activated, you can install the required dependencies.

```
$ pip install -r requirements.txt
```

## Synthesize CloudFormation template
At this point you can now synthesize the CloudFormation template for this code.

```
$ cdk synth
```
It is assumed that you have authenticated successfully to connect to your AWS environment. 

Perform bootstrap function with the following command.
```
cdk bootstrap [--profile <profile name>]
```
Deploy the stack in your AWS environment

```
cdk deploy [--profile <profile name>]
```

## Optional
To add additional dependencies, for example other CDK libraries, just add
them to your `setup.py` file and rerun the `pip install -r requirements.txt`
command.

## Useful commands

 * `cdk ls`          list all stacks in the app
 * `cdk synth`       emits the synthesized CloudFormation template
 * `cdk deploy`      deploy this stack to your default AWS account/region
 * `cdk diff`        compare deployed stack with current state
 * `cdk docs`        open CDK documentation


