from aws_cdk import (
    # Duration,
    Stack,
    # aws_sqs as sqs,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_logs as logs,
    aws_autoscaling as autoscaling,
)
from constructs import Construct
import json 



class NeuronProblemDetectorStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        with open('neuron_problem_detector/ecs_task_definition.json', 'r') as f:
            ecs_task_definition = json.load(f)
            
        vpc = ec2.Vpc(self, "NeuronProblemDetectorVPC", max_azs=2)

        ecs_cluster = ecs.Cluster(self, "NeuronProblemDetectorCluster", vpc=vpc)

        ecs_cluster.add_capacity(
            id="NeuronAutoScalingGroupCapacity",
            machine_image=ecs.EcsOptimizedImage.amazon_linux2(
                ecs.AmiHardwareType.NEURON
            ),
            max_capacity=3,
            min_capacity=1,
            desired_capacity=1,
            instance_type=ec2.InstanceType("inf2.xlarge"),
            ssm_session_permissions=True,
            can_containers_access_instance_role=True,
        )

        # Create the task execution role
        task_execution_role = iam.Role(
            self,
            "NeuronProblemDetectorTaskExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                ),
            ],
        )

        iam_policy_document = iam.PolicyDocument(
            statements=[
                iam.PolicyStatement(
                    actions=[
                        "autoscaling:SetInstanceHealth",
                        "autoscaling:DescribeAutoScalingInstances",
                    ],
                    resources=["*"],
                    effect=iam.Effect.ALLOW,
                ),
                iam.PolicyStatement(
                    actions=["ec2:DescribeInstances"],
                    resources=["*"],
                    effect=iam.Effect.ALLOW,
                ),
                iam.PolicyStatement(
                    actions=["cloudwatch:PutMetricData"],
                    resources=["*"],
                    effect=iam.Effect.ALLOW
                ),
            ]
        )

        iam.PolicyStatement(
            actions=[
                "autoscaling:SetInstanceHealth",
                "autoscaling:DescribeAutoScalingInstances",
            ],
            resources=["*"],
            effect=iam.Effect.ALLOW,
        )

        # Create a task role (if needed)
        task_role = iam.Role(
            self,
            "NeuronProblemDetectorTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            inline_policies={"node-recovery": iam_policy_document},
        )

        # Create an ECS Task Definition
        task_definition = ecs.TaskDefinition(
            self,
            "NeuronNpdAndRecoveryTaskDef",
            family="neuron-npd-and-recovery",
            network_mode=ecs.NetworkMode.AWS_VPC,
            cpu=ecs_task_definition["cpu"],
            memory_mib=ecs_task_definition["memory"],
            compatibility=ecs.Compatibility.EC2,
            execution_role=task_execution_role,
            task_role=task_role
        )

        # Create the device mapping
        device_mapping = ecs.Device(
            host_path=ecs_task_definition["containerDefinitions"][0]["linuxParameters"]["devices"][0]["hostPath"],
            container_path=ecs_task_definition["containerDefinitions"][0]["linuxParameters"]["devices"][0]["containerPath"],
            permissions=[ecs.DevicePermission.READ, ecs.DevicePermission.WRITE],
        )

        linux_parameters = ecs.LinuxParameters(
            self,
            "NpdLinuxParameters",
        )

        linux_parameters.add_devices(device_mapping)

        npd_container = task_definition.add_container(
            ecs_task_definition["containerDefinitions"][0]["name"],
            image=ecs.ContainerImage.from_registry(
                ecs_task_definition["containerDefinitions"][0]["image"]
            ),
            entry_point=ecs_task_definition["containerDefinitions"][0]["entryPoint"],
            command=ecs_task_definition["containerDefinitions"][0]["command"],
            privileged=True,
            logging=ecs.AwsLogDriver(
                stream_prefix=ecs_task_definition["containerDefinitions"][0]["logConfiguration"]["options"]["awslogs-stream-prefix"],
                log_group=logs.LogGroup(
                    self,
                    "NpdLogGroup",
                    log_group_name=ecs_task_definition["containerDefinitions"][0]["logConfiguration"]["options"]["awslogs-group"],
                    retention=logs.RetentionDays.ONE_WEEK,
                ),
            ),
            linux_parameters=linux_parameters,
        )

        npd_container.add_port_mappings(
            ecs.PortMapping(
                name=ecs_task_definition["containerDefinitions"][0]["portMappings"][0]["name"],
                container_port=ecs_task_definition["containerDefinitions"][0]["portMappings"][0]["containerPort"],
                host_port=ecs_task_definition["containerDefinitions"][0]["portMappings"][0]["hostPort"],
                protocol=ecs.Protocol.TCP,
                app_protocol=ecs.AppProtocol.http,
            )
        )

        recovery_container = task_definition.add_container(
            ecs_task_definition["containerDefinitions"][1]["name"],
            image=ecs.ContainerImage.from_registry(
                ecs_task_definition["containerDefinitions"][1]["image"]
            ),
            entry_point=ecs_task_definition["containerDefinitions"][1]["entryPoint"],
            command=ecs_task_definition["containerDefinitions"][1]["command"],
            environment={
                ecs_task_definition["containerDefinitions"][1]["environment"][0]["name"]: ecs_task_definition["containerDefinitions"][1]["environment"][0]["value"]
            },
            readonly_root_filesystem=ecs_task_definition["containerDefinitions"][1]["readonlyRootFilesystem"], 
            logging=ecs.AwsLogDriver(
                stream_prefix=ecs_task_definition["containerDefinitions"][1]["logConfiguration"]["options"]["awslogs-stream-prefix"],
                log_group=logs.LogGroup(
                    self,
                    "RecoveryLogGroup",
                    log_group_name=ecs_task_definition["containerDefinitions"][1]["logConfiguration"]["options"]["awslogs-group"],
                    retention=logs.RetentionDays.ONE_WEEK,
                ),
            ),
        )

        ec2_service = ecs.Ec2Service(
            self,
            "NeuronNpdAndRecoveryDaemonService",
            cluster=ecs_cluster,
            task_definition=task_definition,
            daemon=True,
            enable_execute_command=True,
        )
