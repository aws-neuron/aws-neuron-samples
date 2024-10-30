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


class NeuronProblemDetectorStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        vpc = ec2.Vpc(self, "NeuronProblemDetectorVPC", max_azs=2)

        ecs_cluster = ecs.Cluster(self, "NeuronProblemDetectorCluster", vpc=vpc)

        ecs_cluster.add_capacity(
            id="NeruonAutoScalingGroupCapacity",
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
            cpu="1024",
            memory_mib="3072",
            compatibility=ecs.Compatibility.EC2,
            execution_role=task_execution_role,
            task_role=task_role
        )

        # Create the device mapping
        device_mapping = ecs.Device(
            host_path="/dev/kmsg",
            container_path="/dev/kmsg",
            permissions=[ecs.DevicePermission.READ, ecs.DevicePermission.WRITE],
        )

        linux_parameters = ecs.LinuxParameters(
            self,
            "NpdLinuxParameters",
        )

        linux_parameters.add_devices(device_mapping)

        npd_container = task_definition.add_container(
            "npd",
            image=ecs.ContainerImage.from_registry(
                "registry.k8s.io/node-problem-detector/node-problem-detector:v0.8.19"
            ),
            entry_point=["/bin/sh", "-c"],
            command=[
                'echo \'{"plugin":"kmsg","logPath":"/dev/kmsg","lookback":"5m","bufferSize":10,"source":"kernel-monitor","conditions":[{"type":"NeuronHealth","reason":"NeuronHasNoError","message":"Neuronhasnoerror"}],"rules":[{"type":"permanent","condition":"NeuronHealth","reason":"NeuronHasError_SRAM_UNCORRECTABLE_ERROR","pattern":".*NEURON_HW_ERR=SRAM_UNCORRECTABLE_ERROR.*"},{"type":"permanent","condition":"NeuronHealth","reason":"NeuronHasError_NC_UNCORRECTABLE_ERROR","pattern":".*NEURON_HW_ERR=NC_UNCORRECTABLE_ERROR.*"},{"type":"permanent","condition":"NeuronHealth","reason":"NeuronHasError_HBM_UNCORRECTABLE_ERROR","pattern":".*NEURON_HW_ERR=HBM_UNCORRECTABLE_ERROR.*"},{"type":"permanent","condition":"NeuronHealth","reason":"NeuronHasError_DMA_ERROR","pattern":".*NEURON_HW_ERR=DMA_ERROR.*"},{"type":"permanent","condition":"NeuronHealth","reason":"NeuronHasError_HANG_ON_COLLECTIVES","pattern":".*NEURON_HW_ERR=HANG_ON_COLLECTIVES.*"}]}\' > /config/kernel-monitor.json && /node-problem-detector --v=2 --logtostderr --enable-k8s-exporter=false --config.system-log-monitor=/config/kernel-monitor.json'
            ],
            privileged=True,
            logging=ecs.AwsLogDriver(
                stream_prefix="ecs",
                log_group=logs.LogGroup(
                    self,
                    "NpdLogGroup",
                    log_group_name="/ecs/npd",
                    retention=logs.RetentionDays.ONE_WEEK,
                ),
            ),
            linux_parameters=linux_parameters,
        )

        npd_container.add_port_mappings(
            ecs.PortMapping(
                name="npd-80-tcp",
                container_port=80,
                host_port=80,
                protocol=ecs.Protocol.TCP,
                app_protocol=ecs.AppProtocol.http,
            )
        )

        recovery_container = task_definition.add_container(
            "recovery",
            image=ecs.ContainerImage.from_registry(
                "public.ecr.aws/neuron/neuron-node-recovery:1.2.0"
            ),
            entry_point=["/bin/sh", "-c"],
            command=["python scripts/check-health.py"],
            environment={"ENABLE_RECOVERY": "true"},
            readonly_root_filesystem=True, 
            logging=ecs.AwsLogDriver(
                stream_prefix="ecs",
                log_group=logs.LogGroup(
                    self,
                    "RecoveryLogGroup",
                    log_group_name="/ecs/recovery",
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
