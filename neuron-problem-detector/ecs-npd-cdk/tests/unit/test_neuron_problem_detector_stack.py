import aws_cdk as core
import aws_cdk.assertions as assertions

from neuron_problem_detector.neuron_problem_detector_stack import NeuronProblemDetectorStack

# example tests. To run these tests, uncomment this file along with the example
# resource in neuron_problem_detector/neuron_problem_detector_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = NeuronProblemDetectorStack(app, "neuron-problem-detector")
    template = assertions.Template.from_stack(stack)
    
    template.has_resource_properties("AWS::ECS::Cluster",{})


#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
