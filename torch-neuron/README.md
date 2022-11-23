# PyTorch Neuron (torch-neuron) Samples for AWS Inf1

This directory contains Jupyter notebooks that demonstrate model compilation and inference using PyTorch Neuron for a variety of popular deep learning models. These samples can be run on [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/) (inf1 instances) using [Amazon SageMaker](https://aws.amazon.com/sagemaker) or [Amazon EC2](https://aws.amazon.com/ec2/).

For each sample you will also find additional information such as the model type, configuration used to compile the model, framework version, and a link to the original model implementation.

The following samples are available:

|Model Name	|Model Type	|Input Shape	|NeuronSDK Version	|Framework / Version	|Original Implementation	|
|---	|---	|---	|---	|---	|---	|
|[BERT-base](inference/bertbasecased)	|NLP	|max_length=128	|1.10.1.2.2.0.0	|Pytorch 1.10.2	|[link](https://huggingface.co/bert-base-cased)|
|[BERT-large](inference/bertlargeuncased)	|NLP	|max_length=128	|1.10.1.2.2.0.0	|Pytorch 1.10.2	|[link](https://huggingface.co/bert-large-uncased)|
|[CRAFT](inference/craft)		|CV - Text Detection	|1,3,800,800 - max_length=32|1.10.2.2.3.0.0 |Pytorch 1.10.2 |[link](https://github.com/clovaai/CRAFT-pytorch)|
|[EfficientNet](inference/efficientnet)	|CV - Image Classification	|1,3,224,224	|1.10.1.2.2.0.0	|Pytorch 1.10.1	|[link](https://pytorch.org/vision/stable/models/efficientnet.html)|
|[GFL](inference/gfl_mmdet)		|CV - Object Detection	|1,3,800,1216	|1.10.2.2.3.0.0 |Pytorch 1.10.2 |[link](https://github.com/open-mmlab/mmdetection/blob/master/configs/gfl/README.md)|
|[HRNet](inference/hrnet)	|CV - Pose Estimation	|1,3,384,288	|1.10.2.2.3.0.0	|Pytorch 1.10.2	|[link](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch.git)|
|[MarianNMT](inference/marianmt)	|NLP	|max_length=32 |1.7.\*|Pytorch 1.7|[link](https://huggingface.co/Helsinki-NLP/opus-mt-en-de)|
|[R-CNN](inference/rcnn)   |CV - Image Classification, Detection, and Segmentation   |1,3,800,800 	|1.11.0.2.5.2.0   |Pytorch 1.11.0	 |[link](https://github.com/facebookresearch/detectron2)|
|[ResNet (18,34,50,101,152)](inference/resnet)|CV - Image Classification	|1,3,224,224	|1.10.1.2.2.0.0	|Pytorch 1.10.1	|[link](https://pytorch.org/vision/stable/models/resnet.html)|
|[ResNetX](inference/resnext)	|CV - Image Classification	|1,3,224,224	|1.10.1.2.2.0.0	|Pytorch 1.10.1	|[link](https://pytorch.org/vision/stable/models/resnext.html)|
|[Roberta-base](inference/robertabase)	|NLP	|max_length=128|1.10.1.2.2.0.0	|Pytorch 1.10.2|[link](https://huggingface.co/roberta-base)|
|[SSD (SSD300-VGG16)](inference/ssd)	|CV - Object detection	|1,3,300,300	|1.10.2.2.3.0.0	|Pytorch 1.10.2	|[link](https://pytorch.org/vision/stable/models/ssd.html)|
|[TrOCR](inference/trocr)		|CV - OCR	|1,3,384,384	|1.10.2.2.3.0.0 |Pytorch 1.10.2 |[link](https://huggingface.co/docs/transformers/en/model_doc/trocr)|
|[VGG16](inference/vgg)	|CV - Image Classification	|1,3,224,224	|1.10.1.2.2.0.0	|Pytorch 1.10.1	|[link](https://pytorch.org/vision/stable/models/vgg.html)|
|[ViT](inference/vit)		|CV - Image Classification	|1,3,224,224	|1.10.2.2.3.0.0 |Pytorch 1.10.2 |[link](https://huggingface.co/docs/transformers/model_doc/vit)|
|[YOLOv5](inference/yolov5)	|CV - Object Detection	|1,3,640,640	|1.10.1.2.2.0.0	|Pytorch 1.10.1	|[link](https://github.com/ultralytics/yolov5/releases/tag/v5.0)|
|[YOLOv6](inference/yolov6)	|CV - Object Detection	|1,3,640,640	|1.11.0.2.3.0.0 |Pytorch 1.11.0 |[link](https://github.com/meituan/YOLOv6.git)|
|[YOLOF](inference/yolof_detectron2)	|CV - Object Detection	|1,3,300,300	|1.10.1.2.2.0.0	|Pytorch 1.10.1	|[link](https://github.com/chensnathan/YOLOF)|
|[Fairseq](inference/fairseq)	|NLP|max_length=32|1.10.1.*|Pytorch 1.10.1	|[link](https://github.com/facebookresearch/fairseq)|

### Configuring the environment

In order to run the samples, you first need to [set up a PyTorch Neuron development environment](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-intro/get-started.html).

