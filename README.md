## AWS Neuron Samples

### Inference compatibility table

In this table you find a set of models that were tested with **NeuronSDK** for inference on inf1 instances (EC2 or SageMaker). For each entry you'll see useful information like: the configurations used to compile the model, framework versions, a link to the original implementation, a link to a notebook that shows how to compile and run the models on Inferentia, etc.  

If your model is not listed here, please feel free to cut an issue, but don't forget to add: a link to the original implementation + any other additional information. Additional models will be added to this table based on the number of requests.

**Configuring the environment**  
In order to run the notebooks, you need to setup an environment first. [Please, check the instructions on this page](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-intro/get-started.html) to prepare your development env. 


|#	|Model Name	|Model Type	|Input Shape	|NeuronSDK Version	|Framework/Version	|Original Implementation	|Sample	|
|---	|---	|---	|---	|---	|---	|---	|---	|
|1	|EfficientNet	|CV - Image Classification	|1,3,224,224	|1.10.1.2.2.0.0	|Pytorch 1.10.1	|[link](https://pytorch.org/vision/stable/models/efficientnet.html)|[notebook](torch-neuron/inference/efficientnet)|
|2	|ResNet(18,34,50,101,152)|CV - Image Classification	|1,3,224,224	|1.10.1.2.2.0.0	|Pytorch 1.10.1	|[link](https://pytorch.org/vision/stable/models/resnet.html)|[notebook](torch-neuron/inference/resnet)|
|3	|ResNetX	|CV - Image Classification	|1,3,224,224	|1.10.1.2.2.0.0	|Pytorch 1.10.1	|[link](https://pytorch.org/vision/stable/models/resnext.html)|[notebook](torch-neuron/inference/resnext)	|
|4	|VGG16	|CV - Image Classification	|1,3,224,224	|1.10.1.2.2.0.0	|Pytorch 1.10.1	|[link](https://pytorch.org/vision/stable/models/vgg.html)|[notebook](torch-neuron/inference/vgg)|
|5	|Yolo-v5	|CV - Object Detection	|1,3,640,640	|1.10.1.2.2.0.0	|Pytorch 1.10.1	|[link](https://github.com/ultralytics/yolov5/releases/tag/v5.0)|[notebook](torch-neuron/inference/yolov5)	|
|6	|SSD (SSD300-VGG16)	|CV - Object detection	|1,3,300,300	|1.10.2.2.3.0.0	|Pytorch 1.10.2	|[link](https://pytorch.org/vision/stable/models/ssd.html)|[notebook](torch-neuron/inference/ssd)|
|7	|HRNet	|CV - Pose Estimation	|1,3,384,288	|1.10.2.2.3.0.0	|Pytorch 1.10.2	|[link](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch.git)|[notebook](torch-neuron/inference/hrnet)|
|8	|U-Net	|CV - Semantic Segmentation	|1,3,224,224	|2.5.2.2.1.14.0	|Tensorflow 2.5.2	|[link](https://github.com/jakeret/unet)|[notebook](tensorflow-neuron/inference/unet)|
|9	|BERT-Base	|NLP	|max_length=128	|1.10.1.2.2.0.0	|Pytorch 1.10.2	|[link](https://huggingface.co/bert-base-cased)|[notebook](torch-neuron/inference/bertbasecased)|
|10	|BERT-Large	|NLP	|max_length=128	|1.10.1.2.2.0.0	|Pytorch 1.10.2	|[link](https://huggingface.co/bert-large-uncased)|[notebook](torch-neuron/inference/bertlargeuncased)|
|11	|MarianNMT	|NLP	|max_length=32 |1.7.\*|Pytorch 1.7|[link](https://huggingface.co/Helsinki-NLP/opus-mt-en-de)|[notebook](torch-neuron/inference/marianmt)|
|12	|Roberta-Base	|NLP	|max_length=128|1.10.1.2.2.0.0	|Pytorch 1.10.2|[link](https://huggingface.co/roberta-base)|[notebook](torch-neuron/inference/robertabase)|
|13	|ViT		|CV - Image Classification	|1,3,224,224	|1.10.2.2.3.0.0 |Pytorch 1.10.2 |[link](https://huggingface.co/docs/transformers/model_doc/vit)|[notebook](torch-neuron/inference/vit)	|
|14	|Craft		|CV - Text Detection	|1,3,800,800 - max_length=32|1.10.2.2.3.0.0 |Pytorch 1.10.2 |[link](https://github.com/clovaai/CRAFT-pytorch)|[notebook](torch-neuron/inference/craft)	|
|15	|TrOCR		|CV - OCR	|1,3,384,384	|1.10.2.2.3.0.0 |Pytorch 1.10.2 |[link](https://huggingface.co/docs/transformers/en/model_doc/trocr)|[notebook](torch-neuron/inference/trocr)	|
|16	|GFL		|CV - Object Detection	|1,3,800,1216	|1.10.2.2.3.0.0 |Pytorch 1.10.2 |[link](https://github.com/open-mmlab/mmdetection/blob/master/configs/gfl/README.md)|[notebook](torch-neuron/inference/gfl_mmdet)	|
|17	|Yolov-v6	|CV - Object Detection	|1,3,640,640	|1.11.0.2.3.0.0 |Pytorch 1.11.0 |[link](https://github.com/meituan/YOLOv6.git)|[notebook](torch-neuron/inference/yolov6)|


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the Amazon Software License.