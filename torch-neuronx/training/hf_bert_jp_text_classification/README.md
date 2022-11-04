# Hugging Face Text Classification with BERT Japanese model.

This folder contains example of Hugging Face Text Classification using [Amazon Review Dataset](https://huggingface.co/datasets/amazon_reviews_multi/) and [BERT Japanese model](https://huggingface.co/cl-tohoku). 
The training script uses Hugging Face Trainer API to fine tune the pretrained model. 

These scripts are explained in the following AWS Japan Blogs.

- Blog Part1 : https://aws.amazon.com/jp/blogs/news/aws-trainium-amazon-ec2-trn1-ml-training-part1/
- Blog Part1 : https://aws.amazon.com/jp/blogs/news/aws-trainium-amazon-ec2-trn1-ml-training-part2/

## Getting started

```
pip install -U transformers[ja]==4.16.2 datasets==2.6.1
./test.sh
./test_BF16.sh
```
