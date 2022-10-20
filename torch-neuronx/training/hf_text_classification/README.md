# Hugging Face Text Classification

This folder contains various examples of Hugging Face models that can be trained with AWS Trainium for a Text Classification task. Each Jupyter notebook contains a specific example of training a model using the Hugging Face Trainer API and uses a slightly modified script called [run_glue.py](run_glue.py) to fine tune the pretrained model. 
  
The following models are currently supported and tested with AWS Trainium:
- [BERT base cased](BertBaseCased.ipynb)
- [BERT base uncased](BertBaseUncased.ipynb)
- [BERT large cased](BertLargeCased.ipynb)
- [RoBERTa base](RobertaBase.ipynb)
- [RoBERTa large](RobertaLarge.ipynb)
- [XLM RoBERTa base](XlmRobertaBase.ipynb)
