---
title: "Model Serving"
weight: 10
# bookFlatSection: false
# bookToc: true
# bookHidden: false
bookCollapseSection: false
bookComments: true
# bookSearchExclude: false
---

# Notes on Model Serving 

> **Example: Torchserve, Tf-Serving, Triton, Flask, etc**

TorchServe: 
	- to check model status, I am using port 8081
	- for inference, I am using port 8080
	- if not using ts-config while deploying a model, it generates error
	- Question:
		- when to use 8080 vs 8081 --> Inference api is bind to 8080, management api is bind to 8081
		- how to load test ? --> locusts (fairly easy to use)
		- 
		

serve/examples/image_classifier/mnist/
		
torch-model-archiver --model-name mnist --version 1.0 --model-file mnist.py \
--serialized-file mnist_cnn.pt --handler  mnist_handler.py


mkdir model_store
mv mnist.mar model_store/
torchserve --start --model-store model_store --models mnist=mnist.mar --workers 4 --ts-config config.properties
curl http://127.0.0.1:8080/predictions/mnist -T test_data/0.png 