---
title: "E2E MLOps on AWS: P3 - CV Components and Pipelines Deep Dive"
weight: 10
# bookFlatSection: false
# bookToc: true
# bookHidden: false
bookCollapseSection: false
bookComments: true
# bookSearchExclude: false
---

# End-to-End MLOPS on AWS: Part3 - Computer Vision Components and Pipelines Deep Dive

| |
| --- |
| Authors: [Palash Nimodia](https://www.linkedin.com/in/palash-nimodia-94975b4b/), [Yogendra Yatnalkar](https://www.linkedin.com/in/yogendra-yatnalkar-2477b3148/) |
| Last Edited: 20/06/2023 |
| Previous Blog Link: [E2E MLOps on AWS: Part2 - Computer Vision Simulation with Drift & Retraining](../e2e-mlops-on-aws-p2-cv-simulation/) |
---

*In part 2 of our series, we gave a high-level overview of an MLOps system that handles large-scale image classification, end-to-end simulation and retraining evaluation to address drift. In this part 3, we will dive deeper into the computer vision components and pipelines that make up this system. Specifically, we will cover: how we designed components for batch workloads, how we implemented them on AWS using SageMaker and how we orchestrated them into MLOps pipelines using SageMaker Pipelines.*

| ![Component Mode and SageMaker Service](cv-components-on-sagemaker.png) |
| :--: |
| Component Mode and SageMaker Service |

## Quick Recap:

A **component** is an independent ML functionality or an ML operation which is a part of the larger process. A **pipeline** is a workflow which constitutes one or more components that execute a holistic task. 

There are **4 components** and **2 pipelines** as seen from image above, where each component has 2 modes, which are: 
- **Train mode** and **Serve mode**. 

Each component mode will be executed with any one of the following **Sagemaker Job** which are: 
- SageMaker Training Job or
- SageMaker Processing Job or
- SageMaker BatchTransform Job

As discussed in part1 of our blog, all the sagemaker workloads are developed using docker **containers (BYOC - Bring Your Own Container**).

## Components in detail:

### 1. **Processing Component**:

The processing component handles batch processing of large amounts of image data, using various data augmentation techniques. The batch-processing is usually performed before model training and inference.  

| ![Processing Component Lifecycle](processing-component-sagemaker.png) |   
| :--: |
| Processing Component Lifecycle |

On AWS, the processing component will be built using [**Amazon Sagemaker Processing Jobs** (link)](https://docs.aws.amazon.com/sagemaker/latest/dg/build-your-own-processing-container.html). On completion of the SageMaker Processing job, the output processed data in saved back in AWS S3. 

In the **train mode**, the processing job will transform train and validation data and in the **serve mode**, it will transform the production images. 

### 2. **Algorithm Component**:

The algorithm component performs 2 major tasks, which are **model training** and **model inference**. 

| ![Algorithm Component Lifecycle](algorithm-component-sagemaker.png) |   
| :--: |
| Algorithm Component Lifecycle |

Unlike processing component, the 2 modes of algorithm component will perform completely different tasks.
- The **train mode** will perform the task of **ML model training** on the training-dataset and computing the evaluation metric on the validation-dataset. Based on the ML model used, the train mode will also support **model retraining capabilities**. On AWS, the train mode of the algorithm component will be implemented using [**Sagemaker Training Job**](https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html)   .

- In **serve mode**, the production data is fed into the trained ML model for prediction. This model is the same one that was **trained in train mode**. On AWS, the serve mode of the algorithm component will be implemented using [**Sagemaker BatchTransform Job**](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html).

**`The data input to the algorithm component is processed train, validation or production data that are the outputs of the processing component.`**

### 3. **Monitoring Component**:

**Model monitoring** refers to the process of closely tracking the performance of machine learning models in production. It enables us to detect any technical or performance issues, but mainly it is **used to detect drift**. There are two main types of drift:
- **Data drift:** Drift when characteristics of the underlying data on which the model has trained changes over time.
- **Concept Drift:** Concept drift occurs when the characteristics of the target variable and its relationship with the training variables itself changes overtime.

| ![Monitoring Component Lifecycle](monitoring-component-sagemaker.png) |   
| :--: |
| Monitoring Component Lifecycle |

On AWS, both the **train** and **serve** mode of the monitoring component are developed using **SageMaker Processing Job**. 
- In the **train** mode, the processing job learns the training images data distribution and saves it in a file which is also known as **drift detection artifacts**.
- In the **serve** mode, using the **drift** artifacts from the train mode, **drift is identified on the production data**. 

**`The data input to the monitoring component is usually the raw data itself, as augmentations performed by the processing component might change the input data distribution.`**

### 4. **Explainability Component**:

As the name suggests, the explainability component is used to understand the **model interpretability and cause of its inference decision** behind every inferred data sample. In any production MLOps system, explainable AI is very important as it **adds accountability and compliance** to our production system.

| ![Explainability Component Lifecycle](explainability-component-sagemaker.png) |   
| :--: |
| Explainability Component Lifecycle |

On AWS, both the **train** and **serve** mode of the explainability component are developed using **SageMaker Processing Job**. 

**`The data input to the explainability component is processed train, validation or production data which are the outputs of the processing component.`**

The second input during both the modes of the processing job is the trained ML model itself, which is the output of the **algorithm component train mode.** 

In our case of CV image classification, the explainability component computes the **integrated gradients** for each data sample, from the production data in **serve mode** and training data in **train mode** using the **trained classification model**. 

## Pipelines in detail:

From the image classification use-case (part-2 blog), we are already aware that we have 2 pipelines which are: 
- **Training Pipeline**
- **Batch-Inference Pipeline**

The 2 pipelines seen in the below image are developed using [AWS SageMaker Pipelines (LINK)](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html)

### About SageMaker Pipelines: 

*An Amazon SageMaker Model Building Pipelines pipeline is a series of interconnected steps that are defined using the Pipelines SDK. This pipeline definition encodes a pipeline using a directed acyclic graph (DAG) that can be exported as a JSON definition. This DAG gives information on the requirements for and relationships between each step of defined pipeline. The structure of a pipeline's DAG is determined by the data dependencies between steps.*

| ![SageMaker Pipelines](pipeline.png) |   
| :--: |
| SageMaker Pipelines |

- We have already seen that each component has 2 modes of executions, which are the **"Train"** mode and **"Serve"** mode. All the components chained together in the **training** pipeline are set to run in the **"Train"** mode. Similarly, all the components chained together in the **batch-inference** pipeline are set to run in the **"Serve"** mode. 

- The inputs and outputs of each pipeline execution are stored in **AWS S3**. The S3 paths are then updated in **AWS Parameter Store**. ***This allows components from different pipelines to access each other's outputs***. The Parameter Store keys have a fixed prefix for each component output.

- As we learned in blog 2 (part 2), the batch-inference pipeline runs on a schedule. We used **AWS EventBridge** to set up this scheduled trigger for the SageMaker pipeline. 

- On the other hand, we saw that when drift was detected in the batch-inference pipeline, the training pipeline was triggered automatically to tackle drift. This automatic trigger is carried out using **AWS Lambda Function**. 

- The component execution and pipeline execution logs are pushed to **AWS CloudWatch**.

- The containers used to execute individual components are stored in **AWS ECR (Elastic Container Registry)**.

## The END

Thank you for reading this blog. We hope you gained some new and valuable insights from it. In this blog, we explained the different components of our project and how we developed them on AWS. We also showed you how we used SageMaker Pipelines to orchestrate the workflow and automate the deployment.

**`Please provide us your valuable feedback and stay tuned. The code for the components and pipelines will be released soon....`**

