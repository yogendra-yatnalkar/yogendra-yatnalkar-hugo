---
title: "Self Supervised Learning"
weight: 10
# bookFlatSection: false
# bookToc: true
# bookHidden: false
bookCollapseSection: false
bookComments: true
# bookSearchExclude: false
---

# A Cookbook of Self-Supervised Learning:

```
Initial Notes from: https://arxiv.org/abs/2304.12210
```

## Intro:

- NLP advanced due to SSL --> No need of labelled data to train supervised model

- SSL -> Define a pretext task --> Un-labelled data --> intelligent representation

- NLP: Word2Vec is SSL -- In a sentence, mask a word and predict the surrounding words (It learns context)

- CV: 2 current popular ways: 
  
  - mask a patch and prediction of masked path 
  
  - augmented version of the same sample --> train model such that embeddings from these 2 images are close as compared to any other image. 

- **Why SSL is hard and need of cookbook**
  
  - Computational Cost
  
  - No detailed papers and its proper implementation with parameters 
  
  - unified vocab 

## Origin of SSL:

Discussion about several pre-text tasks which were used few years ago in the field of SSL: 

1. **Information restoration:**  
   
   - Remove something from image and restore it or convert to grayscale and train a ML model to predict the colors. This helps in learning object semantics and boundaries. 
   
   - **Newer Method:** Masked-AutoEncoding - Transformer based where patches are masked

2. **Video Temporal Relationship:** 
   
   - Model training using triplet loss for similarity of two representations of same object in 2 different frames. 
   
   - Remove audio track and predict it based on the video input
   
   - Prediction of depth mapping between un-labelled image pairs. 

3. **Learning spatial context:**
   
   - Random rotation --> predict the amount of rotation
   
   - Jigsaw: convert image to blocks and create pairs --> predict the relative position of each pair. 

4. **:**
