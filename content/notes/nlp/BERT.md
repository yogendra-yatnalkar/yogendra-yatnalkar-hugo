---
title: "BERT"
weight: 10
# bookFlatSection: false
# bookToc: true
bookHidden: true
bookCollapseSection: false
bookComments: true
# bookSearchExclude: false
---

---

# BERT (Bidirectional Encoder Representation From Transformer)

---

Source: 

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding - YouTube](https://www.youtube.com/watch?v=-9evrZnBorM&t=1403s) 

- Original Paper: https://arxiv.org/pdf/1810.04805v2.pdf 







---

Source:

- [BERT Neural Network - EXPLAINED! - YouTube](https://www.youtube.com/watch?v=xI0HHN5XKDo&t=70s)

## Before BERT:

- LSTM's were used.
- Problems:
  - Slow as each word is processed at a time (sequentially)
  - Not truly bi-directional (left to right and right to left at a time in bidirectional LSTM)

- **Bert Architecture:** Multiple encoders stacked on each-other 

- Pretraining and Finetuning

- Pretraining Task is used to learn the language and its context. It is done using two tasks: 
  
  - **Mask Language Model (MLM):** 
    
    - Sentence sentence {Fill_in_the_Blanks} remaining sentence
    
    - Helps bert understand the bidirectional meaning of a sentence
  
  - **Next Sentence Prediction (NSP):** 
    
    - Predict whether the a given sentence is the next sentence of the current sentence. Like a binary classification task. 
    
    - It helps bert in understanding context across different sentences.
  
  - Usually, the MLM task and NSP task are performed simultaneously. 

- **Finetuning:**
  
  - Finetune on task specific data. 
  
  - Fast and compute efficient
  
  - Only replace last few layers of the original architecture
  
  --- 
  
  
