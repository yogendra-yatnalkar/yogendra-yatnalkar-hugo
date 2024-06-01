---
title: "Kullback-Leibler Divergence (KL Divergence)"
weight: 10
# bookFlatSection: false
# bookToc: true
# bookHidden: false
bookCollapseSection: false
bookComments: true
# bookSearchExclude: false
---

# Kullback-Leibler Divergence (KL Divergence)

|             |            |
| ----------- | ---------- |
| Last Edited | 25/06/2023 |

---

### Definition:

- Measures the distance between 2 prabability distributions

### Explanation + Proof:

Base Video: [Intuitively Understanding the KL Divergence - YouTube](https://www.youtube.com/watch?v=SxGYPqCgJWM) 

![](kl-divergence/2023-06-25-13-09-22-image.png)

Sequence of flips: **H -> H -> T .....**

Multiply the probabilities from both the coins for the corresponding heads and tails. It is nothing but: 

- for True coin: P1 raise to something and P2 raise to something else

- For coin2: Q1 raise to soemthing and Q2 raise to something else

![](kl-divergence/2023-06-25-13-13-33-image.png)

- after applying log to the RHS: (** --> Explained at the end)
  
  ![](kl-divergence/2023-06-25-13-14-41-image.png)

- As the number of observations tends towards infinity: 
  
  - **Nh/n ~~ p1**
  
  - **Nt/N ~~ p2**
  
  This leads us to the final log expression: 

![](kl-divergence/2023-06-25-13-23-53-image.png)

#### General Formulae:

  ![](kl-divergence/37beef4003f8bc42829a3442f26431d7c02b70a4.png)

  "This computes the distance between 2 distributions motivated by looking at how likely the 2nd distribution would be able to generate samples from the first distribution"

  **Cross-entropy Loss is very related to KL Divergence**

### Important Notes:

- KL Divergence is un-symmetric i.e the divergence depends on the distribution placed on the denominator. 

- In other words: Divergence of distribution1 wrt distribution2 is not same as divergence of distribution2 wrt distribution1.  
  
   ---

## ** Why take log of probability ?

  ***From the probabilities of ratio, why did we suddenly take log of ratio ??***

- The log of probabilities is closely related entropy. In [information theory](https://en.wikipedia.org/wiki/Information_theory "Information theory"), the **entropy** of a [random variable](https://en.wikipedia.org/wiki/Random_variable "Random variable") is the average level of "information", "surprise", or "uncertainty" inherent to the variable's possible outcomes.
  
  ![](kl-divergence/2023-06-25-16-12-19-image.png)
  
  ##### KL Divergence is also known as relative entropy between 2 distributions.
  
  For good reference to entropy, watch statquest video: 
  
  - [Entropy (for data science) Clearly Explained!!! - YouTube](https://www.youtube.com/watch?v=YtebGVx-Fxw) 

---
### Extra Note: 
20/05/2024

- Adding some random thinking: 
  - I waas just thinking, if I had 2 datasets with different strength (Not probability distribution right away), can I calculate drift with KL Divergence ?