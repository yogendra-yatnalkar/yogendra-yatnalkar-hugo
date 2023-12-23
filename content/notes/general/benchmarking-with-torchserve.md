---
title: "Benchmarking Inference with Torchserve"
weight: 10
# bookFlatSection: false
# bookToc: true
# bookHidden: false
bookCollapseSection: false
bookComments: true
# bookSearchExclude: false
---

# Benchmarking Inference with Torchserve

|             |            |
| ----------- | ---------- |
| Last Edited | 24/12/2023 |

---

- **Instance Type: ml.g4dn.xlarge**
    - GPU: Nvidia T4
    - vCPU no: 4
    - CPU memory: 16 GB
    - GPU memory: 16 GB

- **Max RPS achieved: 32**
    - With various different configuration ranging from min/max worker = 1 to 4 and batch-size 4 to 32, the max RPS possible was only 32. 

- **Max Response time at 95th percentile: ~5-6 sec**

---

![Alt text](benchmarking-with-torchserve/image.png)

![Alt text](benchmarking-with-torchserve/image-3.png)

![Alt text](benchmarking-with-torchserve/image-1.png)


![Alt text](benchmarking-with-torchserve/image-2.png)