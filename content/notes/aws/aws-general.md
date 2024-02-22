---
title: "AWS General"
weight: 10
# bookFlatSection: false
# bookToc: true
# bookHidden: false
bookCollapseSection: false
bookComments: true
# bookSearchExclude: false
---

# Notes on AWS-General:
- Not sure what all I will add but will try to add things which I learnt new on the go. 

## Provisioned Lambda Function Concurrency:  
- Added on 22/02/2024
- **Concurrency**: Number of in-flight lambda requests which can be handled at the same time
    - **Reserved**: Max no of concurrent requests allocated 
        - This requires 0 added charge 
    - **Provisioned**: Number of pre-initialized execution env allocated to the functions.
        - There are additional charges