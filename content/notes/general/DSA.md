---
title: "DSA Basic Notes:"
weight: 10
# bookFlatSection: false
# bookToc: true
# bookHidden: false
bookCollapseSection: false
bookComments: true
# bookSearchExclude: false
---

# DSA

### 1. Bitwise Operations for addition/subtraction/division/multiplcation:
Addition without using + operator: 
- a + b = ?
- xor_val = a ^ b
- and_val = a & b << 1
- while and_val != 0:
    - xor_val = xor_val ^ and_val
    - and_val = xor_val & and_val << 1
- return xor_val 

https://leetcode.com/problems/sum-of-two-integers/description/

**Multiplication (can be applied to division using right shift instead of left shift):**

https://stackoverflow.com/questions/2776211/how-can-i-multiply-and-divide-using-only-bit-shifting-and-adding/2777225#2777225 

![](DSA/2024-05-25-18-04-52.png)

### 2. Heap: 

**Heap is nothing but a priority queue**:
- Python has an inbuilt module for heap named: **heapq**
    - **Methods:** 
        - heapq.heappuush()
        - heapq.heappop()

- Problem on leetcode: https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended/description/

- **Max Heap in Python**: 
    - Python heapq by default only supports **min-heap**.
    - To get a max-heap, multiply each number in the heap with **-1**.




