---
title: "Daily-Scribble-2024"
weight: 10
bookCollapseSection: false
bookComments: true
---

# Daily Scribble: 

Scribbling the content seen in the day: 

## January: 

- **vLLM:** Easy, fast, and cheap LLM serving for everyone
    - https://github.com/vllm-project/vllm 
    - Invented the concept of "**PagedAttention**"
    - uses **xFormer** internally, so indirectly uses FlashAttention and FlashAttention2 as well 
    - Has support for **continuous batching** (very useful for transformer/decoder architecture)
    - When released, claims to be much much faster than huggingface TGI. Now, even HF uses paged-attention for inference 

- **Fooocus:**
    - Fooocus is an image generating software. Fooocus is a rethinking of Stable Diffusion and Midjourney’s designs. 
    - **Made by the controlnet authors** 
    - Has direct support for colab and huggingface. **Made on gradio**
    - Looks quite good and easy to use: 
        - On early analysis, it looks like: it can do inpainting/outpainting and image-super-resolution as well. 