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

- **torchserve:**
    - Started with g4dn.xlarge instance type and model: ViT L16. Later updgraded to G4Dn.2xlarge to check the effect of increasing the CPU number and CPU memory. 
        - Played a lot of dynamic batching and worker count on single GPU. Came to 2 conclusion: 
            1. Dynamic batching helps (see below)
            2. On dynamic batching, once GPU utlization becomes 100%, no one can help later
        - **Conclusion**: Once gpu memory utilization is full, whatever we do, it wont increasing the actual load. 
        - **Importance of AWS elastic inference** (GPU as a service): Even though this ViT L16 is a medium level model with I believe update 300M params, 1 model only uses around 20 to 30% of the memory. With dynamic batching, the utilization is 100% even with 1 worker, so we are effectively not using memory at the fullest. 
    - Does Dynamic batching helps ? On g4dn:
        - Yes. When it was set to 1, the max RPS was 21
            - When batch-size was 1 but workers was also 1, the gpu utilization was around 85%
            - When workers were increased to 4 and batch-size was still 1, the gpu utilization became 100, but that did not affect the RPS. 
            - In both cases, the response time was slightly faster than dynamic batching. 
        - When it was set to 32, the max RPS was 32 
            - when dynamic batching is on, whatever is the worker count, it does no affect the RPS (for this model)