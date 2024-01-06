---
title: "AI Assisted Video Generation with White-Board Animations"
weight: -1
# bookFlatSection: false
# bookToc: true
bookHidden: false
bookCollapseSection: false
bookComments: true
# bookSearchExclude: false

description : "AI Assisted Video Generation (Whiteboard Animations): Story of Shepherd Boy and the Wolf"
tags : [
    "Yogendra Yatnalkar",
    "ML",
    "Computer Vision",
    "GenAI",
]
---

# AI Assisted Video Generation with White-Board Animations

## AI Video Generation for: Story of Shepherd Boy and the Wolf
---

**Github Link:** https://github.com/yogendra-yatnalkar/storyboard-ai 

Last Edited: 06/01/2024

---

## Youtube Video

{{< youtube iSb1HJXRO04 >}}

---

## HOW THE VIDEO WAS MADE ? 

- The story was taken from internet.
- A 4 line summary of the story was generated using ChatGPT
- For each summary line, a corresponding image was generated using Stable Diffusion
- In each image (4 in our case), the important object were masked using MetaAI SAM 
- I have developed a custom image to white-board animation code which converted the images to videos (This was the most time-consuming part of the development process). 
- The audio was generated using gTTS (Google Text-to-Speech)
- The sub-titles were generated with the help of OpenAI whisper 
- All the different audios, videos and subtitles were somehow synchronized using FFMPEG and OpenCV (This part gave a lot of pain🥲... ALL HAIL FFMPEG🙌)