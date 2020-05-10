# Deepstory
Deepstory is an artwork that incorporates Natural Language Generation(NLG) w/GPT-2, Text-to-Speech(TTS) w/Deep Convolutional TTS, speech to animation w/Speech driven animation and image animation w/First Order Motion Model into a media application.

It is a flask web app with a comfortable interface for creating your own story.

## Demo
Available at my blog: https://blog.thetobysiu.com/video/

## Requirements
It is required to have an nvidia GPU with at least 4GB of VRAM to run this project

## Credits
https://github.com/tugstugi/pytorch-dc-tts

https://github.com/DinoMan/speech-driven-animation

https://github.com/AliaksandrSiarohin/first-order-model

https://github.com/huggingface/transformers

## Notes
The whole project uses PyTorch, while tensorflow is listed in requirements.txt, it was used for transformers to convert a model trained from gpt-2-simple to a Pytorch model. 
 
Only the files inside modules folder are slightly modified from the original. The remaining files are all written by me, except some parts that are referenced.

## Training models
There are other repos for tools that I created to preprocess the files. They can be found in my profile.