# Deepstory
Deepstory is an artwork that incorporates Natural Language Generation(NLG) w/GPT-2, Text-to-Speech(TTS) w/Deep Convolutional TTS, speech to animation w/Speech driven animation and image animation w/First Order Motion Model into a media application.

To put it simply, it turns a text/generated text into a video where the character is animated to speak your story using his/her voice.

You can convert image into a video like this:
![result](https://raw.githubusercontent.com/thetobysiu/deepstory/master/result.gif)
It provides a comfortable web interface and backend written with flask to create your own story.

## Folder structure
```
Deepstory
│
├── modules
│   ├── dctts
│   │   ├── hparams.py
│   │   ├── audio.py
│   │   ├── layers.py
│   │   ├── ssrn.py
│   │   ├── text2mel.py
│   │   └── __init__.py
│   ├── fom
│   │   ├── dense_motion.py
│   │   ├── generator.py
│   │   ├── keypoint_detector.py
│   │   ├── util.py
│   │   ├── animate.py
│   │   ├── __init__.py
│   │   └── sync_batchnorm
│   │       ├── batchnorm.py
│   │       ├── comm.py
│   │       ├── __init__.py
│   │       └── replicate.py
│   └── sda
│       ├── encoder_audio.py
│       ├── encoder_image.py
│       ├── img_generator.py
│       ├── rnn_audio.py
│       ├── utils.py
│       ├── sda.py
│       └── __init__.py
├── data
│   ├── dctts
│   │   ├── Geralt
│   │   │   ├── ssrn.pth
│   │   │   ├── t2bm.pth
│   │   │   └── t2m.pth
│   │   └── Yennefer
│   │       ├── ssrn.pth
│   │       └── t2m.pth
│   ├── sda
│   │   ├── grid.dat
│   │   └── image.bmp
│   ├── fom
│   │   ├── vox-256.yaml
│   │   ├── vox-adv-256.yaml
│   │   ├── vox-adv-cpk.pth.tar
│   │   └── vox-cpk.pth.tar
│   ├── images
│   │   ├── Geralt
│   │   │   ├── 0.jpg
│   │   │   ├── 1.jpg
│   │   │   └── 2.jpg
│   │   └── Yennefer
│   │       ├── 0.jpg
│   │       ├── 1.jpg
│   │       └── 2.jpg
│   └── gpt2
│       ├── w3book
│       │   ├── config.json
│       │   ├── merges.txt
│       │   ├── pytorch_model.bin
│       │   └── vocab.json
│       └── dialog53000
│           ├── config.json
│           ├── merges.txt
│           ├── pytorch_model.bin
│           └── vocab.json
├── static
│   ├── bootstrap
│   │   ├── css
│   │   │   └── bootstrap.min.css
│   │   └── js
│   │       └── bootstrap.min.js
│   ├── js
│   │   └── jquery.min.js
│   └── css
│       └── styles.css
├── animator.py
├── generate.py
├── util.py
├── voice.py
├── app.py
├── requirements.txt
├── README.md
├── output
├── templates
│   ├── models.html
│   ├── gpt2.html
│   ├── status.html
│   ├── index.html
│   ├── sentences.html
│   └── deepstory.js
├── deepstory.py
└── export
    ├── base.mp4
    └── animated.mp4
```

## Demo
Available at my blog: https://blog.thetobysiu.com/video/

## Models
They are available at the google drive version of this project. The voice model are not publicly released yet, but you can check my other repo to recreate the same result.

https://drive.google.com/drive/folders/1AxORLF-QFd2wSORzMOKlvCQSFhdZSODJ?usp=sharing

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
There are other repos of tools that I created to preprocess the files. They can be found in my profile.