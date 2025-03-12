# A toolbelt form image, video and audio manipulation

This tool is a command line client and a server to generate images, videos and music in your local computer.
It provides a server that runs background tasks and a cli called editorium to manage the tasks on that server.


## WIP (Work In Progress)

TODOS  
[x] Open the sources  
[x] Add CogVideoX basic commands  
[x] Add Pyramid Flow basic commands  
[x] Add Flux  
[x] Add Stable Diffusion 1.5  
[x] Add SDXL
[x] Add SD 3.5
[x] Add Omnigen
[ ] Replace ffmpeg command by builtin scripts that manipulates the video.  
[ ] Create documentation explaining each command  

## Requirements

You need to install the following components at your ubuntu operating system:

* Docker
* Nvidia runtime for docker
* ffmpeg  (only if you want to use editorium ffmpeg commands)

```bash
sudo apt-get update -qq && sudo apt-get -y install \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libavfilter-dev \
    libavdevice-dev \
    libswresample-dev \
    libswscale-dev \
    libgl-dev \
    lcov \
    libssl-dev \
    python3-pip

```

## Installation

```
sudo apt-get update && apt-get install pipx
pipx install --force git+https://github.com/rodjjo/editorium.git
```

## Server

Before running the cli, it's necessary to start the server.
It's required to define the following paths when we start the server:

* **--models-dir** - Path where the models are stored. In this directory will be the olfline models, not managed by hugging face.
* **--path=** - Path of the root directory of the server. This is going to be the output directory and also where the files the server uses will be.
* **--cache-dir** - It's where the cache directory will be located. It's detault to ~/.cache and it's necessary for hugginface being able to cache the files.

Starting the server:
```bash
editorium-cli server run --docker-image --models-dir=~/my-offline-models --path=~/my-awesome-server
```

## Prompt file format 

To start you can create a prompt file anywhere,  
however the image paths should be relative to the root directory of the server  
for example: `~/my-awesome-server` (the root path you defined to start the server).  

The file format for Cogvideo:
```text
#start
#cogvideo.steps=50
#cogvideo.seed=-1
#cogvideo.num_videos_per_prompt=1
#cogvideo.loop=false
#cogvideo.generate_type=i2v
#cogvideo.should_upscale=false
#cogvideo.stoponthis=false
#cogvideo.strength=55
#cogvideo.count=1
#cogvideo.quant=false
#cogvideo.use_pyramid=false
Your prompt goes here.
#image
#comment The path bellow should allways tart with /app/output_dir and it's the root of our service 
/app/output_dir/<relative_path_to_the_image_or_Video
#end
```

The file format for Pyramid Flow:
```text
#start
#pyramid.generate_type=i2v
#pyramid.num_inference_steps=20,20,20
#pyramid.video_num_inference_steps=10,10,10
#pyramid.height=768
#pyramid.width=1280
#pyramid.temp=16
#pyramid.guidance_scale=9.0
#pyramid.video_guidance_scale=5.0
#pyramid.seed=-1
#pyramid.use768p_model=true
Your prompt goes here.
#image
#comment The path bellow should allways tart with /app/output_dir and it's the root of our service 
/app/output_dir/<relative_path_to_the_image_or_Video
#end
```
You can add multiples entries starting with `#start` and finishing with `#end`, each block is going to be a prompt.



The you can start the cli on the terminal

```bash
editorium cogvideo generate-from-file --prompts-path=prompts.editorium
```

## Help

The cli provides help for each command
```bash
editorium --help
editorium server --help
editorium server run --help
editorium cogvideo --help
editorium cogvideo generate-from-file --help
```


## References

[THUDM/CogVideo](https://github.com/THUDM/CogVideo)  
[ComfyUI-CogVideoXWrapper](https://github.com/kijai/ComfyUI-CogVideoXWrapper)  
[diffusers](https://github.com/huggingface/diffusers/)  
[Pyramid Flow](https://github.com/jy0205/Pyramid-Flow)  
