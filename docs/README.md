# A toolbelt form image, video and audio manipulation

This tool is a command line client and a server to generate images, videos and music in your local computer.
It provides a server that runs background tasks and a cli called editorium to manage the tasks on that server.


## WIP (Work In Progress)

TODOS  
[x] Open the sources  
[x] Add CogVideoX basic command  
[ ] Create documentation explaining each command  

## Requirements

You need to install the following components at your ubuntu operating system:

* Docker
* Nvidia runtime for docker

## Installation

```
sudo apt-get update && apt-get install pipx
pipx install --force git+https://github.com/rodjjo/video-editor-scripts.git
```

## Server

Before running the cli, it's necessary to start the server.
It's required to define the following paths when we start the server:

* **--models-dir** - Path where the models are stored. I this directory will be the olfline models, not managed by hugging face.
* **--path=** - Path of the root directory of the server. This is going to be the output directory and also where the files the server uses will be.
* **--cache-dir** - It's where the cache directory will be located. It's detault to ~/.cache and it's necessary for hugginface being able to cache the files.

Starting the server:
```bash
editorium-cli server run --docker-image --models-dir=~/my-offline-models --path=~/my-cogvideo-root-dir
```

## Prompt file format

To start you can create a prompt file at `~/my-cogvideo-root-dir` (the root path you defined to start the server).
The file name could be `~/my-cogvideo-root-dir/prompts.editorium`:
```text
#start
#config.steps=50
#config.seed=-1
#config.num_videos_per_prompt=1
#config.loop=false
#config.generate_type=i2v
#config.should_upscale=false
#config.stoponthis=false
#config.strength=55
#config.count=1
#config.quant=false
#config.use_pyramid=false
Your prompt goes here.
#image
#comment The path bellow should allways tart with /app/output_dir and it's the root of our service 
/app/output_dir/<relative_path_to_the_image_or_Video
#end
```
You can add multiples entries starting with `#start` and finishing with `#end`, each block is going to be a prompt.


The you can start the cli on the terminal

```bash
# from the container perspective the prompts.editorium file will be located in /app/output_dir/ inside the container
# As you started the server with --path=~/my-cogvideo-root-dir, you should save prompts.editorium there.
editorium cogvideo generate-from-file --prompts-path=prompts.editorium
# the full path para prompts.editorium:
editorium cogvideo generate-from-file --prompts-path=/app/output_dir/prompts.editorium
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
