# A toolbelt form image, video and audio manipulation


## Requirements

* Docker
* Nvidia runtime for docker

## Installation

```
sudo apt-get update && apt-get install pipx
pipx install --force git+https://github.com/rodjjo/video-editor-scripts.git
```

## CogVideoX


## Prompt file format

```text
#start
#config.steps=50
#config.seed=-1
#config.num_videos_per_prompt=1
#config.loop=false
#config.generate_type=i2v
#config.should_upscale=false
#config.stoponthis=true
#config.strength=55
#config.count=1
#config.quant=false
#config.use_pyramid=false
Your prompt goes here.
#image
/app/output_dir/<relative_path_to_the_image_or_Video
#end
```