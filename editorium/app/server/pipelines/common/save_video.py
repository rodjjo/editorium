from PIL import Image
from torchvision import transforms
import os
import torch
from pipelines.common.rife_model import rife_inferece_with_pil_4x
from pipelines.common.utils import export_to_video

to_tensors_transform = transforms.ToTensor()
to_pil_transform = transforms.ToPILImage()


def resize_pil_image(image: Image, hd=False):
    if hd:
        size = (image.size[0]/2, image.size[1]/2)
    else:
        if image.size[0] % 32 == 0 and image.size[1] % 32 == 0:
            size = (image.size[0], image.size[1])
        elif image.size[0] >= 1280:
            size = (1280, 768)
        else:
            size = (736, 496)
    if image.size == size:
        return image
    return image.resize(size)


def get_non_existing_path(output_path: str) -> str:
    if not os.path.exists(output_path):
        return output_path
    file_index = 0
    saved_path = output_path.replace(".mp4", "")
    while True:
        output_path = f"{saved_path}_{file_index}.mp4"
        if not os.path.exists(output_path):
            break
        file_index += 1
    return output_path


def save_video(frames, output_path, upscaler_model=None, fps_model=None,  fps=8):
    #if not os.path.exists("/app/output_dir/output/videos"):
    #    os.makedirs("/app/output_dir/output/videos", exist_ok=True)
    #output_path = os.path.join("/app/output_dir/output/videos", output_path)
    output_path = get_non_existing_path(output_path.replace(".mp4", ".fps.mp4"))
    if len(frames) == 0:
        print("No frames to save")
        return output_path, frames
    
    if upscaler_model:
        frames = [resize_pil_image(frames[i]) for i in range(len(frames))]
        for findex in range(len(frames)):
            frames[findex] = to_tensors_transform(frames[findex])
        
    if upscaler_model:
        print("Upscaling video")
        frames = utils.upscale(upscaler_model, torch.stack(frames).to('cuda'), 'cuda', output_device="cpu")
        frames = [resize_pil_image(to_pil_transform(frames[i].cpu()), True) for i in range(frames.size(0))]
    
    multiplier = 1
    if fps_model and fps < 15:
        print("Increasing video FPS")
        multiplier = 4
        frames = rife_inferece_with_pil_4x(fps_model, frames)

    fps *= multiplier
    print(f"Saving video {fps} fps, frame count {len(frames)}")
    export_to_video(frames, output_path, fps=fps)
    return output_path, frames

def save_video_list(path, video_list, seed=None):
    
    if seed is not None:
        # the seed is going to be part of the file name like this: /directory/filename-seed.mp4, so we need to modify the path to include the seed, remembering the path could have any extension
        path_dir = os.path.dirname(path)
        path_name = os.path.basename(path)
        path_name, path_ext = os.path.splitext(path_name)
        path = os.path.join(path_dir, f"{path_name}-{seed}{path_ext}")
        
            
    path_with_name_ext = path
    if path_with_name_ext.startswith('/app/output_dir/') is False:
        path_with_name_ext = os.path.join('/app/output_dir', path_with_name_ext)
    dir = os.path.dirname(path_with_name_ext) 
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    if not video_list:
        raise ValueError("It's required a image to save. #input=value")
    if type(video_list) is not list:
        raise ValueError("Video is suppose to be a list of images inside a list of videos #input=value")
    if len(video_list) == 0:
        raise ValueError("It's required a video to save. #input=value")
    if type(video_list[0]) is not list:
        raise ValueError("Video is suppose to be a list of images inside a list of videos #input=value")
    for video in video_list:
        if len(video) == 0:
            raise ValueError("It's required a image to save. #input=value")
        if type(video[0]) is not Image.Image:
            raise ValueError("Video is suppose to be a list of images inside a list of videos #input=value")
    
    if '.' not in path_with_name_ext:
        extension = 'mp4'
        path_with_name_ext = path_with_name_ext + f'.{extension}'
    else:
        extension = path_with_name_ext.split('.')[-1]

    for image_index, output in enumerate(video_list):
        video_path = path_with_name_ext.replace(f'.{extension}', f'_{image_index}.{extension}')
        try_index = 0
        while os.path.exists(video_path):
            video_path = path_with_name_ext.replace(f'.{extension}', f'_{try_index}.{extension}')
            try_index += 1
        save_video(output, video_path, fps=16)                
