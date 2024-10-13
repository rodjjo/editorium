from PIL import Image
from torchvision import transforms
import os
import torch
from pipelines.common.rife_model import rife_inference_with_latents
from pipelines.common.utils import export_to_video

to_tensors_transform = transforms.ToTensor()
to_pil_transform = transforms.ToPILImage()


def resize_pil_image(image: Image, hd=False):
    if hd:
        size = (1072, 720)
    else:
        size = (736, 496)
    return image.resize(size)


def get_non_existing_path(output_path: str) -> str:
    file_index = 0
    saved_path = output_path.replace(".mp4", "")
    while os.path.exists(output_path):
        output_path = f"{saved_path}_{file_index}.mp4"
        file_index += 1
    return output_path


def save_video(frames, output_path, upscaler_model, fps_model, should_upscale=False, fps=8):
    if not os.path.exists("/app/output_dir/output/videos"):
        os.makedirs("/app/output_dir/output/videos", exist_ok=True)
    output_path = os.path.join("/app/output_dir/output/videos", output_path)
    output_path = get_non_existing_path(output_path.replace(".mp4", ".fps.mp4"))
    
    frames = [resize_pil_image(frames[i]) for i in range(len(frames))]
    for findex in range(len(frames)):
        frames[findex] = to_tensors_transform(frames[findex])
        if not should_upscale:
            frames[findex] = frames[findex].unsqueeze(0)
        
    if should_upscale:
        print("Upscaling video")
        frames = utils.upscale(upscaler_model, torch.stack(frames).to('cuda'), 'cuda', output_device="cpu")
        frames = [to_tensors_transform(resize_pil_image(to_pil_transform(frames[i].cpu()), True)).unsqueeze(0) for i in range(frames.size(0))]

    print("Increasing video FPS")
    multiplier = 2
    frames = rife_inference_with_latents(fps_model, torch.stack(frames))
    if fps < 16:
        multiplier = 4
        frames = rife_inference_with_latents(fps_model, torch.stack(frames))
    frames = [to_pil_transform(f[0]) for f in frames]

    print("Saving video")
    export_to_video(frames, output_path, fps=fps * multiplier)
