import torch
from diffusers.image_processor import VaeImageProcessor
from torch.nn import functional as F
import cv2
from pipelines.common import utils
from pipelines.cogvideo.rife.pytorch_msssim import ssim_matlab
import numpy as np
import logging
import skvideo.io
from pipelines.cogvideo.rife.RIFE_HDv3 import Model

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


def pad_image(img, scale):
    return img
    #_, _, h, w = img.shape
    #tmp = max(32, int(32 / scale))
    #ph = ((h - 1) // tmp + 1) * tmp
    #pw = ((w - 1) // tmp + 1) * tmp
    #padding = (0, 0, pw - w, ph - h)
    #return F.pad(img, padding)


def make_inference(model, I0, I1, upscale_amount, n):
    middle = model.inference(I0, I1, upscale_amount)
    if n == 1:
        return [middle]
    first_half = make_inference(model, I0, middle, upscale_amount, n=n // 2)
    second_half = make_inference(model, middle, I1, upscale_amount, n=n // 2)
    if n % 2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]


def resize_frame(frame):
    return cv2.resize(frame, (736, 496), interpolation=cv2.INTER_CUBIC)


@torch.inference_mode()
def ssim_interpolation_rife(model, samples, exp=1, upscale_amount=1, output_device="cpu"):
    output = []
    # [f, c, h, w]
    prev_shape = None
    for b in range(samples.shape[0]):
        frame = samples[b : b + 1]
        _, _, h, w = frame.shape
        I0 = samples[b : b + 1]
        I1 = samples[b + 1 : b + 2] if b + 2 < samples.shape[0] else samples[-1:]
        I1 = pad_image(I1, upscale_amount)
        # [c, h, w]
        I0_small = F.interpolate(I0, (32, 32), mode="bilinear", align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)

        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        if ssim > 0.996:
            I1 = I0
            I1 = pad_image(I1, upscale_amount)
            I1 = make_inference(model, I0, I1, upscale_amount, 1)

            I1_small = F.interpolate(I1[0], (32, 32), mode="bilinear", align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            frame = I1[0]
            I1 = I1[0]

        tmp_output = []
        if ssim < 0.2:
            for i in range((2**exp) - 1):
                tmp_output.append(I0)

        else:
            tmp_output = make_inference(model, I0, I1, upscale_amount, 2**exp - 1) if exp else []

        frame = pad_image(frame, upscale_amount)
        tmp_output = [frame] + tmp_output
        for i, frame in enumerate(tmp_output):
            output.append(frame.to(output_device))
    return output


def load_rife_model(model_path):
    model = Model()
    model.load_model(model_path, -1)
    model.eval()
    return model


# Create a generator that yields each frame, similar to cv2.VideoCapture
def frame_generator(video_capture):
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        yield frame
    video_capture.release()


def rife_inference_with_path(model, video_path, path=None, upscale_amount=1):
    video_capture = cv2.VideoCapture(video_path)
    tot_frame = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    pt_frame_data = []
    pt_frame = skvideo.io.vreader(video_path)
    frame_count = 0
    for frame in pt_frame:
        frame_count += 1
        pt_frame_data.append(
            torch.from_numpy(np.transpose(resize_frame(frame), (2, 0, 1))).to("cpu", non_blocking=True).float() / 255.0
        )

    pt_frame = torch.from_numpy(np.stack(pt_frame_data))
    pt_frame = pt_frame.to(device)
    pbar = utils.ProgressBar(tot_frame, desc="RIFE inference")
    frames = ssim_interpolation_rife(model, pt_frame, upscale_amount=upscale_amount)
    frames = torch.stack([frames[i].squeeze(0) for i in range(len(frames))]).to(device)
    frames = ssim_interpolation_rife(model, frames, upscale_amount=upscale_amount)
    frame_count *= 2

    pt_image = torch.stack([frames[i].squeeze(0) for i in range(len(frames))])
    image_np = VaeImageProcessor.pt_to_numpy(pt_image)  # (to [49, 512, 480, 3])
    image_pil = VaeImageProcessor.numpy_to_pil(image_np)
    fps = 8 * (2 ** (frame_count / 49))
    print(f"frame_count: {frame_count} fps: {fps}")
    video_path = utils.save_video(image_pil, fps=fps, path=path)
    if pbar:
        pbar.update(1)
    return video_path


def rife_inference_with_latents(model, latents):
    rife_results = []
    latents = latents.to(device)
    for i in range(latents.size(0)):
        #  [f, c, w, h]
        latent = latents[i]
        frames = ssim_interpolation_rife(model, latent)
        pt_image = torch.stack([frames[i].squeeze(0) for i in range(len(frames))])  # (to [f, c, w, h])
        rife_results.append(pt_image)
    return rife_results


def rife_inferece_with_pil_4x(model, pil_images):
    pt_frame_data = []
    for frame in pil_images:
        frame = np.array(frame.resize((736, 496)))
        pt_frame_data.append(
            torch.from_numpy(np.transpose(frame, (2, 0, 1))).to("cpu", non_blocking=True).float() / 255.0
        )
    
    pt_frame = torch.from_numpy(np.stack(pt_frame_data))
    pt_frame = pt_frame.to(device)
    
    frames = ssim_interpolation_rife(model, pt_frame, upscale_amount=1)
    frames = torch.stack([frames[i].squeeze(0) for i in range(len(frames))]).to(device)
    frames = ssim_interpolation_rife(model, frames, upscale_amount=1)
    
    pt_image = torch.stack([frames[i].squeeze(0) for i in range(len(frames))])
    image_np = VaeImageProcessor.pt_to_numpy(pt_image)  # (to [49, 512, 480, 3])
    image_pil = VaeImageProcessor.numpy_to_pil(image_np)
    
    return image_pil
    


if __name__ == '__main__':
    import sys
    model = load_rife_model("model_rife")
    rife_inference_with_path(model, sys.argv[1], path=sys.argv[2], upscale_amount=1)
    #rife_inference_with_path(model, sys.argv[2], path=f'{sys.argv[2]}-32fps.mp4', upscale_amount=1)
