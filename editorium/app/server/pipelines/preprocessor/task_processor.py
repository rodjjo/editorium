
import cv2
import numpy as np
from PIL import Image
import PIL.ImageOps
import gc
from controlnet_aux import HEDdetector, OpenposeDetector
from transformers import pipeline

from pipelines.preprocessor.line_art import LineartDetector
from pipelines.preprocessor.manga_line import MangaLineExtration
from pipelines.common.task_result import TaskResult


def report(message):
    print(f'[Image pre-processor] - {message}')


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()


def resize_image_with_pad(input_image, resolution):
    img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad

def image_to_canny(input_image):
    def do_it():
        low_threshold = 100
        high_threshold = 200
        image = cv2.Canny(np.array(input_image), low_threshold, high_threshold)
        
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = PIL.ImageOps.invert(Image.fromarray(image))
        return image

    result = do_it()
    gc.collect()
    return result 


def image_to_pose(input_image):
    def do_it():
        openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        image = openpose(input_image)
        return image

    result = do_it()
    gc.collect()
    return result


def image_to_scribble(input_image):
    def do_it():
        hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        image = PIL.ImageOps.invert(hed(input_image, scribble=True))
        return image

    result = do_it()
    gc.collect()
    return result

def image_to_depth(input_image):
    def do_it():
        depth_estimator = pipeline('depth-estimation')
        
        image = depth_estimator(input_image)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image

    result = do_it()
    gc.collect()
    return result


def image_to_lineart(input_image):
    def do_it():
        # la = LineartDetector(LineartDetector.model_default)
        la = LineartDetector(LineartDetector.model_coarse)
        image, remove_pad = resize_image_with_pad(np.array(input_image), 512)
        image = la(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image[image > 96] = 255
        # image[image < 32] = 0
        image = remove_pad(image)
        image = Image.fromarray(image)
        return image
    result = do_it()
    gc.collect()
    return result


def image_to_mangaline(input_image):
    def do_it():
        la = MangaLineExtration()
        image, remove_pad = resize_image_with_pad(np.array(input_image), 512)
        image = la(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = remove_pad(image)
        image = Image.fromarray(image)
        return image
    result = do_it()
    gc.collect()
    return result

def background2mask(input_image):
    from rembg import remove, new_session
    result = remove(
        input_image,
        session=new_session('u2net_human_seg'),
        only_mask=False,
        # alpha_matting=True,
        # alpha_matting_erode_size=15
    )
    background_image = Image.new('RGB', result.size, (255, 255, 255))
    background_image.paste(Image.new('RGB', result.size, (0, 0, 0)), (0, 0), result)
    return background_image


def pre_process_image(control_type: str, im):
    if type(im) is str:
        im = Image.open(im)
    if control_type == 'canny':
        report("extracting canny edges")
        return image_to_canny(im)
    elif control_type == 'scribble':
        report("converting to scribbles")
        return image_to_scribble(im)
    elif control_type == 'pose':
        report("extracting pose")
        return image_to_pose(im)
    elif control_type == 'depth':
        report("extracting depth")
        return image_to_depth(im)
    elif control_type == 'lineart':
        report("extracting lineart")
        return image_to_lineart(im)
    elif control_type == 'mangaline':
        report("extracting lineart")
        return image_to_mangaline(im)
    elif control_type == 'background':
        return background2mask(im)
    return None


def process_workflow_task(base_dir: str, name: str, input: dict, config: dict, callback: callable):
    images = input.get('image', {}).get('output', None) or input.get('image', {}).get('result', None)
    if images is None:
        raise ValueError("It's required a image pre-process the image #config.input=value")
    results = []
    for index, image in enumerate(images):
        if type(image) is str:
            image = Image.open(image)
        image = pre_process_image(config['control_type'], image)
        if not image:
            raise ValueError("The image pre-processing failed: Invalid control type")
    
        path2save = f'{base_dir}/{name}_{config["control_type"]}_{index}.png'
        image.save(path2save)
        results.append(image)
        
    return TaskResult(image, path2save).to_dict()
