from PIL import Image
import numpy as np
import cv2
from skimage import exposure
from blendmodes.blend import blendLayers, BlendType


def color_correction(new_img, original_image):
    original_colors = cv2.cvtColor(np.asarray(original_image.copy()), cv2.COLOR_RGB2LAB)
    new_image_colors = cv2.cvtColor(np.asarray(new_img.copy()), cv2.COLOR_RGB2LAB)
    
    image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
        new_image_colors,
        original_colors,
        channel_axis=2
    ), cv2.COLOR_LAB2RGB).astype("uint8"))

    image = blendLayers(image, new_img, BlendType.LUMINOSITY)

    return image.convert('RGB')

def color_correction_mask(new_img, original_image, mask):
    # convert the pil image new_img into a numpy array
    new_img = np.array(new_img)
    original_image = np.array(original_image)
    mask = np.array(mask.convert("L"))
    
    fg = new_img.astype(np.float32)
    bg = original_image.copy().astype(np.float32)
    w = mask[:, :, None].astype(np.float32) / 255.0
    y = fg * w + bg * (1 - w)
    
    # y is an array of float that contains values from 0.0 to 1.0
    # convert it to uint8 0-255
    y = y.clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(y)
    
    