import numpy as np

def pad_image_center_crop(image, patch_size = 512):
    h, w, c = image.shape

    pad_h = patch_size // 2
    pad_w = patch_size // 2
    
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='wrap')
    return padded.astype(np.uint8)


