import numpy as np

def pad_image(image, target_size):
    h, w, c = image.shape
    new_h = ((h + target_size - 1) // target_size) * target_size
    new_w = ((w + target_size - 1) // target_size) * target_size

    pad_h = new_h - h
    pad_w = new_w - w    
    
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='wrap')
    return padded.astype(np.uint8)



