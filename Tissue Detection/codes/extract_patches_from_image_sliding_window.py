import numpy as np

def extract_patches_from_image_sliding_window(image, patch_size, stride, return_coordinates = False):
    
    if len(image.shape) == 3:
        h, w, _  = image.shape
    elif len(image.shape) == 2:
        h, w  = image.shape
    else:
        print('input image should a 2- or 3- dimensional NumPy array')
        return []
    
    if patch_size <= 0:
        patch_size = min(h, w)
        print(f'patch size was smaller than zero, changed to min(h, w) = {patch_size}')
    
    if stride <= 0:
        stride = min(h, w)  
        print(f'stride was smaller than zero, changed to min(h, w) = {stride}')
    
    patch_size_x = patch_size
    patch_size_y = patch_size
    
    stride_x = stride    
    stride_y = stride    
    
    if patch_size_y > h:
        patch_size_y = h
        stride_y = h
        
    if patch_size_x > w:
        patch_size_x = w
        stride_x = w

    x = np.arange(0, w, stride_x)
    y = np.arange(0, h, stride_y)
    
    x = [i for i in x if i + patch_size_x <= w]
    y = [i for i in y if i + patch_size_y <= h]
    
    x.append(w - patch_size_x)
    y.append(h - patch_size_y)
    
    x_vals = np.unique(x)
    y_vals = np.unique(y)
    
    patches = []
    for y in y_vals:
        for x in x_vals:
            patches.append(image[y : y + patch_size_y, x : x + patch_size_x])
    
    if return_coordinates:
        return patches, x_vals, y_vals
    else:
        return patches