import numpy as np

def stitch_predictions(image, preds, x_vals, y_vals):
    n_channels, patch_size, patch_size = preds[0].shape 
    
    stitched_image = np.zeros((n_channels, image.shape[0], image.shape[1]))
    
    counter = np.zeros((image.shape[0], image.shape[1]))
          
    idx = 0
    for y in y_vals:
        for x in x_vals:
            stitched_image[:, y : y + patch_size, x: x + patch_size] += preds[idx]
            counter[y : y + patch_size, x: x + patch_size] += 1            
            idx += 1
            
    counter[counter == 0] = 1
    
    stitched_image /= counter 
    
    return stitched_image