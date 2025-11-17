import numpy as np

def stitch_predictions_center_crop(image, preds, x_vals, y_vals):
    n_channels, patch_size, patch_size = preds[0].shape 
    
    step = patch_size // 4
    
    window = patch_size // 2
    
    stitched_image = np.zeros((n_channels, image.shape[0], image.shape[1]))

    idx = 0
    for y in y_vals:
        for x in x_vals:            
            stitched_image[:, y + step: y + step + window , 
                           x +step : x + step + window] = preds[idx][:, step : step + window, 
                                                                     step : step + window]            
            idx += 1
            
    return stitched_image


