import cv2
import numpy as np
import scipy.signal
import os
import openslide
from skimage.morphology import remove_small_objects
from PIL import Image
from skimage import color    
from skimage.filters import gaussian
from skimage import filters, morphology

from codes.predict_nn_model_on_batch import predict_nn_model_on_batch
from codes.pad_image import pad_image
from codes.extract_patches_from_image_sliding_window import extract_patches_from_image_sliding_window
from codes.stitch_predictions import stitch_predictions
from codes.pad_image_center_crop import pad_image_center_crop
from codes.stitch_predictions_center_crop import stitch_predictions_center_crop

def jpeg_compression_transform(image, quality=80):   
    image = np.array(image)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, image = cv2.imencode('.jpg', image, encode_param)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)   
    return Image.fromarray(image)

def tissue_detection(slide_path, slide_name, model = 'pathprofiler', method = 'default'):
    
    if model == 'pathprofiler' and method == 'default':   
        
        def read_region(slide, x_y, level, tile_size, downsample_level_0=False):
            if isinstance(tile_size, int):
                tile_size = (tile_size, tile_size)
            x, y = x_y
            if downsample_level_0 and level > 0:
                downsample = round(slide.level_dimensions[0][0] / slide.level_dimensions[level][0])
                x, y = x * downsample, y * downsample
                tile_w, tile_h = tile_size[0] * downsample, tile_size[1] * downsample
                width, height = slide.level_dimensions[0]
            else:
                tile_w, tile_h = tile_size
                width, height = slide.level_dimensions[level]
                
            tile_w = tile_w + x if x < 0 else tile_w
            tile_h = tile_h + y if y < 0 else tile_h
            x = max(x, 0)
            y = max(y, 0)
            tile_w = width - x if (x + tile_w > width) else tile_w
            tile_h = height - y if (y + tile_h > height) else tile_h
            tile  = slide.read_region((x,y), 0 if downsample_level_0 else level, (tile_w, tile_h))
            tile = np.array(tile.convert("RGB")).astype('uint8')
            if downsample_level_0 and level > 0:
                tile_w = tile_w // downsample
                tile_h = tile_h // downsample
                x = x // downsample
                y = y // downsample
                tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_CUBIC)        
                   
            padding = [(y-x_y[1],tile_size[1]-tile_h+min(x_y[1],0)), (x-x_y[0], tile_size[0]-tile_w+min(x_y[0],0))]
            tile = np.pad(tile, padding + [(0,0)]*(len(tile.shape)-2), 'constant', constant_values=0)    
            
            return tile

        def get_best_level_for_downsample(slide, downsample):
            if downsample < slide.level_downsamples[0]:
                return 0
            for i in range(1, slide.level_count):
                if downsample < slide.level_downsamples[i]:
                    return i - 1    
            return slide.level_count - 1

        def get_downsampled_slide(slide, dims):
            downsample = min(a / b for a, b in zip(slide.level_dimensions[0], dims))       
            level = get_best_level_for_downsample(slide, downsample)    
            tile_size = slide.level_dimensions[level]
            slide_downsampled = read_region(slide, (0,0), level, tile_size)    
            slide_downsampled = cv2.resize(slide_downsampled, dims, interpolation=cv2.INTER_CUBIC)    
            return slide_downsampled

        def pad_img_pathprofiler(img, patch_size = 512, subdivisions = 2):
            aug = int(round(patch_size * (1 - 1.0 / subdivisions)))
            more_borders = ((aug, aug), (aug, aug), (0, 0))
            ret = np.pad(img, pad_width=more_borders, mode='reflect')
            return ret

        def extract_patches(img, patch_size = 512, subdivisions = 2):
            step = int(patch_size / subdivisions)
            row_range = range(0, img.shape[0] - patch_size + 1, step)
            col_range = range(0, img.shape[1] - patch_size + 1, step)

            patches = []
            for row in row_range:
                for col in col_range:
                    patches.append(img[row : row + patch_size, col : col + patch_size, :])
            return patches

        def spline_window(patch_size, effective_window_size, power=2):
            """
            Squared spline (power=2) window function:
            https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
            """
            window_size = effective_window_size
            intersection = int(window_size / 4)
            wind_outer = (abs(2 * (scipy.signal.windows.triang(window_size))) ** power) / 2
            wind_outer[intersection:-intersection] = 0

            wind_inner = 1 - (abs(2 * (scipy.signal.windows.triang(window_size) - 1)) ** power) / 2
            wind_inner[:intersection] = 0
            wind_inner[-intersection:] = 0

            wind = wind_inner + wind_outer
            wind = wind / np.average(wind)

            aug = int(round((patch_size - window_size) / 2.0))
            wind = np.pad(wind, (aug, aug), mode='constant')
            wind = wind[:patch_size]

            return wind

        def window_2D(window_size, effective_window_size, power=2):
            """
            Make a 1D window function, then infer and return a 2D window function.
            Done with an augmentation, and self multiplication with its transpose.
            Could be generalized to more dimensions.
            """
            # Memoization
            wind = spline_window(window_size, effective_window_size, power)
            wind = np.expand_dims(np.expand_dims(wind, 1), 2)
            wind = wind * wind.transpose(1, 0, 2)
            return wind

        def unpad_img_pathprofiler(padded_img, patch_size, subdivisions):
            aug = int(round(patch_size * (1 - 1.0 / subdivisions)))
            ret = padded_img[aug:-aug, aug:-aug, :]
            return ret

        def merge_patches(patches, patch_size, subdivisions, padded_img_size):
            n_dims = patches[0].shape[-1]
            img = np.zeros([padded_img_size[0], padded_img_size[1], n_dims], dtype=np.float32)

            step = int(patch_size / subdivisions)
            row_range = range(0, img.shape[0] - patch_size + 1, step)
            col_range = range(0, img.shape[1] - patch_size + 1, step)

            for index1, row in enumerate(row_range):
                for index2, col in enumerate(col_range):                
                    tmp = patches[(index1 * len(col_range)) + index2]
                    tmp *= window_2D(window_size = patch_size, effective_window_size = patch_size, power = 2)    
 
                    img[row:row + patch_size, col:col + patch_size, :] = \
                        img[row:row + patch_size, col:col + patch_size, :] + tmp            
            
            img = img / (subdivisions ** 2)
            return img
        
        path_slide = os.path.join(slide_path, slide_name)    
        slide = openslide.OpenSlide(path_slide)      
            
        slide_mpp =  np.float64(slide.properties.get("openslide.mpp-x", "0"))
        slide_w = slide.level_dimensions[0][0] # slide width at the hieghst magnification
        slide_h = slide.level_dimensions[0][1] # slide height at the highest magnification    
        
        mpp2mag = {.25: 40, .5: 20, 1: 10}
        magnification = 1.25    
        wsi_highest_magnification = mpp2mag[.25 * round(float(slide_mpp) / .25)]    
        downsample = wsi_highest_magnification / magnification
        slide_level_dimensions = (int(np.round(slide.level_dimensions[0][0]/downsample)),
                                  int(np.round(slide.level_dimensions[0][1]/downsample)))    
        
        thumbnail = get_downsampled_slide(slide, slide_level_dimensions)   # uses pathprofiler method to get the downsampled thumbnail image
        
        patch_size = 512
        subdivisions = 2
        thumbnail_padded = pad_img_pathprofiler(thumbnail, patch_size, subdivisions)
    
        patches = extract_patches(thumbnail_padded, patch_size, subdivisions)    
        
        preds = predict_nn_model_on_batch('pathprofiler', patches)   
        preds = np.array(preds)
        preds = preds.transpose(0, 2, 3, 1) 
        
        tissue_mask = merge_patches(preds, patch_size, subdivisions, thumbnail_padded.shape)
        
        tissue_mask = unpad_img_pathprofiler(tissue_mask, patch_size, subdivisions)
        
        tissue_mask = np.argmax(tissue_mask, axis=2) * 255.0
        tissue_mask = tissue_mask.astype(np.uint8)
        tissue_mask = np.clip(tissue_mask, 0, 255).astype(np.uint8)
        
        tissue_mask = remove_small_objects(tissue_mask == 255, 50**2)
        
        tissue_mask[tissue_mask != 0] = 1    
        tissue_mask = tissue_mask.astype('uint8')
        
    elif model == 'pathprofiler' and method == 'fc':
        path_slide = os.path.join(slide_path, slide_name)
        slide = openslide.OpenSlide(path_slide)  
        
        slide_mpp =  np.float64(slide.properties.get("openslide.mpp-x", "0"))
        slide_w = slide.level_dimensions[0][0] # slide width at the hieghst magnification
        slide_h = slide.level_dimensions[0][1] # slide height at the highest magnification    
        
        mpp = 8 # model mpp
        thumbnail_reduction_factor = mpp / slide_mpp
        
        thumbnail = slide.get_thumbnail((slide_w // thumbnail_reduction_factor, 
                                                 slide_h // thumbnail_reduction_factor))        
        thumbnail = np.array(thumbnail.convert("RGB"))         
        
        thumbnail_padded = pad_image(thumbnail, target_size = 512)
        
        pred_fc = predict_nn_model_on_batch('pathprofiler', [thumbnail_padded])[0]    
        
        pred_fc = pred_fc[:, :thumbnail.shape[0], :thumbnail.shape[1]]   
        
        pred_fc = np.argmax(pred_fc, axis = 0)
        
        tissue_mask = (pred_fc).astype('uint8')
        
    elif model == 'pathprofiler' and method == 'sw':
        path_slide = os.path.join(slide_path, slide_name)
        slide = openslide.OpenSlide(path_slide)  
        
        slide_mpp =  np.float64(slide.properties.get("openslide.mpp-x", "0"))
        slide_w = slide.level_dimensions[0][0] # slide width at the hieghst magnification
        slide_h = slide.level_dimensions[0][1] # slide height at the highest magnification    
        
        mpp = 8 # model mpp
        thumbnail_reduction_factor = mpp / slide_mpp
        
        thumbnail = slide.get_thumbnail((slide_w // thumbnail_reduction_factor, 
                                                 slide_h // thumbnail_reduction_factor))        
        thumbnail = np.array(thumbnail.convert("RGB"))        
        
        thumbnail_padded = pad_image(thumbnail, target_size = 512)
        
        patch_size = 512
        stride = 256
        patches, x_vals, y_vals = extract_patches_from_image_sliding_window(thumbnail_padded, 
                                                                            patch_size, 
                                                                            stride, 
                                                                            return_coordinates = True)    
         
        preds = predict_nn_model_on_batch('pathprofiler', patches, batch_size = 32)   
        
        pred_sw = stitch_predictions(thumbnail_padded, preds, x_vals, y_vals)
        
        pred_sw = pred_sw[:, :thumbnail.shape[0], :thumbnail.shape[1]]    
        
        pred_sw = np.argmax(pred_sw, axis = 0)   
        
        tissue_mask = (pred_sw).astype('uint8')
        
    elif model == 'pathprofiler' and method == 'cc':        
        path_slide = os.path.join(slide_path, slide_name)
        slide = openslide.OpenSlide(path_slide)  
        
        slide_mpp =  np.float64(slide.properties.get("openslide.mpp-x", "0"))
        slide_w = slide.level_dimensions[0][0] # slide width at the hieghst magnification
        slide_h = slide.level_dimensions[0][1] # slide height at the highest magnification    
        
        mpp = 8 # model mpp
        thumbnail_reduction_factor = mpp / slide_mpp
        
        thumbnail = slide.get_thumbnail((slide_w // thumbnail_reduction_factor, 
                                                 slide_h // thumbnail_reduction_factor))        
        thumbnail = np.array(thumbnail.convert("RGB"))        
        
        thumbnail_padded = pad_image(thumbnail, target_size = 512)
        
        thumbnail_padded_cc = pad_image_center_crop(thumbnail_padded, patch_size = 512)
        
        patch_size = 512
        stride = 256
        patches, x_vals, y_vals = extract_patches_from_image_sliding_window(thumbnail_padded_cc, 
                                                                            patch_size, 
                                                                            stride, 
                                                                            return_coordinates = True)    
         
        preds = predict_nn_model_on_batch('pathprofiler', patches, batch_size = 32)   
        
        pred_cc = stitch_predictions_center_crop(thumbnail_padded_cc, preds, x_vals, y_vals)
        
        pred_cc = pred_cc[:, patch_size // 2 : -patch_size // 2, patch_size // 2 : - patch_size // 2]
        
        pred_cc = pred_cc[:, :thumbnail.shape[0], :thumbnail.shape[1]]    
        
        pred_cc = np.argmax(pred_cc, axis = 0) 
        
        tissue_mask = (pred_cc).astype('uint8')   
        
    elif model == 'grandqc' and method == 'default':        
        # MODEL TISSUE DETECTION:
        MPP_MODEL_TD = 10  # 1X model mpp
        p_s = 512 # model input patch size
        
        path_slide = os.path.join(slide_path, slide_name)
        slide = openslide.OpenSlide(path_slide)        
        
        w_l0, h_l0 = slide.level_dimensions[0]
        mpp = round(float(slide.properties["openslide.mpp-x"]), 4)
        reduction_factor = MPP_MODEL_TD / mpp

        image_or = slide.get_thumbnail((w_l0 // reduction_factor, h_l0 // reduction_factor))          
   
        image = jpeg_compression_transform(image_or, quality=80)
        
        width, height = image.size

        wi_n = width // p_s
        he_n = height // p_s

        overhang_wi = width - wi_n * p_s
        overhang_he = height - he_n * p_s

        for h in range(he_n + 1):
            for w in range(wi_n + 1):
                if w != wi_n and h != he_n:
                    x_l = w * p_s # left x
                    y_u = h * p_s # upper y
                    x_r = (w + 1) * p_s # right x
                    y_l = (h + 1) * p_s # lower y                
                    
                elif w == wi_n and h != he_n:
                    x_l = width - p_s
                    y_u  = h * p_s
                    x_r = width
                    y_l = (h + 1) * p_s
                    
                elif w != wi_n and h == he_n:
                    x_l = w * p_s
                    y_u = height - p_s
                    x_r = (w + 1) * p_s
                    y_l = height
                    
                else:
                    x_l = width - p_s
                    y_u = height - p_s
                    x_r = width
                    y_l = height
                    
                image_work = image.crop((x_l, y_u, x_r, y_l))
                
                predictions = predict_nn_model_on_batch('grandqc', [image_work])[0]

                mask = np.argmax(predictions, axis=0).astype('uint8')

                if w == 0:
                    temp_image = mask
                    
                elif w == wi_n:
                    mask = mask[:, p_s - overhang_wi:p_s]
                    temp_image = np.concatenate((temp_image, mask), axis=1)

                else:
                    temp_image = np.concatenate((temp_image, mask), axis=1)      
                
            if h == 0:
                end_image = temp_image
       
            elif h == he_n:
                temp_image = temp_image [p_s - overhang_he:p_s,]
                end_image = np.concatenate((end_image, temp_image), axis=0)
                
            else:
                end_image = np.concatenate((end_image, temp_image), axis=0)
                
        end_image = 1 - end_image
        tissue_mask = end_image.astype('uint8')   
        
    elif model == 'grandqc' and method == 'fc':
        path_slide = os.path.join(slide_path, slide_name)
        slide = openslide.OpenSlide(path_slide)  
       
        slide_mpp =  np.float64(slide.properties.get("openslide.mpp-x", "0"))
        slide_w = slide.level_dimensions[0][0] # slide width at the hieghst magnification
        slide_h = slide.level_dimensions[0][1] # slide height at the highest magnification    
        
        mpp = 10 # model mpp
        thumbnail_reduction_factor = mpp / slide_mpp
        
        thumbnail = slide.get_thumbnail((slide_w // thumbnail_reduction_factor, 
                                                 slide_h // thumbnail_reduction_factor))        
        thumbnail = np.array(thumbnail.convert("RGB"))            
       
        thumbnail_padded = pad_image(thumbnail, target_size = 512) 
        
        thumbnail_padded = jpeg_compression_transform(thumbnail_padded)  
        
        pred_fc = predict_nn_model_on_batch('grandqc', [thumbnail_padded])[0]    
        
        pred_fc = pred_fc[:, :thumbnail.shape[0], :thumbnail.shape[1]]   
        
        pred_fc = np.argmax(pred_fc, axis = 0)
        
        tissue_mask = (1 - pred_fc).astype('uint8')
        
    elif model == 'grandqc' and method == 'sw':
        path_slide = os.path.join(slide_path, slide_name)
        slide = openslide.OpenSlide(path_slide)  
       
        slide_mpp =  np.float64(slide.properties.get("openslide.mpp-x", "0"))
        slide_w = slide.level_dimensions[0][0] # slide width at the hieghst magnification
        slide_h = slide.level_dimensions[0][1] # slide height at the highest magnification    
        
        mpp = 10 # model mpp
        thumbnail_reduction_factor = mpp / slide_mpp
        
        thumbnail = slide.get_thumbnail((slide_w // thumbnail_reduction_factor, 
                                                 slide_h // thumbnail_reduction_factor))        
        thumbnail = np.array(thumbnail.convert("RGB"))            
       
        thumbnail_padded = pad_image(thumbnail, target_size = 512) 
        
        thumbnail_padded = jpeg_compression_transform(thumbnail_padded)  
        
        patch_size = 512
        stride = 256
        patches, x_vals, y_vals = extract_patches_from_image_sliding_window(np.array(thumbnail_padded), 
                                                                            patch_size, 
                                                                            stride, 
                                                                            return_coordinates = True)    
         
        preds = predict_nn_model_on_batch('grandqc', patches, batch_size = 32)   
        
        pred_sw = stitch_predictions(np.array(thumbnail_padded), preds, x_vals, y_vals)
        
        pred_sw = pred_sw[:, :thumbnail.shape[0], :thumbnail.shape[1]]    
        
        pred_sw = np.argmax(pred_sw, axis = 0)   
        
        tissue_mask = (1 - pred_sw).astype('uint8')          
        
    elif model == 'grandqc' and method == 'cc':
        path_slide = os.path.join(slide_path, slide_name)
        slide = openslide.OpenSlide(path_slide)  
       
        slide_mpp =  np.float64(slide.properties.get("openslide.mpp-x", "0"))
        slide_w = slide.level_dimensions[0][0] # slide width at the hieghst magnification
        slide_h = slide.level_dimensions[0][1] # slide height at the highest magnification    
        
        mpp = 10 # model mpp
        thumbnail_reduction_factor = mpp / slide_mpp
        
        thumbnail = slide.get_thumbnail((slide_w // thumbnail_reduction_factor, 
                                                 slide_h // thumbnail_reduction_factor))        
        thumbnail = np.array(thumbnail.convert("RGB"))            
       
        thumbnail_padded = pad_image(thumbnail, target_size = 512) 
        
        thumbnail_padded = jpeg_compression_transform(thumbnail_padded)  
        
        thumbnail_padded_cc = pad_image_center_crop(np.array(thumbnail_padded), patch_size = 512)
        
        patch_size = 512
        stride = 256
        patches, x_vals, y_vals = extract_patches_from_image_sliding_window(thumbnail_padded_cc, 
                                                                            patch_size, 
                                                                            stride, 
                                                                            return_coordinates = True)    
         
        preds = predict_nn_model_on_batch('grandqc', patches, batch_size = 32)   
        
        pred_cc = stitch_predictions_center_crop(thumbnail_padded_cc, preds, x_vals, y_vals)
        
        pred_cc = pred_cc[:, patch_size // 2 : -patch_size // 2, patch_size // 2 : - patch_size // 2]
        
        pred_cc = pred_cc[:, :thumbnail.shape[0], :thumbnail.shape[1]]    
        
        pred_cc = np.argmax(pred_cc, axis = 0) 
        
        tissue_mask = (1 - pred_cc).astype('uint8')    
        
    elif model == 'otsu' and method == 'otsu':
        path_slide = os.path.join(slide_path, slide_name)
        slide = openslide.OpenSlide(path_slide)  
       
        slide_mpp =  np.float64(slide.properties.get("openslide.mpp-x", "0"))
        slide_w = slide.level_dimensions[0][0] # slide width at the hieghst magnification
        slide_h = slide.level_dimensions[0][1] # slide height at the highest magnification    
        
        mpp = 10 # otsu mpp
        thumbnail_reduction_factor = mpp / slide_mpp
        
        thumbnail = slide.get_thumbnail((slide_w // thumbnail_reduction_factor, 
                                                 slide_h // thumbnail_reduction_factor))        
        thumbnail = np.array(thumbnail.convert("RGB"))
        
        gray = color.rgb2gray(thumbnail)  
        gray_blurred = gaussian(gray, sigma=1)
        
        # Otsu thresholding    
        otsu_thresh = filters.threshold_otsu(gray_blurred)
        binary_mask = gray_blurred < otsu_thresh  # Tissue usually darker than background
        
        # Clean up using morphological operations, remove small holes and specks    
        cleaned_mask = morphology.remove_small_holes(binary_mask, area_threshold=500)
        cleaned_mask = morphology.remove_small_objects(cleaned_mask, min_size=100)
        tissue_mask = cleaned_mask.astype('uint8')            
    
    else:
        print('Model should be "pathprofiler" or "grandqc", Method should be "default", or "fc", or "sw", or "cc"')
        tissue_mask = []
        
    return tissue_mask