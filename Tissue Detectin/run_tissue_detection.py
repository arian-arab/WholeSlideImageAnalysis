from PIL import Image
import openslide
import numpy as np
import cv2

from codes.tissue_detection import tissue_detection
from codes.calculate_cm import calculate_cm
from codes.calculate_dice import calculate_dice

slide_path = 'example/'
slide_name = 'slide.tiff'
slide_mask_name = 'slide_mask.tiff'
output_path = 'outputs/'

########## load slide and slide mask images and save them ###########
slide = openslide.OpenSlide(slide_path + slide_name)
slide_w = slide.level_dimensions[0][0] # slide width at the hieghst magnification
slide_h = slide.level_dimensions[0][1] # slide height at the highest magnification    
slide_mpp =  np.float64(slide.properties.get("openslide.mpp-x", "0"))
mpp = 10 # loading thumbnail at mpp of 10
thumbnail_reduction_factor = mpp / slide_mpp 
thumbnail = slide.get_thumbnail((slide_w // thumbnail_reduction_factor, slide_h // thumbnail_reduction_factor))    
thumbnail = np.array(thumbnail.convert("RGB"))
thumbnail = Image.fromarray(thumbnail)
thumbnail.save(output_path + 'thumbnail.png')  
thumbnail = np.array(thumbnail)

mask = openslide.OpenSlide(slide_path + slide_mask_name)
mask_gt = mask.get_thumbnail((slide_w // thumbnail_reduction_factor, slide_h // thumbnail_reduction_factor))    
mask_gt = np.array(mask_gt.convert("RGB"))
mask_gt = mask_gt[:,:,0]    
mask_gt[mask_gt != 0 ] = 255
mask_gt = Image.fromarray(mask_gt)
mask_gt.save(output_path + 'tissue_mask_gt.png')
mask_gt = np.array(mask_gt)
mask_gt[mask_gt != 0] = 1

########## Running Tissue Detection Models ###########
models = ['pathprofiler', 'grandqc']
methods = ['default', 'fc', 'sw', 'cc']

dice_vals = []
for model in models:        
    for method in methods:             
        tissue_mask = tissue_detection(slide_path, slide_name,  model, method)  
        tissue_mask = cv2.resize(tissue_mask, (thumbnail.shape[1], thumbnail.shape[0]))                   
        
        cm = calculate_cm(mask_gt, tissue_mask, number_of_classes = 2)
        dice_vals.append([model, method, calculate_dice(cm, class_index = 1)])            
        
        base = Image.fromarray(thumbnail).convert("RGBA")    #
        overlay_color = np.zeros((tissue_mask.shape[0], tissue_mask.shape[1], 4), dtype=np.uint8)
        overlay_color[tissue_mask == 1] = [0, 255, 0, int(0.3 * 255)]  # red with 30% opacity
        overlay = Image.fromarray(overlay_color, mode="RGBA")
        composite = Image.alpha_composite(base, overlay)
        composite.save(output_path + 'overlay_' + model + '_' + method + '.png')
        
        tissue_mask[tissue_mask != 0 ] = 255
        tissue_mask = Image.fromarray(tissue_mask)
        tissue_mask.save(output_path + 'tissue_mask_' + model + '_' + method + '.png') 
    
                    
tissue_mask = tissue_detection(slide_path, slide_name,  'otsu', 'otsu')  
tissue_mask = cv2.resize(tissue_mask, (thumbnail.shape[1], thumbnail.shape[0]))                   

cm = calculate_cm(mask_gt, tissue_mask, number_of_classes = 2)
dice_vals.append(['otsu', 'otsu', calculate_dice(cm, class_index = 1)])  

base = Image.fromarray(thumbnail).convert("RGBA")    #
overlay_color = np.zeros((tissue_mask.shape[0], tissue_mask.shape[1], 4), dtype=np.uint8)
overlay_color[tissue_mask == 1] = [0, 255, 0, int(0.3 * 255)]  # red with 30% opacity
overlay = Image.fromarray(overlay_color, mode="RGBA")
composite = Image.alpha_composite(base, overlay)
composite.save(output_path + 'overlay_otsu.png')

tissue_mask[tissue_mask != 0 ] = 255
tissue_mask = Image.fromarray(tissue_mask)
tissue_mask.save(output_path + 'tissue_mask_otsu.png')           
    
with open(output_path + 'dice.txt', "w") as f:
    for item in dice_vals:
        f.write(str(item) + "\n")
    
# Calculating Tissue Mask Area
# h_t, w_t = tissue_mask.shape # tissue mask width and height        
# scaling_factor = w / w_t # scaling factor         
# mpp_mask = scaling_factor * mpp # mpp at the tissue mask level         
# area_pixel = (mpp_mask * 0.001)**2 # area of a pixel at the tissue mask level        
# area_t = np.sum(tissue_mask) # area of the tissue mask in pixels        
# area = area_pixel * area_t # area of the tissue mask in mm^2        
# areas.append(area)    
    