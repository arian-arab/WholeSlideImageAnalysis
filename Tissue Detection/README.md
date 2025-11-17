Download the model weights from https://zenodo.org/records/14507273/files/Tissue_Detection_MPP10.pth?download=1 and put it in the "/codes/" folder, rename it to : "grandqc_10mpp.pth".

Also download the model weights from https://drive.google.com/file/d/1otWor5WnaJ4W9ynTOF1XS755CsxEa4qj/view and put it in the "/codes/" folder, rename it to : "path_profiler.pth".

The original models are from:
https://github.com/cpath-ukk/grandqc and https://github.com/MaryamHaghighat/PathProfiler

To obtain tissue segmentation masks run "run_tissue_detection.py", the outpus are put in the "outputs" folder and the input image is read from "example".

There are two models that can be choosen from : "pathprofiler" and the "grandqc".

The method can take four possible options "default", "fc", "sw", and "cc". "default" is the original implementation (as it was used by the model developers), "fc" standa for fully-connected and passes the thubnmail image directly to the network if  memory allows, "sw" stands for the sliding window approach which uses a patch size of 512 and strides of 256, "cc" stands for the center crop approach which also uses strides of 256.

If you set both model and mehtod to "otsu" it performs the otsu algorithm for tissue detection.

The main function is located at: "codes/tissue_detection.py". It takes the following arguments as input: tissue_detection(slide_path, slide_name, model = 'pathprofiler', method = 'default') and outputs a binary tissue_segmentation mask.
