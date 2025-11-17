import numpy as np

def calculate_cm(mask,pred,number_of_classes):
    # this function takes in a predicted segmentation mask and an annotated ground truth mask for a k-class segmentation task
    # to calculate confusion matrix, the rows are the ground-turth labels and the columns are the predictions       
    if mask.shape == pred.shape:
        mask = mask.flatten()
        pred = pred.flatten()
        cm = np.zeros((number_of_classes,number_of_classes))
        for i in range(number_of_classes):
            for j in range(number_of_classes):
                cm[i,j] = np.sum(np.logical_and(mask==i, pred==j))
        return cm
    else:
        print('The prediction and the ground truth segmentation masks are of different size')
        return []