import numpy as np

def calculate_dice_binary_matrix(cm):  
    if np.sum(cm, axis =1)[0] ==0:
        dice = np.nan  
    else:
        dice = 2*cm[0, 0]/(2*cm[0, 0] + cm[0, 1] + cm[1, 0])  
    return dice

def convert_m_by_m_matrix_to_2_by_2(mat, index = 3):
    mat_copy = np.copy(mat)
    mat_copy[[index,0],:] = mat_copy[[0,index],:]  
    mat_copy[:,[index,0]] = mat_copy[:,[0,index]]  
    two_by_two_matrix = np.zeros((2,2))
    two_by_two_matrix[0,0] = mat_copy[0,0]
    two_by_two_matrix[0,1] = np.sum(mat_copy[0,1:])
    two_by_two_matrix[1,0] = np.sum(mat_copy[1:,0])
    two_by_two_matrix[1,1] = np.sum(mat_copy[1:,1:])    
    return two_by_two_matrix

def calculate_dice(cm, class_index):
    # CM is a numpy array of size (k,k) where k is the number of class labels
    # If the ground truth label does not contain a class label, a DICE of NA is obtained for that class.
    # dice_val is returned for class k
    cm_conv = convert_m_by_m_matrix_to_2_by_2(cm, index = class_index)
    dice_val = calculate_dice_binary_matrix(cm_conv)
    return dice_val