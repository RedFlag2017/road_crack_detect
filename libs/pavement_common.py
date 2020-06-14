import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
import math 
import datetime
from libs.common import *



    
def to_patches(image,patch_size,over_lenth):
    import math, cv2
    import numpy as np
    patch_batch = []
    # BGR to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print('gray.shape={}'.format(gray.shape))
    gray_ch_hw = gray.reshape(1,gray.shape[0],gray.shape[1])   #(1,64,64)
    # print('gray_ch_hw.shape={}'.format(gray_ch_hw.shape))


    gray_ch_hw = gray_ch_hw/128 - 1
    gray_ch_hw = gray_ch_hw.reshape(1,gray_ch_hw.shape[0],gray_ch_hw.shape[1],gray_ch_hw.shape[2])
   
    #whole image to patches

    
    last_patch_row = gray.shape[0] - patch_size  #1080-64
    last_patch_col = gray.shape[1] - patch_size  #1920-64

    patch_num_row = math.floor((last_patch_row)/over_lenth)+1  #行分块数
    patch_num_col = math.floor((last_patch_col)/over_lenth)+1  #列分块数

    # print('patch_num_row={}'.format(patch_num_row))
    # print('patch_num_col={}'.format(patch_num_col))

    patch_row_step = np.arange(0,last_patch_row+1,over_lenth)  #加1确保整除
    patch_col_step = np.arange(0,last_patch_col+1,over_lenth)

    # print('patch_row_step={}'.format(patch_row_step))
    # print('patch_col_step={}'.format(patch_col_step))

    patch_pos_row_arr = patch_row_step.repeat(patch_num_col)  #元素复制1 [0,1].repeat(3) ~[0,0,0,1,1,1]
    patch_pos_col_arr = np.tile(patch_col_step,patch_num_row) #元素复制2 np.tile([0,1],3) ~[0,1,0,1,0,1] #行*列  列*行

    # print('patch_pos_row_arr.size={}'.format(patch_pos_row_arr.shape))
    # print('patch_pos_col_arr.size={}'.format(patch_pos_col_arr.shape))

    patch_batch = np.zeros((patch_num_row*patch_num_col,1,patch_size,patch_size))   #(m*n,1,64,64)

    for i in range(0,patch_pos_col_arr.shape[0]):
        patch_batch[i,:,:,:] = gray_ch_hw[:,:, patch_pos_row_arr[i]:patch_pos_row_arr[i] + patch_size, patch_pos_col_arr[i]:patch_pos_col_arr[i] + patch_size]  # 截取像素点
    
    return patch_batch,patch_pos_row_arr,patch_pos_col_arr    
