
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import math
import cv2
import heatmap
from datetime import datetime


from keras.models import  model_from_yaml
from keras.optimizers import Adam, SGD

from glob import glob
from libs.common import my_resize
from libs.pavement_common import to_patches

from argparse import ArgumentParser as argparse_ArgumentParser



ap = argparse_ArgumentParser()
ap.add_argument("-inputdir", required = False,
	help = "Path of input images")
ap.add_argument("-outputdir", required = False,
	help = "Path of output images")
args = vars(ap.parse_args())



image_files_path = args["inputdir"]
nrootdir = args["outputdir"]
if image_files_path is None:
    image_files_path = './input_images/'
    
if nrootdir is None:
    nrootdir = "./images_dst/"



################################################################
# change these parameters:
destination_directory = '../../Dataset/GAPs/GAPs-dataset/images/'

image_files_directory = image_files_path + '/**/*.jpg'





saved_model = './models/pavement_model.yaml'
pre_trained_weight = './models/pavement_weights_best.hdf5'

normal_txt_path = nrootdir + 'normal_pic' + '.txt'
disdress_txt_path = nrootdir + 'disdress_pic' + '.txt'


PATCH_SIZE = 64    
OVER_LENTH = 32    
W_DES = 1920
H_DES = 1080

PREDIC_THRESH = 0.5
MIN_PATCH_NUM_DIS = 4.8
#################################################################


image_files = glob(image_files_directory,recursive=True)
jpg_total_num = len(image_files)
    

with open(saved_model) as yamlfile:
    loaded_model_yaml = yamlfile.read()
model = model_from_yaml(loaded_model_yaml)

model.load_weights(pre_trained_weight)






normal_pic = open(normal_txt_path,'w')
disdress_pic = open(disdress_txt_path,'w')

time1_s = datetime.now()
pic_index = 0
for image_filename in sorted(image_files):
    Dis_Valid = False   
    percent   = 0
    pic_index = pic_index +1
    print("Processing {}/{} image,file name = {}....".format(pic_index,jpg_total_num,image_filename))
    [dirname, filename] = os.path.split(image_filename)
    [fname, fename] = os.path.splitext(filename)

    image_origin = cv2.imread(image_filename)

    

    image_resized = my_resize(image = image_origin,w_des = W_DES,h_des = H_DES)
    

       
    image_copy = image_resized.copy()

    patch_batch,patch_pos_row_arr,patch_pos_col_arr = to_patches(image = image_resized,patch_size = PATCH_SIZE,over_lenth = OVER_LENTH)
    


    result = model.predict_on_batch( patch_batch) 


    
    result_arr = np.array(result)
    if result_arr.shape[1] == 2:        
        mask_y = np.where(result_arr[:,1] > PREDIC_THRESH)
    elif result_arr.shape[1] == 1:
        mask_y = np.where(result_arr[:,0] > PREDIC_THRESH)


    mask_y_arr = np.array(mask_y)
    
    pts_new = (patch_pos_col_arr[mask_y_arr] + int(PATCH_SIZE/2),image_resized.shape[0]-patch_pos_row_arr[mask_y_arr] - int(PATCH_SIZE/2))
    pts_new_arr = np.array(pts_new)    

    pts_new_arr_resh  = np.reshape(pts_new_arr, -1, order='F')
    pts_new_arr_resh2 = np.reshape(pts_new_arr_resh, (-1,2))
    pts_new_arr_t = tuple(pts_new_arr_resh2)
    pts = pts_new_arr_t

    hm = heatmap.Heatmap()    
    if len(pts) > 0:
    
        hm_img_1ch = hm.heatmap(pts, dotsize=120, opacity=50, scheme ='classic', size=(image_resized.shape[1],image_resized.shape[0]),
                                area=((0, 0), (image_resized.shape[1],image_resized.shape[0])))

        cv_img = cv2.cvtColor(np.asarray(hm_img_1ch), cv2.COLOR_RGBA2BGR)

        cv_img_copy = cv_img.copy()

        mask = np.zeros(image_resized.shape, np.uint8)  
        mask[:, :, ] = 0

        cv_img_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(cv_img_gray, thresh=80, maxval=255, type=cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        total_dis_area = 0
        
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area >= PATCH_SIZE * PATCH_SIZE * MIN_PATCH_NUM_DIS:
                cv2.drawContours(image=mask, contours=contours, contourIdx=i, color=(255, 255, 255), thickness=-1)
                total_dis_area = total_dis_area + area
                Dis_Valid = True

        cv_img_dst = cv2.bitwise_and(mask, cv_img_copy)


        percent =  total_dis_area / (mask.shape[0] * mask.shape[1]) *100
        percent = round(percent, 1)


        alpha = 0.1
        dst_image = cv2.addWeighted(cv_img_dst, alpha, image_copy,1, 0) 


        font = cv2.FONT_HERSHEY_SIMPLEX  
        image_add = cv2.putText(dst_image, 'DistresssArea:' + str(percent) +"%", (0, 80), font, 1.8, (0, 255, 0), 4)

    if Dis_Valid == True:
        cv2.imwrite(nrootdir + fname + "_note"  +".jpg", image_add)  # dst_image to image_add
        disdress_pic.write(dirname +"\\"+ fname + ".jpg" + "--"+ str(percent) +"%"+ "\n")
    else:
        normal_pic.write(dirname + "\\" + fname + ".jpg" + "--"+ str(percent) +"%"+ "\n")



time1_e = datetime.now()
 
time1_secs = (time1_e - time1_s).seconds    

print("The total  time is :" + str(time1_secs) + " seconds.")

disdress_pic.close()
normal_pic.close()
