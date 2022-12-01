import os
import json
import math
import yaml
import glob
import random
import numpy as np
import pandas as pd
import cv2
import pathlib
from loss import *
import tensorflow as tfb
import earthpy.plot as ep
import earthpy.spatial as es
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
from dataset import read_img, transform_data
from tensorflow.keras.models import load_model



def val_show_predictions(dataset, model):
    
    with open("/home/mdsamiul/github_project/road_segmentation/data/json/valid_patch_phr_cb_256.json", 'r') as j:  # opening the json file
        patch_test_dir = json.loads(j.read())

    df = pd.DataFrame.from_dict(patch_test_dir)  # read as panadas dataframe
    test_dir = pd.read_csv("/home/mdsamiul/github_project/road_segmentation/data/valid.csv")  # get the csv file
    total_score = 0.0

    i = 0 #random.randint(0, len(test_dir))
    # loop to traverse full dataset
    
    mask_s = transform_data(read_img(test_dir["masks"][i], label=True), 2)
    mask_size = np.shape(mask_s)
    print("image size: ", mask_size)
    # for same mask directory get the index
    idx = df[df["masks"] == test_dir["masks"][i]].index
    
    # print("image: ", i)
    # print(idx)

    # construct a single full image from prediction patch images
    pred_full_label = np.zeros((mask_size[0], mask_size[1]), dtype=int)
    for j in idx:
        p_idx = patch_test_dir["patch_idx"][j]
        #print(p_idx)
        feature, mask, indexNum = dataset.get_random_data(j)
        pred_mask = model.predict(feature)
        pred_mask = np.argmax(pred_mask, axis=3)
        pred_full_label[p_idx[0]:p_idx[1], p_idx[2]:p_idx[3]] = pred_mask[0]   

    # read original image and mask
    feature_img = read_img(test_dir["feature_ids"][i])          #, in_channels=config['in_channels']
    mask = transform_data(read_img(test_dir["masks"][i], label=True), 2)
        
    # calculate keras MeanIOU score
    # m = keras.metrics.MeanIoU(num_classes=2)
    # m.update_state(np.argmax([mask], axis=3), [pred_full_label])
    # score = m.result().numpy()
    # total_score += score

    # # plot and saving image
        
    # display({"image": feature_img,      # change in the key "image" will have to change in the display
    #         "Mask": np.argmax([mask], axis=3)[0],
    #         "Prediction (miou_{:.4f})".format(score): pred_full_label 
    #         }, indexNum, config['prediction_val_dir'], score, config['experiment'])
    
    
# model = load_model("/home/mdsamiul/github_project/road_segmentation/model/fapnet/fapnet_ex_phr_cb_ep_500_22-Nov-22.hdf5", compile=False)
# val_show_predictions(1, 1)



def save_patch_idx(path, patch_size=256, stride=64, test=None, patch_class_balance=None):
    """
    Summary:
        finding patch image indices for single image based on class percentage. work like convolutional layer
    Arguments:
        path (str): image path
        patch_size (int): size of the patch image
        stride (int): how many stride to take for each patch image
    Return:
        list holding all the patch image indices for a image
    """
    # read the image
    img = np.random.rand(375,1242)   #cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img[img < 105] = 0
    img[img > 104] = 1
    # img = np.where((img<105) | (img>105), 0, 1)

    # calculating number patch for given image
    # [{(image height-patch_size)/stride}+1]
    patch_height = int((img.shape[0]-patch_size)/stride)+1
    # [{(image weight-patch_size)/stride}+1]
    patch_weight = int((img.shape[1]-patch_size)/stride)+1

    # total patch images = patch_height * patch_weight
    patch_idx = []

    # image column traverse
    for i in range(patch_height+1):
        # get the start and end row index
        s_row = i*stride
        e_row = s_row+patch_size 
        if e_row > img.shape[0]:
            s_row = img.shape[0] - patch_size
            e_row = img.shape[0]
        
        if e_row <= img.shape[0]:

            # image row traverse
            for j in range(patch_weight+1):
                # get the start and end column index
                start = (j*stride)
                end = start+patch_size
                
                if end > img.shape[1]:
                    start = img.shape[1] - patch_size
                    end = img.shape[1]
                
                patch_idx.append([s_row, e_row, start, end])
                     
                if end==img.shape[1]:
                    break
                
        if e_row==img.shape[0]:
            break
    
    print(patch_idx)
    print(len(patch_idx))


save_patch_idx('/home/mdsamiul/github_project/road_segmentation/data//gt_image/um_road_000073.png')