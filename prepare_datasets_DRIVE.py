#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image
import cv2



def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
# original_imgs_train = "./DRIVE/training/images/"
# groundTruth_imgs_train = "./DRIVE/training/1st_manual/"
# borderMasks_imgs_train = "./DRIVE/training/mask/"
#test
original_imgs_test =  "./selected_paths.txt" #"20_sample_images/originals/" #"../sample_retinal_images_colorfundus/"

# load filepaths
<<<<<<< HEAD
with open(filepaths) as f:
    original_imgs_test = f.readlines()

=======
f = open(original_imgs_test, "r")
>>>>>>> parent of 092e5b1... Fixed bug in prepare_datasets_DRIVE.py
files = f.readlines()

groundTruth_imgs_test = "20_sample_images/dummy_mask/" #"../dummy_masks_for_colorfundus/"
borderMasks_imgs_test = "20_sample_images/border_masks/" #"../border_masks_for_colorfundus/"
#---------------------------------------------------------------------------------------------

Nimgs = 20
channels = 3
height = 584
width = 565
dataset_path = "20_sample_images_datasets/"

def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,train_test="null"):

    # Initialize empty arrays
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    border_masks = np.empty((Nimgs,height,width))

    #for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path

    for i in range(len(imgs_dir)):

        #original
        print "original image: " +imgs_dir[i]
        img = Image.open(imgs_dir[i])
        img = np.asarray(img)

        # Resize image
        if img.shape != (height, width, channels):
            img = cv2.resize(img, (width, height))

        # Convert to RGB if its a grayscale
        if img.shape[-1] != 3:
            img = np.expand_dims(img, -1)
            img = np.repeat(img, 3, -1)

        # Append to array
        imgs[i] = img

        #corresponding ground truth
        groundTruth_name = "dummy_mask.jpg"
        print "ground truth name: " + groundTruth_name
        g_truth = Image.open(groundTruth_dir + groundTruth_name)
        g_truth = np.asarray(g_truth, dtype='float64') #[:, :, 0]

        if g_truth.shape != (width, height):
            g_truth = cv2.resize(g_truth, (width, height))

        groundTruth[i] = g_truth

        #corresponding border masks
        border_masks_name = ""
        if train_test=="train":
            border_masks_name = files[i][0:2] + "_training_mask.gif"
        elif train_test=="test":
            border_masks_name = "border_mask.jpg" #[0:2]
        else:
            print "specify if train or test!!"
            exit()
        print "border masks name: " + border_masks_name
        b_mask = Image.open(borderMasks_dir + border_masks_name)
        b_mask = np.asarray(b_mask, dtype='float64') #[:, :, 0]
        if b_mask.shape != (height, width):
            b_mask = cv2.resize(b_mask, (width, height))
        border_masks[i] = b_mask

    print "imgs max: " +str(np.max(imgs))
    print "imgs min: " +str(np.min(imgs))

    # print(np.max(groundTruth), np.max(border_masks))
    # print(np.min(groundTruth), np.min(border_masks))
    assert(np.max(groundTruth)==0.0 and np.max(border_masks)==255.0)
    assert(np.min(groundTruth)==0.0 and np.min(border_masks)==0.0)
    print "ground truth and border masks are correctly withih pixel value range 0-255 (black-white)"
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert(groundTruth.shape == (Nimgs,1,height,width))
    assert(border_masks.shape == (Nimgs,1,height,width))

    return imgs, groundTruth, border_masks

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
#getting the training datasets
#imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train,"train")
#print "saving train datasets"
#write_hdf5(imgs_train, dataset_path + "DRIVE_dataset_imgs_train.hdf5")
#write_hdf5(groundTruth_train, dataset_path + "DRIVE_dataset_groundTruth_train.hdf5")
#write_hdf5(border_masks_train,dataset_path + "DRIVE_dataset_borderMasks_train.hdf5")

#getting the testing datasets
imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test,groundTruth_imgs_test,borderMasks_imgs_test,"test")
print "saving test datasets"
write_hdf5(imgs_test,dataset_path + "original_imgs.hdf5")
write_hdf5(groundTruth_test, dataset_path + "gt_masks.hdf5")
write_hdf5(border_masks_test,dataset_path + "border_masks.hdf5")
