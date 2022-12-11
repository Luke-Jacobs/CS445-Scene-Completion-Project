#!/usr/bin/env python
# coding: utf-8

# Given an input image, a GIST descriptor is computed by
# 1.  Convolve the image with 32 Gabor filters at 4 scales, 8 orientations, producing 32 feature maps of the same size of the input image.
# 2.  Divide each feature map into 16 regions (by a 4x4 grid), and then average the feature values within each region.
# 3.  Concatenate the 16 averaged values of all 32 feature maps, resulting in a 16x32=512 GIST descriptor.
# Intuitively, GIST summarizes the gradient information (scales and orientations) for different parts of an image, which provides a rough description (the gist) of the scene.
# 
# Reference:
# 1.  Modeling the shape of the scene: a holistic representation of the spatial envelope

# In[13]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

from tqdm import tqdm
import pickle

dataset = './dataset/'


# In[14]:


def make_square_img(img):
    
    H, W, _ = img.shape
    
    side_length = (int(min(H,W)/4)) * 4
    half_side_length = side_length // 2
    
    top = H//2 - half_side_length
    bottom = side_length + H//2 - half_side_length
    
    left = W//2 - half_side_length
    right = side_length + W//2 - half_side_length
    
    new_img = img[top:bottom, left:right, :]
    
    return np.array(new_img)


# In[15]:


def match(feats, ref_feats):
    
    errors = []
    
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if (error != 0):
            errors.append((error, i))
            
    errors.sort(key = lambda x: x[0])
    
    return errors[:20]


# In[16]:


def compute_gist_descriptor(img, kernels):
    
    img_squared = make_square_img(img)
    img_squared = img_squared.astype('float32')
    img_grayscale = cv2.cvtColor(img_squared, cv2.COLOR_RGB2GRAY)
         
    shrink = (slice(0, None, 3), slice(0, None, 3))
    
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(img_grayscale, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    
    return feats


# In[17]:


def generate_descriptors(dataset_dir):
    descriptors = []
    file_paths = []

    total_files = 0
    for path, subdirs, files in os.walk(dataset_dir):
        for name in files:
            total_files += 1

    count = 0
    for path, subdirs, files in os.walk(dataset_dir):
        for name in files:

            count += 1
            if (count % 10 == 1):
                print("Progress: {}/{}".format(count, total_files))

            file_path = os.path.join(path, name)
            file_paths.append(file_path)

            img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB) / 255.0

            descriptors.append(compute_gist_descriptor(img, kernels))
            
    descriptors = np.array(descriptors)

    output = open(dataset_dir + 'descriptors.pkl', 'wb')
    pickle.dump(descriptors, output)
    output.close()

    output = open(dataset_dir + 'file_paths.pkl', 'wb')
    pickle.dump(file_paths, output)
    output.close()
    
    return descriptors, file_paths


# In[18]:


# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)


# In[19]:


dataset_dir = dataset + "5k_of_each/"


# In[20]:


descriptors_pkl_path = dataset_dir + 'descriptors.pkl'
file_paths_pkl_path = dataset_dir + 'file_paths.pkl'

if os.path.exists(descriptors_pkl_path):
    
    descriptors_pkl_file = open(descriptors_pkl_path, 'rb')
    descriptors = pickle.load(descriptors_pkl_file)
    
    file_paths_pkl_file = open(file_paths_pkl_path, 'rb')
    file_paths = pickle.load(file_paths_pkl_file)
    
else:
    descriptors, file_paths = generate_descriptors(dataset_dir)

