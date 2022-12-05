import cv2
import numpy as np
from typing import Tuple, Any, List
import pickle

from local_context_matching import *

# This will import the descriptors and similar images to the target
from semantic_scene_matching import *

if __name__ == '__main__':
    hole_mask_full = cv2.imread('0008_3_mask.jpg', cv2.IMREAD_GRAYSCALE) == 255

    input_image = cv2.imread(dataset_dir + "0008 (3).jpg") / 255.0
    plt.figure()
    plt.imshow(input_image)

    feature = compute_gist_descriptor(input_image, kernels)
    errors_indices = match(feature, descriptors)
    similar_imgs = []
    for i in range(len(errors_indices)):
        index = errors_indices[i][1]
        match_image = cv2.imread(file_paths[index]) / 255.0
        similar_imgs.append(file_paths[index])

    print('Finding the best candidate for input image among %d candidate images' % (len(similar_imgs),))

    input_image = (input_image * 255).astype(np.uint8)
    findBestHoleFill(input_image, hole_mask_full, similar_imgs)
