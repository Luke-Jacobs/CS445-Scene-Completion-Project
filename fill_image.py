import cv2
import numpy as np
from typing import Tuple, Any, List
import pickle

from local_context_matching import *

# This will import the descriptors and similar images to the target
from semantic_scene_matching import *

if __name__ == '__main__':
    print('Finding the best candidate for input image among %d candidate images' % (len(similar_imgs),))

    hole_mask_full = cv2.imread('0009_mask.jpg', cv2.IMREAD_GRAYSCALE) == 255

    input_image = (input_image * 255).astype(np.uint8)
    findBestHoleFill(input_image, hole_mask_full, similar_imgs)
