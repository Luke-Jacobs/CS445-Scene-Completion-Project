import cv2
import numpy as np
from typing import Tuple, Any, List
import pickle

from local_context_matching import *

# This will import the descriptors and similar images to the target
from semantic_scene_matching import *

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Semantic Hole Filler',
        description='Fills holes in images with a semantically-relevant cutout of another image')
    parser.add_argument('input', type=str)  # positional argument
    parser.add_argument('mask', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    hole_mask_full = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE) == 255

    input_image = cv2.imread(args.input) / 255.0
    plt.figure()
    plt.imshow(input_image)

    feature = compute_gist_descriptor(input_image, kernels)
    errors_indices = match(feature, descriptors)  # NOTE: descriptors needs to be populated by semantic_scene_matching
    similar_imgs = []
    for i in range(len(errors_indices)):
        index = errors_indices[i][1]
        match_image = cv2.imread(file_paths[index]) / 255.0
        similar_imgs.append(file_paths[index])

    print('Finding the best candidate for input image among %d candidate images' % (len(similar_imgs),))

    input_image = (input_image * 255).astype(np.uint8)
    filled, blended_mask, candidate = findBestHoleFill(input_image, hole_mask_full, similar_imgs)

    cv2.imwrite(args.output, filled)
    cv2.imwrite('blended_mask.jpg', blended_mask.astype(np.uint8)*255)
    cv2.imwrite('best_candidate.jpg', candidate)
