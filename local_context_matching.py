import cv2
import numpy as np
from typing import Tuple, Any, List

# Search for the surrounding region to the cutout in all the semantically-similar images (up to 200 best-matching scenes)
# The "surrounding region" or local context is considered to be "all pixels within an 80 pixel radius of the hole's boundary"
# The full surrounding region is scanned across each candidate image at 3 scales: 0.81, 0.90, and 1.0.
# - I believe that when we scale the surrounding region, we need to undo this scaling when we blend the selected region
#   back into the target image
# The error of each local context match is the SSD error weighted by translational offset. This discourages distant matches.

# In addition
# to the pixel-wise alignment score, we also compute a texture similarity score to measure coarse compatibility of the proposed fill-in
# region to the source image within the local context. A simple texture descriptor is computed as a 5x5 median filter of image gradient
# magnitude at each pixel, and the descriptors of the two images are compared via SSD.


def getMaskEdge(bin_mask: np.ndarray) -> Tuple[tuple, ...]:
    np_pts = np.where(bin_mask)
    pts = list(zip(np_pts[0], np_pts[1]))
    edge_list = set()
    for pt in pts:
        # If the point has at least one neighbor who is False
        if len(getFalseNeighbors(bin_mask, pt)) != 0:
            edge_list.add(pt)
    return tuple(edge_list)


def getFalseNeighbors(bin_mask: np.ndarray, pt: tuple) -> Tuple[tuple, ...]:
    if not ((0 < pt[0] < bin_mask.shape[0]) and (0 < pt[1] < bin_mask.shape[1])):
        return tuple()
    row, col = pt
    neighbors = [(row+1, col), (row-1, col), (row, col+1), (row, col-1),
                 (row+1, col+1), (row+1, col-1), (row-1, col+1), (row-1, col-1)]
    neighbors = filter(lambda pt_: 0 <= pt_[0] < bin_mask.shape[0] and 0 <= pt_[1] < bin_mask.shape[1], neighbors)
    false_neighbors = filter(lambda pt_: not bin_mask[pt_], neighbors)
    return tuple(false_neighbors)


def growBinaryMask(bin_mask: np.ndarray, edge_list: Tuple[tuple, ...]):
    """
    Grows the given binary mask by one pixel in 8-connected space.
    Returns new grown mask and new edge list.
    """
    assert np.all(bin_mask[tuple(zip(*edge_list))])  # All edges must be in the mask

    grown_edges = set()
    for pt in edge_list:
        unfilled_neighbors = getFalseNeighbors(bin_mask, pt)
        grown_edges.update(unfilled_neighbors)
        if len(tuple(zip(*unfilled_neighbors))) > 0:
            bin_mask[tuple(zip(*unfilled_neighbors))] = True
    return tuple(grown_edges)


def getSurroundingRegion(target_img: np.ndarray):
    """
    Returns a boolean array of the image which contains True values at the locations of pixels in the local context.
    - Must identify transparency to find the hole
    - Must work around edges
    """
    assert target_img.dtype == np.uint8

    mask = target_img[:, :, 3] == 0  # Get transparent region
    mask_edge = getMaskEdge(mask)
    grown_mask = mask.copy()

    # Grow the mask 80 times to get a (roughly) 80-point radius
    for i in range(80):
        mask_edge = growBinaryMask(grown_mask, mask_edge)

    # Find only the grown region
    grown_mask[mask] = False
    return grown_mask


def scoreSceneImage(candidate_img: np.ndarray, local_context_img: np.ndarray, local_context_descriptor: np.ndarray):
    """
    Scores scene images by the metric of a weighted sum between the negative of the magnitude of the fill's translational
    offset, the pixel-wise SSD alignment score between the local context of the hole and the local context of the fill,
    and the SSD between the texture descriptor of the local context of the hole and the texture descriptor of the local
    context of the fill.
    """
    pass


if __name__ == '__main__':
    target = cv2.imread(r"C:\Users\luked\OneDrive\Documents\UIUC\CS 445\Final Project\CS445-Scene-Completion-Project\target_img.png", cv2.IMREAD_UNCHANGED)
    reg = getSurroundingRegion(target)
    mask = np.ones_like(target) * 255
    mask[reg] = np.array([0, 0, 0, 255])
    cv2.imwrite('test.png', mask)
