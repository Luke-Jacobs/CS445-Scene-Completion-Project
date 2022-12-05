import cv2
import numpy as np
from typing import Tuple, Any, List
import pickle

from poisson_graph_cut import *

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

LC_SCALES = [1.0, 0.90, 0.81]

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


def getLocalContext(target_img: np.ndarray, hole_mask_full: np.ndarray) -> Tuple[np.ndarray, tuple]:
    """
    Returns a boolean array of the image which contains True values at the locations of pixels in the local context.
    - Must identify transparency to find the hole
    - Must work around image border
    """
    assert target_img.dtype == np.uint8

    LC_CONTEXT_RADIUS = 80

    mask_edge = getMaskEdge(hole_mask_full)
    grown_mask = hole_mask_full.copy()

    # Grow the mask 80 times to get a (roughly) 80-pixel radius
    for i in range(LC_CONTEXT_RADIUS):
        mask_edge = growBinaryMask(grown_mask, mask_edge)

    # Find the box surrounding this region (useful for the scoreCandidate function)
    mask_edge = np.array(mask_edge)
    topLeft = mask_edge.min(axis=0)
    bottomRight = mask_edge.max(axis=0)

    # Return only the grown region (LC) and the box around the LC
    grown_mask[hole_mask_full] = False
    return grown_mask, (topLeft[0], topLeft[1], bottomRight[0]-topLeft[0], bottomRight[1]-topLeft[1])  # Boolean mask image and box coords


def scoreCandidate(candidate_img: np.ndarray, lc_mask: np.ndarray, lc_box: tuple, original_img: np.ndarray):
    """
    Scores scene images by the metric of a weighted sum between the negative of the magnitude of the fill's translational
    offset, the pixel-wise SSD alignment score between the local context of the hole and the local context of the fill,
    and the SSD between the texture descriptor of the local context of the hole and the texture descriptor of the local
    context of the fill.

    Returns the best local context fit score for this candidate and coordinates where to align the LC box within the
    candidate image ()
    """
    PER_PIXEL_TRANSLATION_COST = 1e6  # TODO Adjust this value to properly punish far-away hole fills

    H, W, _ = candidate_img.shape

    M_crop_lc = lc_mask[lc_box[0]:lc_box[0]+lc_box[2], lc_box[1]:lc_box[1]+lc_box[3]].astype(np.uint8)          # Mask that is true on the LC region in the target image
    T_crop_lc = original_img[lc_box[0]:lc_box[0]+lc_box[2], lc_box[1]:lc_box[1]+lc_box[3]].astype(np.uint8)     # Target image containing the LC
    I = candidate_img.astype(np.float32)  # Candidate image that has regions which might fit the LC

    best_ssd_score = np.inf  # We are looking for a small SSD score
    best_lc_fit = None  # This will be (row_pos, col_pos, rows, cols)

    for scale_i, scale in enumerate(LC_SCALES):
        # Scale local context
        res_shape = (int(M_crop_lc.shape[0] * scale), int(M_crop_lc.shape[1] * scale))
        # The size given to cv2.resize needs to be in (width, height)
        res_lc_mask = cv2.resize(M_crop_lc, (res_shape[1], res_shape[0])).astype(np.float32)
        res_lc_temp = cv2.resize(T_crop_lc, (res_shape[1], res_shape[0])).astype(np.float32)

        # If we cannot fully fit the resized local context into an image, throw it out
        if (H-res_shape[0]+1) < 0 or (W-res_shape[1]+1) < 0:
            continue

        # Note that the SSD dimension changes depending on the scale of the template
        res_ssd_cost = np.zeros((H-res_shape[0]+1, W-res_shape[1]+1), dtype=np.float32)
        for ch in range(3):
            Tch = res_lc_temp[:,:,ch]
            Mch = res_lc_mask
            Ich = I[:,:,ch]
            uncropped_ssd_cost = ((Mch*Tch)**2).sum() - 2.0*cv2.filter2D(Ich, ddepth=-1, kernel=Mch*Tch, anchor=(0,0)) + \
                                 cv2.filter2D(Ich**2, ddepth=-1, kernel=Mch, anchor=(0,0))
            # Crops the SSD image so that only fully-fit SSD's are included
            res_ssd_cost += uncropped_ssd_cost[:-Tch.shape[0]+1, :-Tch.shape[1]+1]

        # Punish far translations- FIXME, how to account for different res_ssd_cost shapes from different
        row_coords, col_coords = np.meshgrid(np.arange(res_ssd_cost.shape[1]), np.arange(res_ssd_cost.shape[0]))
        perc_down = lc_box[0] / original_img.shape[0]
        perc_right = lc_box[1] / original_img.shape[1]
        res_top_left = (perc_down * res_ssd_cost.shape[0], perc_right * res_ssd_cost.shape[1])
        punishment_matrix = np.abs((row_coords - res_top_left[0]) + (col_coords - res_top_left[1]))  # res_top_left is wrong
        res_ssd_cost += PER_PIXEL_TRANSLATION_COST * punishment_matrix

        res_best_score = res_ssd_cost.min() / scale  # In order to account for a smaller kernel size, the SSD score is divided by the scale
        if res_best_score < best_ssd_score:  # Lower scores are better
            best_ssd_score = res_best_score
            best_lc_fit = np.unravel_index(res_ssd_cost.argmin(), res_ssd_cost.shape) + (res_shape[0], res_shape[1])

    return best_ssd_score, best_lc_fit


def paintRegion(canvas: np.ndarray, cutout_region: np.ndarray, cutout_mask: np.ndarray, dst_box_lc_crop: tuple):
    """
    Paints `cutout_region` onto `canvas` at the region specified by `dst_box_lc_crop` and `cutout_mask`.
    """

    assert cutout_region.shape[:2] == cutout_mask.shape

    src_H, src_W = cutout_region.shape[:2]
    dst_y, dst_x, dst_H, dst_W = dst_box_lc_crop

    if not (src_H == dst_H and src_W == dst_W):
        cutout_region = cv2.resize(cutout_region, (dst_W, dst_H))

    canvas[dst_y:dst_y+dst_H, dst_x:dst_x+dst_W][cutout_mask] = cutout_region[cutout_mask]


def findBestHoleFill(original_img: np.ndarray, hole_mask: np.ndarray, candidates: List[str]):
    lc_mask_full, lc_box = getLocalContext(original_img, hole_mask)
    lc_mask_full = lc_mask_full.astype(np.uint8)

    # This function returns the score of the best LC match and the region of pixels in the candidate that should be
    # resized the put in the target image
    best_candidate = None
    best_candidate_score = np.inf
    print('Looking at %d candidates...' % len(candidates))
    for i, candidate_path in enumerate(candidates):
        candidate = cv2.imread(candidate_path)
        score, src_lc_box = scoreCandidate(candidate, lc_mask_full == 1, lc_box, original_img)
        if score < best_candidate_score:
            best_candidate_score = score
            src_row, src_col, src_H, src_W = src_lc_box
            best_candidate = (candidate_path, src_row, src_col, src_H, src_W)

    lc_dim = (lc_box[3], lc_box[2])
    candidate_path, src_row, src_col, src_H, src_W = best_candidate
    best_candidate_img = cv2.imread(candidate_path)
    cutout_region = best_candidate_img[src_row:src_row + src_H, src_col:src_col + src_W]
    cutout_region = cv2.resize(cutout_region, lc_dim)

    cv2.imwrite('unfilled_target.jpg', original_img)
    # should be 1's
    hole_mask_lc_crop = hole_mask[lc_box[0]:lc_box[0] + lc_box[2], lc_box[1]:lc_box[1] + lc_box[3]].astype(np.uint8)
    # should be 0's
    lc_mask_lc_crop = lc_mask_full[lc_box[0]:lc_box[0] + lc_box[2], lc_box[1]:lc_box[1] + lc_box[3]].astype(np.uint8)
    orig_lc_crop = original_img[lc_box[0]:lc_box[0] + lc_box[2], lc_box[1]:lc_box[1] + lc_box[3]]

    # ===== Graphcuts processing =====
    # Build composite mask
    comp_mask = np.ones_like(hole_mask_lc_crop) * 2  # Background is 2's
    comp_mask[hole_mask_lc_crop == 1] = 1
    comp_mask[lc_mask_lc_crop == 1] = 0
    blended_lc_reg = graphCutSegmentation(cutout_region, comp_mask, orig_lc_crop).astype(np.uint8)

    # Blend entire LC box region into full image
    # blended_lc_reg_full = np.zeros_like(lc_mask_full)
    # blended_lc_reg_full[lc_box[0]:lc_box[0] + lc_box[2], lc_box[1]:lc_box[1] + lc_box[3]] = blended_lc_reg
    for i in range(3):
        cutout_region[:,:,i] = poissonBlend(cutout_region[:,:,i].astype(np.float64), blended_lc_reg == 1, orig_lc_crop[:,:,i].astype(np.float64), (0,0)).astype(np.uint8)

    filled_img = original_img.copy()
    filled_img[lc_box[0]:lc_box[0] + lc_box[2], lc_box[1]:lc_box[1] + lc_box[3]] = cutout_region
    # filled_img[lc_box[0]:lc_box[0] + lc_box[2], lc_box[1]:lc_box[1] + lc_box[3]] = cutout_region
    cv2.imwrite('filled_target_blended.jpg', filled_img)

    return filled_img


if __name__ == '__main__':
    target = cv2.imread(r"target_img.png", cv2.IMREAD_UNCHANGED)
    hole_mask_full = target[:, :, 3] == 0.0  # "full" means full size of target image
    target = target[:, :, [0, 1, 2]]

    findBestHoleFill(target, hole_mask_full, ['original_resized.jpg'])

