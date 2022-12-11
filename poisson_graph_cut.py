import cv2
import maxflow
import scipy
import scipy.sparse.linalg
import numpy as np
from typing import Tuple, Any, List

# Seamlessly merge two pictures together given the pictures and a mask. Because we do not want to 
# cut across any important solid objects for either picture, we need to use graph cut 
# segmentation to determine the best seam to cut both pictures at and combine. This is done by 
# using the max-flow min-cut theorem, where the cost depends on the SSD of the pixel intensities.
# Once we have found our new mask, we use Poisson blending to combine the two pictures while looking
# visually believable.

# Can be used universally, it is used here to compute gradient differences
def SSD(a, b):
    """
    Returns the sum of square diffferences between two pixels
    a: the first pixel we wish to compute the loss function for 
    b: the second pixel we wish to compute the loss function for 
    """ 
    assert a.shape == b.shape

    ssd = 0.0
    for i in range(a.shape[0]):
        ssd += (a[i] - b[i]) ** 2

    ssd = ssd ** 0.5

    return ssd

# Still needs testing, using Jupyter notebook
# Might be worthwhile to test different cost functions here. This currently compares differences in 
# gradient, as the paper says this is better than just comparing pixel intensities between the two 
# pictures. I believe other sites which have worked upon the original paper have also noted better results
# from using frequencies rather than direct intensities, which I have not tried.
def graphCutSegmentation(object_img, mask, bg_img):
    """
    Compute the composite mask between two images using the graph-cut algorithm, given which pixels must
    be from the object and background image
    object_img: the image containing the foreground object
    mask: a color mask - here, 0 = part of the mask we wish to solve for, 
                               1 = pixels must come from object_img, (candidate)
                               2 = pixels must come from bg_img (original)
    bg_img: the background image
    """ 
    # Create initial graph
    num_nodes = mask.shape[0] * mask.shape[1]
    num_edges = 4 * num_nodes
    graph = maxflow.Graph[float](num_nodes, num_edges)
    nodes = graph.add_nodes(num_nodes)

    # Map each pixel we want to solve for to a number, representing a node
    im2var = {}
    index = 0
    for j in range(mask.shape[0]):
        for i in range(mask.shape[1]):
            im2var[(j, i)] = index
            index += 1

    # Fill the rest of the graph with edges
    for j in range(mask.shape[0]):
        for i in range(mask.shape[1]):
            # Connect pixel from picture 1 to dest node
            if (mask[j, i] == 1):
                graph.add_tedge(nodes[im2var[(j, i)]], 0, np.inf)
            # Connect pixel from picture 2 to source node
            if (mask[j, i] == 2):
                graph.add_tedge(nodes[im2var[(j, i)]], np.inf, 0)
            # Compute cost from SSD based on gradient, add edge
            if (j + 1 < mask.shape[0]):
                cost = (SSD(object_img[j, i], bg_img[j, i]) + SSD(object_img[j + 1, i], bg_img[j + 1, i])) 
                #/ (((object_img[j, i] - object_img[j + 1, i]) ** 2).sum() + ((bg_img[j, i] - bg_img[j + 1, i]) ** 2).sum() + 1)
                #cost = SSD(object_img[j, i], object_img[j + 1, i]) + SSD(bg_img[j, i], bg_img[j + 1, i])
                graph.add_edge(nodes[im2var[(j, i)]], nodes[im2var[(j + 1, i)]], cost, cost)
            if (i + 1 < mask.shape[1]):
                cost = SSD(object_img[j, i], bg_img[j, i]) + SSD(object_img[j, i + 1], bg_img[j, i + 1]) 
                #/ (((object_img[j, i] - object_img[j, i + 1]) ** 2).sum() + ((bg_img[j, i] - bg_img[j, i + 1]) ** 2).sum() + 1)
                #cost = SSD(object_img[j, i], object_img[j, i + 1]) + SSD(bg_img[j, i], bg_img[j, i + 1])
                graph.add_edge(nodes[im2var[(j, i)]], nodes[im2var[(j, i + 1)]], cost, cost)

    flow = graph.maxflow()

    # Complete the mask
    graph_mask = np.zeros((mask.shape[0], mask.shape[1]))
    for cor, index in im2var.items():
        graph_mask[cor[0], cor[1]] = int(graph.get_segment(nodes[index]))

    # 0 = object_img
    # 1 = bg_img
    return graph_mask

# Pretty much what was implemented from Project 3, this was taken from the Poisson blend function, 
# as implemented by dnh2. 
def poissonBlend(object_img, object_mask, bg_img, bg_ul):
    """
    Returns a Poisson blended image with masked object_img over the bg_img at position specified by bg_ul.
    Can be implemented to operate on a single channel or multiple channels
    object_img: the image containing the foreground object
    object_mask: the mask of the foreground object in object_img
    background_img: the background image 
    bg_ul: position (row, col) in background image corresponding to (0,0) of object_img 
    """ 
    # Map each pixel to a variable number
    im_h, im_w = object_img.shape
    im2var = np.arange(im_h * im_w).reshape(im_h, im_w)
    
    # Create matrices for least sqaures - (Av - b)^2
    neq = (4 * im_h * im_w) - im_h - im_w
    A = scipy.sparse.lil_matrix((neq, im_h * im_w), dtype='double')
    b = np.zeros((neq, 1), dtype='double')
    
    # Extract the background patch equal to the location of the source image
    bg_patch = bg_img[bg_ul[0]:bg_ul[0] + im_h, bg_ul[1]:bg_ul[1] + im_w].copy()
    
    # Minimize the Poisson gradient equation
    e = -1
    for x in range(im_w):
        for y in range(im_h):
            # Only worry about intensity values for pixels in mask
            if object_mask[y][x]:
                # Go through the four neighbors of pixel
                neighbors = [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
                for neighbor in neighbors:
                    j = neighbor[0]
                    i = neighbor[1]
                    # Ensure the neighbor is valid
                    if j in range(im_h) and i in range(im_w):
                        # Objective 1: Minimize gradient of two variable pixels
                        if object_mask[j][i]:
                            e = e + 1
                            A[e, im2var[y][x]] = 1
                            A[e, im2var[j][i]] = -1
                            b[e] = object_img[y][x] - object_img[j][i]
                        
                        # Objective 2: Minimize gradient of one variable and one target region
                        else:
                            e = e + 1
                            A[e, im2var[y][x]] = 1
                            b[e] = bg_patch[j][i] + object_img[y][x] - object_img[j][i]
                            
    # Solve for v in least-squares problem
    v = scipy.sparse.linalg.lsqr(A.tocsr(), b, atol=1e-12, btol=1e-12);
    v_patch = v[0].reshape((im_h, im_w))

    # Modify the patch to include new intensity values of source
    output_patch = (v_patch * object_mask) + (bg_patch * (abs(object_mask - 1)))
    
    # Copy patch back into background image
    output_img = bg_img.copy()
    # output_patch -= np.min(output_patch)
    # output_patch /= np.max(output_patch)
    # output_patch *= 255.0
    output_patch = np.clip(output_patch, 0.0, 255.0)
    output_img[bg_ul[0]:bg_ul[0] + im_h, bg_ul[1]:bg_ul[1] + im_w] = output_patch

    return output_img

if __name__ == '__main__':
    pass