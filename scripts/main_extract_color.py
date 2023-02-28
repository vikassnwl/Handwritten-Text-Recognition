from dist_extract_color import extract_color_using_dist
from range_extract_color import extract_color_using_range
import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


# paths
inp_img_dir = "../data/cropped/"
out_img_dir = "../data/processed/"


for filename in tqdm(os.listdir(inp_img_dir)):
    inp_img_pth = f'{inp_img_dir}{filename}'

    # reading image in bgr
    img_bgr = cv2.imread(inp_img_pth)

    # lab dist based extraction
    extracted1 = extract_color_using_dist(img_bgr.copy(), "blue")
    # applying median blur to remove small noise
    blurred1 = cv2.medianBlur(extracted1, 5)

    # hsv range based extraction
    lower_range = np.array([110, 15, 0])
    upper_range = np.array([150,255,210])
    # upper_range = np.array([170,255,210])
    extracted2 = extract_color_using_range(img_bgr.copy(), lower_range, upper_range)
    # applying median blur to remove small noise
    blurred2 = cv2.medianBlur(extracted2, 5)
    # applying dilation to include the nearby pixels
    dilated2 = cv2.dilate(blurred2, np.ones((5, 10), np.uint8), iterations=2)

    # finding coordinates of white pixels from image having extracted color using lab dist based approach
    coords_white = set(map(lambda pt: tuple(pt), np.argwhere(blurred1 == 255)))


    ## TRAVERSING OVER THE IMAGE HAVING EXTRACTED COLORS USING HSV RANGE BASED APPROACH

    # finding height and width of the image
    h, w = img_bgr.shape[:2]

    # creating canvas to draw white pixels
    canvas = np.zeros(shape=(h, w))

    # update coords to traverse neighbour pixels using following values
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    # dx = [-1, 1, 0, 0, -1, 1, -1, 1]
    # dy = [0, 0, -1, 1, -1, -1, 1, 1]

    h, w = extracted2.shape
    canvas = np.zeros(shape=(h, w))

    def isValid(pt):
        if pt[0] < 0 or pt[0] > h-1 or pt[1] < 0 or pt[1] > w-1 or dilated2[pt] == 0 or vis[pt]:
            return False
        return True

    from collections import deque

    while coords_white:
        pt = coords_white.pop()
        q = deque()
        vis = np.zeros(shape=extracted2.shape)
        q.append(pt)
        vis[pt] = 1
        canvas[pt] = 255

        while q:
            curr_pt = q.popleft()
            for i in range(len(dx)):
                next_pt = curr_pt[0]+dx[i], curr_pt[1]+dy[i]
                if isValid(next_pt):
                    try:
                        coords_white.remove(next_pt)
                    except:
                        pass
                    q.append(next_pt)
                    vis[next_pt] = 1
                    if blurred2[next_pt] == 255:
                        canvas[next_pt] = 255


    # saving images
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(30, 30))
    axes[0].imshow(blurred1, cmap="gray")
    axes[1].imshow(blurred2, cmap="gray")
    axes[2].imshow(canvas, cmap="gray")
    # Save figure
    fig.savefig(f'{out_img_dir}{filename}', bbox_inches='tight')
