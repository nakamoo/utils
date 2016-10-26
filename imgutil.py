import os
import math
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Agg')  # Workaround to save images when running over ssh sessions
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def tile_ims(filename, directory):
    """Load all images in the given directory and tile them into one."""
    im_path = os.listdir(directory)
    im_path = filter(lambda x: x[0] != '.', im_path)
    ims = [mpimg.imread(os.path.join(directory, f)) for f in
           sorted(im_path)]
    save_ims(filename, np.array(ims))


def save_im(filename, im):
    # h, w, c = im.shape
    cv.imwrite(filename, im)


def save_ims(filename, ims):
    n, h, w, c = ims.shape

    # Plot the images on a grid
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(round(math.sqrt(n)))

    # Each subplot should have the same resolutions as the image dimensions

    # TODO: Consider proper heights and widths for the subplots
    h = 64
    w = 64

    fig, axes = plt.subplots(rows, cols, figsize=(h, w))
    fig.subplots_adjust(hspace=0, wspace=0)

    for i, ax in enumerate(axes.flat):
        ax.axis('off')  # Hide x, y axes completely
        if i < n:
            ax.imshow(ims[i])

    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
