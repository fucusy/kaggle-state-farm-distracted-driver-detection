__author__ = 'fucus'

from skimage import color, exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
import skimage.io
from config import Project as p
from skimage.feature import local_binary_pattern
import numpy as np
from skimage import data

def hist(ax, lbp):
    n_bins = lbp.max() + 1
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')


def get_lbp(img):
    # settings for LBP
    radius = 4
    n_points = 8 * radius
    METHOD = 'uniform'
    lbp = local_binary_pattern(color.rgb2gray(img), n_points, radius, METHOD)
    return lbp


def get_lbp_his(img):
    lbp = get_lbp(img)
    flat = lbp.reshape(-1)
    max_num = 256

    his = np.zeros(max_num)
    for x in flat:
        if x >= 0 and x < max_num:
            his[int(x)] += 1
    return his


if __name__ == '__main__':
    img = skimage.io.imread(p.test_img_example_path)
    lbp = get_lbp(img)



    img2 = skimage.io.imread(p.test_img_folder_path_2)
    lbp2 = get_lbp(img2)

    # plot histograms of LBP of textures
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(9, 6))
    plt.gray()

    ax1.imshow(img)
    ax1.axis('off')

    hist(ax2, lbp)

    ax3.imshow(img2)
    ax3.axis('off')
    hist(ax4, lbp2)


    plt.show()