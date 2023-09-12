import argparse
import pathlib
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # to enable opening of large images
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


def get_difference_from_local_mean_of_lidar_measurement(image_path):
    """
    A flat surface have the same value at a specific point as the mean of the local area.
    If the terrain is changing in elevation a lot this is in general not true.
    Subtracting the local mean from the value at a point therfore gives an indicator of how the lidar values in that area differs.
    :param image_path: path to a lidar DTM data image
    :return: the data - the blurred version of the data
    """
    orig_img = Image.open(image_path)
    blurr_img = ndimage.gaussian_filter(orig_img, sigma=(5, 5), order=0)

    #difference between negative slope and positive slope
    difference_im = abs(np.array(orig_img)- np.array(blurr_img))
    min = difference_im.flatten().min()
    max = difference_im.flatten().max()

    normalized_difference = np.array(255*((difference_im-min)/((max-min)+0.0000000001)),dtype=np.uint8)
    difference_im= Image.fromarray(normalized_difference)

    return difference_im




if __name__ == "__main__":
    example_usage= r"python data_preprocessing.py -i path/to/lidar_im.tif "
    print("visualizing the lidar measurements")
    print("best used together with DTM data")

    print("########################EXAMPLE USAGE########################")
    print(example_usage)
    print("#############################################################")


    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--lidar_image", help="path/to/lidar.png  or path/to/folder",required=True)

    args = parser.parse_args()
    image_path = pathlib.Path(args.lidar_image)
    difference_im = get_difference_from_local_mean_of_lidar_measurement(image_path)
    plt.imshow(difference_im, interpolation='nearest')
    plt.show()
