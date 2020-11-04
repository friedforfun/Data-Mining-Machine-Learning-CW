import matplotlib.pyplot as plt
import math
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean

def downscale(X_data, downscale_shape=(2,2)):
    """Downscale the images along each row, using local mean downscaling

    :param X_data: The dataset
    :type X_data: numpy.array
    :param downscale_shape: The degree of downscaling on each axis, defaults to (2,2)
    :type downscale_shape: tuple, optional
    :return: X_data but downscaled
    :rtype: numpy.array
    """
    return np.apply_along_axis(downscale_image, 1, X_data)

def downscale_image(image, downscale_shape=(2, 2)):
    """Downscale an image vector using local mean downscaling

    :param image: The image vector to downscale
    :type image: numpy.array
    :param downscale_shape: The degree of downscaling to perform on each asix, defaults to (2, 2)
    :type downscale_shape: tuple, optional
    :return: The image vector but downscaled
    :rtype: numpy.array
    """
    d = int(round(math.sqrt(image.shape[0]), 0))
    image = image.reshape(d,d)
    downscaled = downscale_local_mean(image, downscale_shape)
    downscaled = downscaled.flatten()
    return downscaled

def rescale_image(image, rescale_amt=0.25):
    """rescale an image vector
    :param image: The image vector to downscale
    :type image: numpy.array
    :param downscale_shape: The degree of downscaling to perform on each asix, defaults to (2, 2)
    :type downscale_shape: tuple, optional
    :return: The image vector but downscaled
    :rtype: numpy.array
    """
    d = int(round(math.sqrt(image.shape[0]), 0))
    image = image.reshape(d, d)
    rescaled = rescale(image, rescale_amt, anti_aliasing=False)
    rescaled = rescaled.flatten()
    return rescaled

def visualise_downsample(img):
    """Display the image & its downsampled versions
    From: https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py

    :param img: A single image vector with 2304 pixels
    :type img: numpy.array
    """
    img = img.reshape(48, 48)

    image_rescaled = rescale(img, 0.25, anti_aliasing=False)
    image_resized = resize(img, (img.shape[0] // 4, img.shape[1] // 4), anti_aliasing=True)
    image_downscaled = downscale_local_mean(img, (2, 2))

    fig, axes = plt.subplots(nrows=2, ncols=2)

    ax = axes.ravel()

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(image_rescaled, cmap='gray')
    ax[1].set_title("Rescaled image (aliasing)")

    ax[2].imshow(image_resized, cmap='gray')
    ax[2].set_title("Resized image (no aliasing)")

    ax[3].imshow(image_downscaled, cmap='gray')
    ax[3].set_title("Downscaled image (no aliasing)")

    plt.tight_layout()
    plt.show()
