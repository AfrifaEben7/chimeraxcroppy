"""
Doing some tests to simplify our workflow to do everything automatically.
Steps:
1) Given a path from the microscope S:\\ transfer to the X:\\ drive
2) Perform the deskew, deconvolution (or not) and rotation
3) Crop down to the coverslip area
4) Ensure that MIP maintains the proper metadata i.e., Time, Pixel size, etc.
5) Transfer the raw data and cropped data to the Z:\\ drive
_____________________________
|                           |
|  ______________________   |
|  [  /              ]  /   |
|  [ /               ] /    | This represents the coverslip and parallelogram
|  [/________________]/     | of where data is relative to the padded data
|                           | when it is rotated that we need to crop to.
|___________________________|

"""

__author__ = "Brandon Scott"
__version__ = "0.2.0"
__license__ = "MIT"

import os
import argparse
from itertools import cycle
from easygui import diropenbox
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
from multiprocess.pool import Pool
from scipy.ndimage import generate_binary_structure, iterate_structure
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_erosion
import llspy


def robocopy(source: str, destination: str, file='*.*', move='', recur='') -> None:
    """ This function will copy the data to the SSD as fast as possible """
    # Flags: /W:1 /R:1 [Try once] [/NFL (Log start/stop)] /j (Unbuff I/O)
    # Flags: /MT:64 [Most important flag, specifies multithreading]
    # Flags: move= '/MOVE' will move the file rather than copy
    # Flags: recur = '/E' will recursively copy the tree
    os.system("robocopy {} {} {} /W:1 /R:1 /NFL /j /MT:64 {} {} ".format(
        source, destination, file, move, recur))


def threshold_li(image: np.ndarray) -> np.ndarray:
    """Return threshold value based on Li's MCE method.

    From skimage.filters.threshold_li
    Parameters
    ----------
    image : (N, M) ndarray
        Input image.
    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
    References
    ----------
    .. [1] Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"
           Pattern Recognition, 26(4): 617-625
           DOI:10.1016/0031-3203(93)90115-D
    .. [2] Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum
           Cross Entropy Thresholding" Pattern Recognition Letters, 18(8):
           771-776 DOI:10.1016/S0167-8655(98)00057-9
    .. [3] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165
           DOI:10.1117/1.1631315
    .. [4] ImageJ AutoThresholder: http://fiji.sc/wiki/index.php/Auto_Threshold
    """
    # Make sure image has more than one value
    if np.all(image == image.flat[0]):
        raise ValueError(
            "threshold_li is expected to work with images "
            "having more than one value. The input image seems "
            "to have just one value {0}.".format(image.flat[0])
        )

    # Copy to ensure input image is not modified
    image = image.copy()
    # Requires positive image (because of log(mean))
    immin = np.min(image)
    image -= immin
    imrange = np.max(image)
    tolerance = 0.5 * imrange / 256
    # Calculate the mean gray-level
    mean = np.mean(image)
    # Initial estimate
    new_thresh = mean
    old_thresh = new_thresh + 2 * tolerance

    # Stop the iterations when the difference between the
    # new and old threshold values is less than the tolerance
    while abs(new_thresh - old_thresh) > tolerance:
        old_thresh = new_thresh
        threshold = old_thresh + tolerance  # range
        # Calculate the means of background and object pixels
        mean_back = image[image <= threshold].mean()
        mean_obj = image[image > threshold].mean()
        temp = (mean_back - mean_obj) / (np.log(mean_back) - np.log(mean_obj))

        if temp < 0:
            new_thresh = temp - tolerance
        else:
            new_thresh = temp + tolerance
    out_thresh = threshold + immin
    return out_thresh


def diamond(nsize=4) -> np.ndarray:
    """Helper function to generate the diamond structuring element"""
    struct = iterate_structure(
        generate_binary_structure(2, 1), nsize).astype(int)
    return struct


def coverslipbounds(img, sigma=2, pad=10) -> np.ndarray:
    """Returns the min and max for z and y on coverslip.
        Uses coverlsip and the parallelogram to crop.
    """
    if img.ndim == 3:
        # Uses the yz projection as this displays the full shape
        img = np.squeeze(img.max(1))
    img = img.astype(float)
    mask = img > 0  # This gives the position of padded parallelogram
    # Erode the sides to ensure edge artifacts are removed.
    mask_erode = binary_erosion(mask, diamond(15))

    im_gaus = gaussian_filter(img, sigma)
    coverslip = im_gaus > threshold_li(im_gaus)
    # Helps to clean up obj further
    coverslip_erode = binary_erosion(coverslip * mask_erode, diamond(4))
    bounds = np.zeros(4, dtype=int)
    bounds[0] = np.nonzero(np.sum(coverslip_erode, -1))[0][0] - pad
    bounds[1] = np.nonzero(np.sum(mask, -1))[0][-1]

    # The following crops in y to the edge at what is the coverslip.
    mask_edge = np.where(mask[bounds[0]])[0]
    bounds[2], bounds[3] = mask_edge[0], mask_edge[-1]
    return bounds


def crop_yz(img: np.ndarray, cropping_info: np.ndarray, is_mip=False) -> np.ndarray:
    """This will crop the data in y and z, keeping x at the total length"""
    zmin, zmax = cropping_info[:2]
    ymin, ymax = cropping_info[-2:]
    if img.ndim == 3:
        if is_mip:  # Need to crop in y, but keep time
            img = img[:, :, ymin:ymax]
        else:  # Crop in yz
            img = img[zmin:zmax, :, ymin:ymax]
    elif img.ndim > 3:
        if is_mip:  # Need to crop in y, but keep time
            img = img[:, :, :, ymin:ymax]
        else:  # Crop in yz
            img = img[..., zmin:zmax, :, ymin:ymax]
    return img


def determine_filenames(source=None, channel="488") -> list:
    """Find which files need used"""
    if not source:
        source = diropenbox(msg='Choose the directory with images')
        if not source:
            print('Directory not chosen, exiting.')
            return None
        source = source + '\\'
    print('Input Directory is', source)
    os.chdir(source)
    filenames = list(filter(lambda a: ".tif" in a, os.listdir(source)))
    crop_file = list(filter(lambda a: channel + "nm" in a, filenames))[0]
    cropping_info = coverslipbounds(tf.imread(source+crop_file))
    inputs = list(zip(filenames, cycle([cropping_info])))
    return inputs


def read_crop_write(inputs: list) -> None:
    """This is where the cropping occurs """
    filename = inputs[0]
    cropping_info = inputs[1]
    data = tf.imread(filename)
    data = crop_yz(data, is_mip=False, cropping_info=cropping_info)
    tf.imsave(filename, data, compress=5)


def read_crop_write_mip(filename: str, cropping_info: np.ndarray) -> None:
    """Reads in data, crops MIP, and writes a new MIP """
    md_orig = dict()
    with tf.TiffFile(filename) as tif:
        data = tif.asarray()
        md_orig = tif.imagej_metadata
    tif.close()
    data = crop_yz(data, is_mip=True, cropping_info=cropping_info)

    # Array must be in TZCYX order
    if md_orig.get('channels'):
        axis = 1
    else:
        axis = (1, 2)
    data = np.expand_dims(data, axis=axis)

    meta_data = {
        "unit": md_orig.get('unit', 'micron'),
        "spacing": md_orig.get('spacing', 1),
        "finterval": md_orig.get('finterval', 0),
        "hyperstack": True,
        "mode": "composite",
        "loop": True,
    }
    tf.imsave(
        filename,
        data,
        imagej=True,
        resolution=(1/0.1028, 1/0.1028),
        metadata=meta_data,
    )


def show_images(data: np.ndarray, axis=0) -> None:
    """Simple function to display max images"""
    if data.ndim == 3:
        plt.imshow(np.max(data, axis=axis))
    else:
        plt.imshow(data)
    plt.show()


def main(source=None, channel="488", crop_switch=False, copy_switch=False) -> None:
    """1) Given a path from the microscope S:\\ transfer to the X:\\ drive
   2) Perform the deskew, deconvolution (or not) and rotation
   3) Crop down to the coverslip area
   4) Ensure that MIP maintains the proper metadata: Time, Pixel size, etc.
   5) Transfer the raw data and cropped data to the Z:\\ drive
   If crop_switch is True then the deconvolution will not be run.
   """
    if not source:
        source = diropenbox()
        if not source:
            print('Directory not chosen, exiting.')
            return
        if crop_switch:
            if not source.endswith("GPUdecon"):
                addme = "GPUdecon\\"
            else:
                addme = "\\"
            source = os.path.join(source, addme)
        os.chdir(source)

    print('Input Directory is: ', source)

    if copy_switch:
        destination = "X:\\DeconSandbox\\"
        hidden_dir = "X:\\.empty\\"
        if not os.path.exists(hidden_dir):
            os.makedirs(hidden_dir)
        # Copy files using 64 threads, and no logging information displayed.
        robocopy(source, destination, '*.*')
    else:
        destination = source

    if not crop_switch:  # This will do the deskew, decon, and rotate
        lls_data = llspy.LLSdir(destination)
        options = {"correctFlash": False,
                   "nIters": 10,
                   "otfDir": 'Z:/data/LLSM_Alignment/OTF/',
                   "background": 95,
                   "bRotate": True,
                   "rMIP": (True, True, True),
                   "rotate": lls_data.parameters.angle}
        # Perform the deskewing, decon, and rotation.
        lls_data.autoprocess(**options)

    # Perform autocropping
    inputs = determine_filenames(
        source=os.path.join(destination, "GPUdecon\\"),
        channel=channel)
    mip_path = os.path.join(destination, "GPUdecon\\MIPs\\")
    if os.path.exists(mip_path):
        mip_file = os.listdir(mip_path)[0]
        read_crop_write_mip(mip_path + mip_file, inputs[0][-1])

    Pool().map(read_crop_write, inputs)

    if copy_switch:
        # Move files using 64 threads, and no logging information displayed.
        os.chdir("X:\\")
        robocopy(destination, source, '*.txt', '/MOVE')
        robocopy(destination+"GPUdecon\\", source +
                 "GPUdecon\\", '*.*', '/MOVE', '/E')

        # Remove the raw data that was copied to destination.
        robocopy(hidden_dir, destination, '*.*', '/MIR')
        if os.path.exists(hidden_dir):
            os.removedirs(hidden_dir)


if __name__ == '__main__':
    # This is executed when run from the command line.
    parser = argparse.ArgumentParser()
    # Optional argument flag for the source directory
    parser.add_argument(
        "-s",
        "--source",
        action="store",
        help="The directory path containing data",
        default=None)

    # Optional argument flag for channel selection
    parser.add_argument(
        "-c",
        "--channel",
        action="store",
        help="Channel for autocropping",
        default="488")

    # Optional argument flag for cropping only
    parser.add_argument(
        "-x",
        "--crop",
        action="store_true",
        dest="crop_switch",
        help="Use this flag if you only want to crop.",
        default=False)

    # Optional argument flag for local processing
    parser.add_argument(
        "-m",
        "--copy",
        action="store_true",
        dest="copy_switch",
        help="Use this flag if you want to copy to the X:/ before processing.",
        default=False)

    # Specify output of "--version"
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))
    args = parser.parse_args()

    main(
        source=args.source,
        channel=args.channel,
        crop_switch=args.crop_switch,
        copy_switch=args.copy_switch
    )
