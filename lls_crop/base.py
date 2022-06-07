"""
Chimeraxcroppy base module.

Doing some tests to simplify our workflow to do everything automatically.
Steps:
1) Given a path from the Z:\\ transfer to the X:\\ drive
2) Perform the deskew, deconvolution (or not) and rotation
3) Crop down to the coverslip area
4) Ensure that MIP maintains the proper metadata i.e., Time, Pixel size, etc.
5) Transfer the cropped data back to the Z:\\ drive
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
__version__ = "0.6.0"
__license__ = "MIT"

from os import chdir, listdir, system
from itertools import cycle

from numpy import ndarray, expand_dims, where, sum, max, squeeze, nonzero, zeros
from tifffile import TiffFile, imread, imsave
from scipy.ndimage import generate_binary_structure, iterate_structure
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_erosion
from matplotlib.pyplot import imshow, show
from skimage.filters import threshold_yen


def robocopy(source: str, destination: str, file: str = '*.*', move: str = '', recur: str = '') -> None:
    """ This function will copy the data to the SSD as fast as possible """
    # Flags: /W:1 /R:1 [Try once] [/NFL (Log start/stop)] /j (Unbuff I/O)
    # Flags: /MT:64 [Most important flag, specifies multithreading]
    # Flags: move= '/MOVE' will move the file rather than copy
    # Flags: recur = '/E' will recursively copy the tree
    system("robocopy {} {} {} /W:1 /R:1 /NFL /j /MT:64 {} {} ".format(
        source, destination, file, move, recur))


def diamond(nsize: int = 4) -> ndarray:
    """Helper function to generate the diamond structuring element"""
    struct = iterate_structure(
        generate_binary_structure(2, 1), nsize).astype(int)
    return struct


def coverslipbounds(img: ndarray, sigma: int = 2, pad: int = 10) -> ndarray:
    """Returns the min and max for z and y on coverslip.
        Uses coverslip and the parallelogram to crop.
        Changed in version 0.6.0 to use threshold_yen for finding the coverslip based on skimage.filters.try_all_threshold() results.
        fig, ax = try_all_threshold(im_gaus, figsize=(10, 8), verbose=False)
    """
    if img.ndim == 3:
        # Uses the yz projection as this displays the full shape
        img = squeeze(img.max(1))
    img = img.astype(float)
    mask = img > 0  # This gives the position of padded parallelogram
    # Erode the sides to ensure edge artifacts are removed.
    mask_erode = binary_erosion(mask, diamond(25))

    im_gaus = gaussian_filter(img * mask_erode, sigma)
    coverslip = im_gaus > threshold_yen(im_gaus)
    # Helps to clean up obj further
    coverslip_erode = binary_erosion(coverslip, diamond(4))
    bounds = zeros(4, dtype=int)
    bounds[0] = nonzero(sum(coverslip_erode, -1))[0][0] - pad
    bounds[1] = nonzero(sum(mask, -1))[0][-1]

    # The following crops in y to the edge at what is the coverslip.
    mask_edge = where(mask[bounds[0]])[0]
    bounds[2], bounds[3] = mask_edge[0], mask_edge[-1]
    return bounds


def crop_yz(img: ndarray, cropping_info: ndarray, is_mip: bool = False) -> ndarray:
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


def determine_filenames(source: str, channel: str = "488") -> list:
    """Find which files need used"""
    chdir(source)
    filenames = list(filter(lambda a: ".tif" in a, listdir(source)))
    crop_file = list(filter(lambda a: channel + "nm" in a, filenames))[0]
    cropping_info = coverslipbounds(imread(source+crop_file))
    inputs = list(zip(filenames, cycle([cropping_info])))
    return inputs


def read_crop_write(inputs: list) -> None:
    """This is where the cropping occurs """
    filename = inputs[0]
    cropping_info = inputs[1]
    data = imread(filename)
    data = crop_yz(data, is_mip=False, cropping_info=cropping_info)
    imsave(filename, data, compress=5)


def read_crop_write_mip(filename: str, cropping_info: ndarray) -> None:
    """Reads in data, crops MIP, and writes a new MIP """
    md_orig = dict()
    with TiffFile(filename) as tif:
        data = tif.asarray()
        md_orig = tif.imagej_metadata
    tif.close()
    data = crop_yz(data, is_mip=True, cropping_info=cropping_info)

    # Array must be in TZCYX order
    if md_orig.get('channels'):
        axis = 1
    else:
        axis = (1, 2)
    data = expand_dims(data, axis=axis)

    meta_data = {
        "unit": md_orig.get('unit', 'micron'),
        "spacing": md_orig.get('spacing', 1),
        "finterval": md_orig.get('finterval', 0),
        "hyperstack": True,
        "mode": "composite",
        "loop": True,
    }
    imsave(
        filename,
        data,
        imagej=True,
        resolution=(1/0.1028, 1/0.1028),
        metadata=meta_data,
    )


def show_images(data: ndarray, axis: int = 0) -> None:
    """Simple function to display max images"""
    if data.ndim == 3:
        imshow(max(data, axis=axis))
    else:
        imshow(data)
    show()
