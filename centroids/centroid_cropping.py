"""Load in tif files, crop based on track, and save to subfolder

Parameters
----------

image_dir: directory where the images are found
centroids: The tracked centroids from trackpy as pandas DataFrame

Outputs
-------

track_crop: Subset of images which have been cropped and saved as subfolder

"""
__author__ = "Brandon Scott"
__version__ = "0.1.0"
__license__ = "MIT"

import os
from pathlib import Path
from functools import partial
from multiprocess.pool import Pool
import xml.etree.ElementTree as et
import pprint

import numpy as np
import easygui
import pandas
import tifffile


def choose_directory(source=None, dir_flag=True):
    """Choose the image folder or cmm file"""
    count = 0
    while not source and count < 3:
        if dir_flag:
            source = easygui.diropenbox(msg='Choose the image directory')
        else:
            source = easygui.fileopenbox(
                msg='Choose the cmm file', default='*.cmm')
        count += 1
    if not source:
        pprint.pprint('You need to choose, exiting...')
        return source

    if dir_flag:
        source = source + '\\'
        pprint.pprint('Image directory is: ' + source)
    else:
        pprint.pprint('Centroid file is: ' + source)
    return source


def xml_2_pd(xml_file):
    """Convert input xml into pandas dataframe for cropping"""
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    all_marks = []
    for child in xroot:
        track = child.attrib
        for subchild in child:
            yval = int(np.round(float(subchild.attrib['y'])))
            xval = int(np.round(float(subchild.attrib['x'])))
            frame = int(float(subchild.attrib['frame']))
            subch = subchild.attrib
            subch.update(track)
            subch.update({'y': yval, 'x': xval, 'frame': frame})
            all_marks.append(subch)
    return pandas.DataFrame(all_marks, columns=['y', 'x', 'frame', 'name'])


def crop_image(data, cds):
    '''Crop the data with padded coordinates'''
    cropped = data[..., cds[0]:cds[1], cds[2]:cds[3]]
    return cropped


def write_cropped(file, subfolder, cropped):
    '''Append the subfolder to directory and save with compression'''
    s_dir = file.parent
    f_name = file.name
    src = Path.joinpath(s_dir, subfolder, f_name)
    tifffile.imwrite(src, cropped, compress=5)


def get_info(data_frame, frame):
    """Convert grouped data_frame to info"""
    info = data_frame.get_group(frame).to_numpy()
    return info


def file_from_pattern(file_dir, frame):
    """Find the image for the given stack in the file_directory"""
    fpattern = '*stack{:04d}*.tif'.format(frame)
    fname = Path(file_dir).glob(fpattern)
    return fname


def get_crop_coords(pad, dims, val):
    '''Given the input image and pad, define the ROI'''
    coords = np.zeros(4).astype(int)
    coords[0] = np.max((val[0]-pad, 0))
    coords[1] = np.min((val[0]+pad, dims[0]))
    coords[2] = np.max((val[1]-pad, 0))
    coords[3] = np.min((val[1]+pad, dims[1]))
    return coords


def crop_here(file_dir, pad, d_frame, frame):
    '''Set up so we can use partial and map async'''
    files = file_from_pattern(file_dir, frame)
    for name in files:
        data = tifffile.imread(name)
        dims = data.shape[-2:]
        infos = get_info(d_frame, frame)
        for info in infos:
            cds = get_crop_coords(pad, dims, info)
            cropped = crop_image(data, cds)
            write_cropped(name, info[-1], cropped)


def main(file_dir=None, xml_file=None, pad=100):
    """Load each time point and save subregions for any tracks"""
    if not file_dir:
        file_dir = choose_directory()
        if not file_dir:
            return
    os.chdir(file_dir)

    if not xml_file:
        xml_file = choose_directory(dir_flag=False)
        if not xml_file:
            return

    data_frame = xml_2_pd(xml_file)
    frames = list(np.unique(data_frame['frame']))
    d_frame = data_frame.groupby('frame')

    for name in np.unique(data_frame['name']):
        if not os.path.exists(name):
            os.mkdir(name)

    par_crop = partial(crop_here, file_dir, pad, d_frame)
    Pool().map(par_crop, frames)


if __name__ == '__main__':
    main()
