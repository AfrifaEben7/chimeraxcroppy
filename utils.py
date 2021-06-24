"""Utilities for common tasks """
__author__ = "Brandon Scott"
__version__ = "0.2.1"
__license__ = "MIT"

import os
import easygui
import numpy as np
import tifffile as tf
import trackpy as tp
import matplotlib.pyplot as plt


def choose_directory(source=None, dir_flag=True):
    """Choose either a folder or file"""
    count = 0
    while not source and count < 3:
        if dir_flag:
            source = easygui.diropenbox(msg='Choose a directory')
        else:
            source = easygui.fileopenbox(msg='Choose a file')
        count += 1
    if not source:
        print('You failed choose, exiting...')
        return source
    return source


def find_all_files(source, file_ext='.tif'):
    """Find all tiff files in the directory """
    files = [os.path.join(direct[0], file) for direct in os.walk(source)
             for file in direct[2] if file.endswith(file_ext)]
    return files


def find_specific_files(files, pattern='_thumb'):
    """Create two lists, one with and one without pattern"""
    pattern_found = [x for x in files if pattern in x]
    pattern_absent = [x for x in files if pattern not in x]
    return pattern_found, pattern_absent


def find_save_all(source, files):
    """Loop through all the possible sites, max of 10"""
    ind = 1
    _, oth = find_specific_files(files, pattern='_thumb')
    _, oth = find_specific_files(oth, pattern='_combo')
    while oth:
        print('site: {}'.format(ind))
        subf, oth = find_specific_files(oth, pattern='_s{}_'.format(ind))
        save_sorted_recur(source, subf, 's{}_'.format(ind))
        ind += 1


def save_sorted_recur(source, sorted_files, site='_s1'):
    """Save all the wavelengths as a combo stack"""
    ind = 1
    print('wavelength: {}'.format(ind))
    wv_ind, oth = find_specific_files(sorted_files, pattern='_w{}'.format(ind))
    comb_img = tf.imread(wv_ind)
    comb_img = np.expand_dims(comb_img, axis=-1)
    ind += 1
    while oth:
        print('wavelength: {}'.format(ind))
        wv_ind, oth = find_specific_files(oth, pattern='_w{}'.format(ind))
        new_img = tf.imread(wv_ind)
        new_img = np.expand_dims(new_img, axis=-1)
        comb_img = np.append(comb_img, new_img, axis=-1)
        ind += 1

    tf.imwrite(os.path.join(source, site + 'combo.tif'), comb_img)


def detect_centroids(source, site, size=17):
    """Find where the RBCs are in the movie"""
    data = tf.imread(os.path.join(source, site + 'combo.tif'))
    data = np.moveaxis(data, -1, 0)
    frames = data[-1, ...]
    centroids = tp.locate(frames[0], size, minmass=1000)

    return centroids


def main():
    """Find all the files for a given set of directories """
    source = choose_directory()
    files = find_all_files(source, file_ext='.tif')
    find_save_all(source, files)


if __name__ == '__main__':
    main()
