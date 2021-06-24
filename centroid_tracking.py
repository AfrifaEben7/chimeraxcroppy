"""Import centroids from ChimeraX, remove duplicates, creates tracks,
fill gaps and returns XML file with the tracks.

Parameters
----------


"""
__author__ = "Brandon Scott"
__version__ = "0.2.0"
__license__ = "MIT"

import os
import pprint
import itertools
import xml.etree.ElementTree as et

import easygui
import numpy as np
import pandas as pd
import trackpy as tp
from scipy.spatial import distance


def choose_xml():
    """Choose the centroids cmm"""
    source = None
    count = 0
    while not source and count < 3:
        source = easygui.fileopenbox(
            msg='Choose the centroids cmm', default='C:\\ProgramData\\ChimeraX\\*.cmm')
        count += 1
    if not source:
        pprint.pprint('You forgot to choose the file')
        return source
    pprint.pprint('Centroids file is: ' + source)
    return source


def xml_2_pd(xml_file):
    """Convert input xml into pandas dataframe"""
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    all_marks = []
    for child in xroot:
        all_marks.append(child.attrib)
    data_frame = pd.DataFrame(
        all_marks, columns=['id', 'x', 'y', 'z', 'frame'])
    return data_frame.dropna(axis='index', how='any')


def remove_close(frame, closest=10):
    """Remove duplicate centroids leftover from chimeraX"""
    # Take only the coordinates
    frame = frame[1]
    points_array = frame.values[:, 1:]
    dist = distance.pdist(points_array[:, :-1])
    too_close = np.where(dist < closest)[0]
    combos = list(itertools.combinations(range(frame.shape[0]), 2))
    for val in too_close:
        points_array[combos[val][1], 0] = np.nan
    return pd.DataFrame(points_array, columns=frame.columns[1:])


def remove_close_time(data_frame, closest=10):
    """Loop over time and remove duplicate centroids"""
    frames = data_frame.groupby('frame')
    frames = [remove_close(frame, closest) for frame in frames]
    frames = pd.concat(frames, ignore_index=True)
    frames = frames.dropna()
    return frames


def track_centroids(data_frame, gap=2, min_len=10):
    """Uses trackpy to link centroids into tracks"""
    tp.quiet()
    tracks = tp.link_df(data_frame, search_range=50, pos_columns=[
        'x', 'y', 'z'], t_column='frame', memory=gap)
    tracks = tp.filter_stubs(tracks, min_len)
    label1 = tracks['particle'].unique()
    tracks['track'] = tracks.apply(lambda x: np.argwhere(
        label1 == x.particle)[0][0] + 1, axis=1)
    tracks = tracks.drop(columns='particle')
    return tracks


def missing_elements(elem):
    """This is how we determine gaps in tracks"""
    start, end = elem[0], elem[-1]
    return sorted(set(range(start, end + 1)).difference(elem))


def fill_gaps(tracks):
    """Fill any gaps identified in the tracks"""
    out = pd.DataFrame()
    num_tracks = max(tracks.track.values.astype(int))
    for xii in range(1, num_tracks):
        grouped = tracks.groupby(['track']).get_group(xii)
        frames = grouped.frame.values
        gaps = missing_elements(frames)
        for gap in reversed(gaps):
            before = grouped.iloc[np.where(frames < gap)[0][-1]]
            after = grouped.iloc[np.where(frames > gap)[0][0]]
            new_x = np.round((float(before.x) + float(after.x))/2, 3)
            new_y = np.round((float(before.y) + float(after.y))/2, 3)
            new_z = np.round((float(before.z) + float(after.z))/2, 3)
            a_row = pd.Series([new_x, new_y, new_z, int(gap), int(xii)],
                              index=['x', 'y', 'z', 'frame', 'track'],
                              name=str(gap))
            grouped = grouped.append(a_row)
        grouped.index.name = 'Frame'
        out = out.append(grouped.sort_values(by='frame'))
    return out


def pd_2_xml(tracks, save_file='tracked.cmm'):
    """Convert the tracked pandas dataframe into an xml chimerax opens"""
    xml = ['<marker_sets>']
    extras = 'r="1" g="1" b="0" radius="0.5"'
    for xii in range(1, max(tracks.track.values.astype(int)) + 1):
        grouped = tracks.groupby(['track']).get_group(xii)
        time = grouped['track'].values.astype(int)
        frame = grouped['frame'].values.astype(int)
        xval = grouped['x'].values
        yval = grouped['y'].values
        zval = grouped['z'].values
        xml.append('<marker_set name="track_{0}">'.format(xii))
        for i in range(len(time)):
            id_text = 'id="{0}" '.format(i + 1)
            xyz_text = 'x="{0}" y="{1}" z="{2}" '.format(
                xval[i], yval[i], zval[i])
            xml.append('<marker ' + id_text + xyz_text +
                       extras + ' frame="{0}" />'.format(frame[i]))
        xml.append('</marker_set>')
    xml.append('</marker_sets>')
    xml = '\n'.join(xml)
    file = open(save_file, 'w')
    file.write(xml)
    file.close()


def main(xml_file=None):
    """Main has 5 steps:
    1. Convert the input xml to dataframe
    2. Find any duplicate centroids
    3. Track the centroids and link together
    4. Identify and fill any gaps in the tracks
    5. Convert the tracks into a chimerax readable xml
    """
    if not xml_file:
        xml_file = choose_xml()
        if not xml_file:
            return

    file_path = os.path.split(xml_file)[0]
    save_file = os.path.join(file_path, "tracked_centroids.cmm")
    data_frame = xml_2_pd(xml_file)
    data_frame = remove_close_time(data_frame)
    tracks = track_centroids(data_frame)
    tracks = fill_gaps(tracks)
    pd_2_xml(tracks, save_file)


if __name__ == '__main__':
    main()
