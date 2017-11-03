from __future__ import print_function

import os
import argparse

from self_similarity import segmentation
from song_classes import Song, Slice, beatTrack
from features import Features

import matplotlib.pyplot as plt

display_ = False
verbose_ = False

def main():
    '''
    usage: train.py house Playlist/

    This is a script that trains a Kernel Density Estimator based on song features

    positional arguments:
      genre       (string) classifier to train a model for
      [folder]    (path) song files to analyze (grouped by genre)

    optional arguments:
      -h, --help     show this help message and exit
      -d, --display  display matplotlib graphs
      -v, --verbose  output individual steps to console
    '''
    parser = argparse.ArgumentParser(description="This is a script that trains a Kernel Density Estimator based on song features",
                                     usage='%(prog)s house HousePlaylist/')
    parser.add_argument(dest="genre",
                        help="(string) classifier to train a model for")
    parser.add_argument(dest="dir",
                        help="(path) from root to folder containing song files to analyze", metavar="[dir]",
                        action=readable_dir)
    parser.add_argument("-d", "--display", help="display matplotlib graphs",
                        action="store_true")
    parser.add_argument("-v", "--verbose", help="output individual steps to console",
                        action="store_true")
    args = parser.parse_args()
    if not args.genre and args.dir:
        return

    global display_; display_ = args.display
    global verbose_; verbose_ = args.verbose

    train_kde(args.genre, args.dir)
    return

def train_kde(genre, dir, n_beats=4):
    mp3s = []
    target = os.path.abspath(dir)
    for root, subs, files in os.walk(target):
        for f in files:
            if os.path.splitext(f)[1] == '.mp3':
                mp3s.append((f, os.path.join(target, f)))
    print('Loaded {} songs'.format(len(mp3s)))

    songs = []
    update = update_info(len(mp3s))
    fail_count = 0
    for m in mp3s:
        update.next(m[0], (verbose_ and 'Loading'))
        song = Song(m[0], m[1])
        songs.append(song)

        verbose_ and update.state('Chunking')
        song.beat_track = beatTrack(y=song.load.y, sr=song.load.sr)

        # first send the batch to the trainer function to analyze song for it's major segments
        verbose_ and update.state('Segmenting')
        duration = (song.beat_track.tempo / 60) * n_beats
        max_pair = segment(song, duration)

        if all(p == 0 for p in max_pair):
            fail_count += 1
            continue

        # then take a N beat slice from the spectrogram that is from the most major segment
        verbose_ and update.state('Slicing')
        song.slice = Slice(song.path, offset=max_pair[0], duration=duration)

        # gather the features from the slice
        verbose_ and update.state('Scanning')
        song.slice.features = Features(song.slice)

        verbose_ and update.state('Plotting')
        kde(song.slice.features)

    stdout.write('\x1b[2K')
    print('Analyzed {} songs. Failed {} songs.'.format(len(songs) - fail_count, fail_count))


    #return the feature scatterplot from the slice to the main script to be stored alongside each
    return

def segment(song, duration):
    song.segments = segmentation(song=song, display=display_)

    max_pair = (0, 0)
    for k, dk in song.segments.items():
        for pair in dk:
            diff = pair[1] - pair[0]
            max_diff = max_pair[1] - max_pair[0]
            if (diff >= duration) & (diff > max_diff):
                max_pair = pair

    return max_pair

from features import kd_feature
def kde(features, bandwidth=5.0):
    kp = features.kp
    detector = features.detector

    xx, yy, zz = kd_feature(kp, bandwidth, metric='manhattan')

    plt.pcolormesh(xx, yy, zz)  # , cmap=plt.cm.gist_heat)
    plt.scatter(x=kp[:, 1], y=kp[:, 0], s=2 ** detector.scales, facecolor='white', alpha=.5)
    plt.axis('off')
    plt.show()

class readable_dir(argparse.Action):
    '''
    This class validates a given directory and raises an exception if the directory is invalid.
    Taken from: https://stackoverflow.com/a/11415816
    '''
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir=values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace,self.dest,prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))

from sys import stdout
class update_info(object):
    def __init__(self, steps):
        self.steps = steps
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self, name, status=''):
        if self.n < self.steps:
            self.n += 1
            self.name = name
            self.state(status)
        else:
            raise StopIteration()

    def state(self, status):
        stdout.write('\x1b[2K')
        s = '| Status: {}'.format(status) if status else ''  # fight me
        stdout.write('[{}/{}] {} {}\r'.format(self.n, self.steps, self.name, s))

if __name__ == '__main__':
    main()