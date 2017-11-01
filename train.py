from __future__ import print_function

import os
import argparse
import string

def main():
    '''
    usage: train.py house HousePlaylist/

    This is a script that trains a Kernel Density Estimator based on song features

    positional arguments:
      genre       (string) classifier to train a model for
      [folder]    (path) song files to analyze (grouped by genre)

    optional arguments:
      -h, --help  show this help message and exit
    '''
    parser = argparse.ArgumentParser(description="This is a script that trains a Kernel Density Estimator based on song features",
                                     usage='%(prog)s house HousePlaylist/')
    parser.add_argument(dest="genre",
                        help="(string) classifier to train a model for")
    parser.add_argument(dest="dir",
                        help="(path) from root to folder containing song files to analyze", metavar="[dir]",
                        action=readable_dir)
    args = parser.parse_args()
    if not args.genre and args.dir:
        return

    train_kde(args.genre, args.dir)
    return

from self_similarity import segmentation

def train_kde(genre, dir):
    songs = []
    target = os.path.abspath(dir)
    for root, subs, files in os.walk(target):
        for f in files:
            if os.path.splitext(f)[1] == '.mp3':
                songs.append((f, os.path.join(target, f)))
    print('Loaded {} songs'.format(len(songs)))

    #first send the batch to the trainer function to analyze song for it's major segments
    for song in songs:
        print('Song: [{}] | Status: Segmenting'.format(song[0]), end='\r')
        segments = segmentation(path=song[1])

    #then take a N beat slice from the spectrogram that is from the most major segment

    #return the feature scatterplot from the slice to the main script to be stored alongside each
    return

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


if __name__ == '__main__':
    main()