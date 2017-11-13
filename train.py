from __future__ import print_function

import os
import argparse

from tinydb import TinyDB, Query
import json

from playsound import playsound

from self_similarity import segmentation, slicer
from song_classes import Song, Slice, beatTrack
from features import Features
from kernel_density import kde, kd_feature

import matplotlib.pyplot as plt

display_ = False
preview_ = False
verbose_ = True

supported_ext = ['.mp3', '.wav']
temp_dir = 'audio/_temp'

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
    parser.add_argument("-p", "--preview", help="play a preview of the slice while scanning",
                        action="store_true")
    # parser.add_argument("-v", "--verbose", help="output individual steps to console",
    #                     action="store_true")
    args = parser.parse_args()
    if not args.genre and args.dir:
        return

    global display_; display_ = args.display
    global preview_; preview_ = args.preview
    # global verbose_; verbose_ = args.verbose

    preview_ and os.makedirs(temp_dir, exist_ok=True)

    train(args.genre, args.dir)
    return

def train(genre, dir, n_beats=16):
    '''
    This is the main driver for now.
    This function takes a directory and scans it for all of its songs.
    It calls analyze upon to fill the song class structure with attributes.
    Upon storing all song data, it will calculate a Kernel Density Estimator for the combined scatterplots.

    :param genre: (string) | input genre to store alongside the song
    :param dir: (string)   | directory of the song folder
    :param n_beats: (int)  | number of beats to record for the slice
    :return: None
    '''
    mp3s = []
    target = os.path.abspath(dir)
    for root, subs, files in os.walk(target):
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext in supported_ext:
                strip = os.path.splitext(f)[0]
                mp3s.append((strip, os.path.join(target, f)))
    print('Loaded {} songs'.format(len(mp3s)))

    songs = []
    update = update_info(len(mp3s))
    fail_count = 0
    for m in mp3s:
        try:
            song = analyze_song(m, genre, n_beats, update)
            print(json.dumps(song.toJSON(), cls=ComplexEncoder))
        except IndexError:
            verbose_ and update.state('Failed!', end='\n')
            fail_count += 1

    stdout.write('\x1b[2K')
    print('Analyzed {} songs. Failed {} songs.'.format(len(songs) - fail_count, fail_count))
    clear_folder(temp_dir)

    #return the feature scatterplot from the slice to the main script to be stored alongside each
    for song in songs:
        verbose_ and update.state('Plotting')
        kde(song.slice.features)

    return

def analyze_song(mp3, genre, n_beats, update):
    '''
    This function takes a song and fills all of the nested class components found in song_classes.py
    It first takes song and loads it using Librosa. Then it segments the song, splitting it up into its major portions.
    Using the most prominent slice it will record a fixed n_beats portion of the song, using the bpm as a calculation
    for duration. Upon getting the slice it will gather it's features using CENSURE image recognition.

    :param mp3: (string)         | path to the song
    :param genre: (string)       | genre to store with the song
    :param n_beats: (int)        | n beat portion to record for the slice
    :param update: (update_info) | class hook to display the status of the function
    :return: song: (Song)        | filled song class returned to the driver
    '''
    update.next(mp3[0], status=(verbose_ and 'Loading'))
    song = Song(name=mp3[0], path=mp3[1])
    song.genre = genre

    verbose_ and update.state(status='Chunking')
    song.beat_track = beatTrack(y=song.load.y, sr=song.load.sr)

    # first send the batch to the trainer function to analyze song for it's major segments
    verbose_ and update.state(status='Segmenting')
    duration = (60 / song.beat_track.tempo) * n_beats  # beats per second
    song.segments = segmentation(song=song)

    # then take a N beat slice from the spectrogram that is from the most major segment
    verbose_ and update.state(status='Slicing')
    max_pair = slicer(song, duration)
    song.slice = Slice(song.path, offset=max_pair[0], duration=duration)
    preview_ and preview_slice(song)

    # gather the features from the slice
    verbose_ and update.state(status='Scanning')
    song.slice.features = Features(song.slice)
    return song

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

class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj,'toJSON'):
            return obj.toJSON()
        else:
            return json.JSONEncoder.default(self, obj)

from sys import stdout
class update_info(object):
    def __init__(self, steps):
        '''
        This is a helper class to provide a visual aid to the analyzing process.
        It takes in a total number of steps as an input and acts as a generator:
            '[n/100] songName | Status: Segmenting | BPM: 220'
        Everytime next is called, the step count will increase
        Next calls state() to display this info to the user but this function can also be called
        public to update the Status or info without the increment.

        :param steps: total number of steps to increment to
        '''
        self.steps = steps
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self, title, status='', info=''):
        '''
        Generator function that increments the n/total info
        Takes in a new title for the song and an optional status and info

        :param title: (string) name of the song
        :param status: (string) current stage of the song's loading process
        :param info: (tuple) Additional information to be displayed to the user.
        :return: None
        '''
        if self.n < self.steps:
            self.n += 1
            self.title = title
            self.state(status, info)
        else:
            raise StopIteration()

    def state(self, status='', info='', end='\r'):
        '''
        State displays information about the song's process to the user.
        Info and status are both optional as with next()
        It updates its info in line as to not take up much terminal space.
        This inline behaviour can be modified with the end character

        :param status: (string) current stage of the song's loading process
        :param info: (tuple) Additional information to be displayed to the user.
        :param end: The end character
        :return: None
        '''
        stdout.write('\x1b[2K')
        short_name = (self.title[:35] + ' ... ' + self.title[-5:]) if len(self.title) > 40 else self.title
        s = '| Status: {}'.format(status) if status else ''  # fight me
        i = '| {}: {}'.format(*info) if info else ''
        stdout.write('[{}/{}] {} {} {}{}'.format(self.n, self.steps, short_name, s, i, end))

def preview_slice(song):
    '''
    This function outputs a song to a _temp directory in wav format and plays it.

    :param song: (string) The filepath of the song
    :return: None
    '''
    try:
        clear_folder(temp_dir)
        path = temp_dir
        filename = (song.name + '.wav').replace(' ', '-')
        song.slice.output_wav(path, filename)
        full_path = os.path.join(path, filename)
        playsound("{}".format(full_path), block=False)
    except OSError:
        return

def clear_folder(d):
    '''
    This is a helper function to preview_slice to clear the _temp folder before storage.

    :param d: (string) directory to clear
    :return: None
    '''
    list(map(os.unlink, (os.path.join(d, f) for f in os.listdir(d))))

if __name__ == '__main__':
    main()