from __future__ import print_function

import os
import argparse

from tinydb import TinyDB
from datetime import datetime
import json_tricks.np as jt

from song_classes import Song, Slice, beatTrack, Features
from src.self_similarity import segmentation, slicer
from src.kernel_density import kde

from sys import stdout
from playsound import playsound

load_dir_ = ''
ldb_dir_ = ''
preview_ = False
verbose_ = True

supported_ext = ['.mp3', '.wav']
db_root = 'db/'
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
    parser.add_argument("-l", "--load", help="(path) from root to folder containing song files to analyze",
                        metavar="[load dir]", action=readable_dir, dest="load_dir")
    parser.add_argument("-t", "--train", help="load pre-analyzed songs from database",
                        metavar="[train dir]", action=readable_file, dest="ldb_dir")
    parser.add_argument("-p", "--preview", help="play a preview of the slice while scanning",
                        action="store_true")
    args = parser.parse_args()
    if not args.genre and (args.load or args.train):
        return

    global load_dir_; load_dir_ = args.load_dir
    global ldb_dir_; ldb_dir_ = args.ldb_dir
    global preview_; preview_ = args.preview

    load_dir_ and preview_ and os.makedirs(temp_dir, exist_ok=True)

    genre_db_dir = os.path.join(db_root, args.genre)
    if not os.path.exists(genre_db_dir):
        os.makedirs(genre_db_dir, exist_ok=True)

    load_dir_ and load(args.genre, args.load_dir)
    ldb_dir_ and train(args.genre, args.ldb_dir)
    return

def load(genre, load_dir, n_beats=16):
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

    name = os.path.basename(os.path.normpath(load_dir))
    db = TinyDB(os.path.join(db_root, genre, ''.join(name + '-' + str(datetime.now())) + '.json'))

    target = os.path.abspath(load_dir)
    for root, subs, files in os.walk(target):
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext in supported_ext:
                strip = os.path.splitext(f)[0]
                mp3s.append((strip, os.path.join(target, f)))
    print('Loaded {} songs'.format(len(mp3s)))

    update = update_info(len(mp3s))
    succ_count = 0
    fail_count = 0
    for m in mp3s:
        try:
            song = analyze_song(m, genre, n_beats, update)
            json = {'{}'.format(succ_count): jt.dumps(song)}
            db.insert(json)
            succ_count += 1
        except IndexError or TypeError:
            verbose_ and update.state('Failed!', end='\n')
            fail_count += 1

    stdout.write('\x1b[2K')
    print('Analyzed {} songs. Failed {} songs.'.format(succ_count - fail_count, fail_count))
    clear_folder(temp_dir)
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
    song.features = Features(song.slice)
    return song

def train(genre, json, n_beats=16):
    db = TinyDB(json)

    features = []
    l = len(db)
    printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=50)
    for i, item in enumerate(db):
        song = jt.loads(item[str(i)], cls_lookup_map=globals())
        features.append(song.features)
        printProgressBar(i + 1, l, prefix='Progress:', suffix='Complete', length=50)

    # return the feature scatterplot from the slice to the main script to be stored alongside each
    for feature in features:
        # print(feature.kp.shape)
        kde(feature)

class readable_dir(argparse.Action):
    '''
    This class validates a given directory and raises an exception if the directory is invalid.
    Taken from: https://stackoverflow.com/a/11415816
    '''
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir=values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError('readable_dir:{0} is not a valid path'.format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace,self.dest,prospective_dir)
        else:
            raise argparse.ArgumentTypeError('readable_dir:{0} is not a readable dir'.format(prospective_dir))

class readable_file(argparse.Action):
    '''
    This class validates a given directory and raises an exception if the directory is invalid.
    '''
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_path=values
        if not os.path.exists(prospective_path):
            raise argparse.ArgumentTypeError('readable_dir:{0} is not a valid path'.format(prospective_path))
        if os.access(prospective_path, os.R_OK):
            setattr(namespace,self.dest,prospective_path)
        else:
            raise argparse.ArgumentTypeError('readable_dir:{0} is not a valid path'.format(prospective_path))

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

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 2, length = 100, fill = '█'):
    '''
    Prints a progression bar
    Stolen from: https://stackoverflow.com/a/34325723

    :param iteration: (int) | current iteration
    :param total: (int)     | total iterations (Int)
    :param prefix: (string) | prefix string (Str)
    :param suffix: (string) | suffix string (Str)
    :param decimals: (int)  | positive number of decimals in percent complete (Int)
    :param length: (int)    | character length of bar (Int)
    :param fill: (string)   | bar fill character (Str)
    :return: None
    '''
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

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