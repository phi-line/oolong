import numpy as np
from json_tricks.np import dump, dumps, load, loads, strip_comments

# Loading songs from a path
import os
from librosa import load, beat, output

class Song:
    def __init__(self, name, path):
        '''
        This is the main Song class. It holds basic data about the song name, genre, and path.
        Each of the other attributes are set via the analyze_song() function and represent
        the sub-classes below.

        :param name: (string)         | song name
        :param path: (string)         | full path to the song
        :attr load: (Load)            | y and sr from Librosa's load()
        :attr genre: (string)         | genre for the song (manually set for training)
        :attr beat_track: (beatTrack) | beat track from Librosa's beat.beat_track()
        :attr segments: (dict)        | dictionary of the segments of a song and their bounds as tuples
        :attr slice: (Slice)          | variant of Load that takes a slice of a song at an offset and duration
        '''
        self.name = name
        self.path = path
        self.load = Load(path)
        self.genre = None
        self.beat_track = None
        self.segments = None
        self.slice = None

    def __json_encode__(self):
        return {'name': self.name, 'path': self.path, 'segments': self.segments,
                'slice': dumps(self.slice), 'beat_track': dumps(self.beat_track)}

    def __json_decode__(self, **attrs):
        self.name = attrs['name']
        self.path = attrs['path']
        self.load = Load(attrs['path'])
        self.genre = attrs['genre']
        self.segments = attrs['segments']
        self.slice = loads(attrs['slice'])
        self.beat_track = loads(attrs['slice'])

class Load:
    def __init__(self, path, **kwargs):
        '''
        Load takes in a path and calls Librosa's load.load_song() on it.

        :param path: (string)    | path to load
        :param kwargs: (arr)     | to pass arguments to the Slice sub class
        :attr y: (numpy.ndarray) | audio time series
        :attr sr: (int)          | sample rate
        '''
        y, sr = load_song(path, **kwargs)
        self.y = y
        self.sr = sr

    def __iter__(self):
        return iter([self.y, self.sr])

    def output_wav(self, folder, filename):
        '''
        Small function to output a Load or Slice to a wav file

        :param folder: (string)   | the folder to output to
        :param filename: (string) | filename to output as
        :return:
        '''
        audio = os.path.join(folder, filename)
        output.write_wav(audio, self.y, self.sr)

class Slice(Load):
    def __init__(self, path, offset=None, duration=None):
        '''
        This is a sub-class of Load. It 'records' a slice

        :param path: (string)       | path to load
        :param offset: (float)      | offset to start load (in seconds)
        :param duration: (float)    | duration to 'record' (in seconds)
        :attr: features: (Features) | class to store feature scatterplot
        '''
        super().__init__(path, offset=offset, duration=duration)
        self.offset = offset
        self.duration = duration
        self.features = None

    def __json_encode__(self):
        return {'y': self.y, 'sr': self.sr,
                'offset': self.offset, 'duration': self.duration, 'features': dumps(self.features)}

    def __json_decode__(self, **attrs):
        self.y = attrs['y']
        self.sr = attrs['sr']
        self.offset = attrs['offset']
        self.duration = attrs['duration']
        self.features = loads(attrs['features'])

class beatTrack():
    def __init__(self, y, sr):
        '''
        Generates a Librosa beat.beat_track() for the song.

        :param tempo: (float) | beats per minute
        :param beats: (list)  | list of beat frames
        '''
        tempo, beats = beat.beat_track(y=y, sr=sr, trim=False)
        self.tempo = tempo
        self.beats = beats

    def __iter__(self):
        return iter([self.tempo, self.beats])

    def __json_encode__(self):
        return {'tempo': self.tempo, 'beats': self.beats}

    def __json_decode__(self, **attrs):
        self.tempo = attrs['tempo']
        self.beats = attrs['beats']

def load_song(path, **kwargs):
    '''
    Small helper function to load song from path
    :param path: (string)    | path to load
    :param kwargs: (arr)     | to pass arguments to the Slice sub class
    :return:
    '''
    y, sr = load(path=path, **kwargs)
    return (y, sr)