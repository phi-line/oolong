import numpy as np

# Loading songs from a path
import os
from librosa import load, beat, output

class Song:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.load = Load(path)
        self.genre = None
        self.beat_track = None
        self.segments = []
        self.slice = None

class Load:
    def __init__(self, path, **kwargs):
        y, sr = load_song(path, **kwargs)
        self.y = y
        self.sr = sr

    def __iter__(self):
        return iter([self.y, self.sr])

    def output_wav(self, folder, filename):
        audio = os.path.join(folder, filename)
        output.write_wav(audio, self.y, self.sr)

class Slice(Load):
    def __init__(self, path, offset=None, duration=None):
        super().__init__(path, offset=offset, duration=duration)
        # Load.__init__(self, path, offset=offset, duration=duration)
        self.offset = offset
        self.duration = duration
        self.features = None

class beatTrack():
    def __init__(self, y, sr):
        tempo, beats = beat.beat_track(y=y, sr=sr, trim=False)
        self.tempo = tempo
        self.beats = beats

    def __iter__(self):
        return iter([self.tempo, self.beats])

def load_song(path, **kwargs):
    y, sr = load(path=path, **kwargs)
    return (y, sr)