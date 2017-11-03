from librosa import load, beat

class Song:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.load = Load(path)
        self.beat_track = None
        self.segments = []
        self.slice = None

class Load:
    def __init__(self, path, **kwargs):
        y, sr = load_song(path, **kwargs)
        self.y = y
        self.sr = sr

class Slice(Load):
    def __init__(self, path, offset=None, duration=None):
        Load.__init__(self, path, offset=offset, duration=duration)
        self.offset = offset
        self.duration = duration

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