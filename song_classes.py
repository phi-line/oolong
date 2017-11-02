from librosa import load

class Song:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.load = Load(*load_song(path))
        self.slice = None
        self.bpm = None

class Load:
    def __init__(self, y, sr):
        self.y = y
        self.sr = sr

class Slice:
    def __init__(self, y, sr, onset=None, duration=None):
        self.load = Load(y, sr)
        self.onset = onset
        self.duration = duration

def load_song(path):
    y, sr = load(path=path)
    return (y, sr)