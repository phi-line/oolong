import matplotlib.pyplot as plt

import os
import numpy as np
import librosa
import librosa.display

# Gathering features from slice
from librosa import effects, feature, logamplitude
from skimage.feature import CENSURE

class Features:
    def __init__(self, slice):
        detector, kp = self.getFeatures(slice)
        self.detector = detector
        self.kp = kp

    @staticmethod
    def getFeatures(slice, mels=256):
        y, sr = tuple(slice)
        # y = effects.percussive(y)
        S = feature.melspectrogram(y, sr=sr, n_mels=mels)
        log_S = logamplitude(S, ref_power=np.max)

        detector = CENSURE()
        detector.detect(log_S)
        kp = detector.keypoints
        return detector, kp

def feat_censure(slice):
    from skimage.feature import CENSURE

    y, sr = tuple(slice)
    y = librosa.effects.percussive(y)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=256)
    log_S = librosa.logamplitude(S, ref_power=np.max)

    detector = CENSURE()

    detector.detect(log_S)
    kp = detector.keypoints
    return kp

def display_spec(S, sr):
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')
    plt.tight_layout()
    plt.show()