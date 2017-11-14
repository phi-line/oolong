import matplotlib.pyplot as plt

import os
import numpy as np
import librosa
import librosa.display

# Gathering features from slice
from librosa import effects, feature, logamplitude
from skimage.feature import CENSURE

from json_tricks.np import dump, dumps, load, loads, strip_comments

class Features:
    def __init__(self, slice):
        '''
        Features takes a slice and uses CENSURE image detection to create a scatterplot of the notable features.
        Read more here: https://link.springer.com/chapter/10.1007/978-3-540-88693-8_8

        :param slice: (Slice)     | slice to find features for
        :attr detector: (CENSURE) | CENSURE image detector
        :attr kp: (numpy.ndarray) | feature scatterplot
        '''
        detector, kp = self.getFeatures(slice)
        self.detector = detector
        self.kp = kp

    def __json_encode__(self):
        return {'kp': self.kp}

    def __json_decode__(self, **attrs):
        self.detector = CENSURE()
        self.kp = attrs['kp']

    @staticmethod
    def getFeatures(slice, mels=256):
        '''
        Helper function to get the features of a song. It first computes the spectrogram and performs the CENSURE
        image algorithm on it to return the features scatterplot.

        :param slice: (Slice)        | slice to find features for
        :param mels: (int)           | 'resolution' of the spectrogram
        :return: detector: (CENSURE) | CENSURE image detector
        :return: kp: (numpy.ndarray) | feature scatterplot
        '''
        y, sr = tuple(slice)
        # y = effects.percussive(y)
        S = feature.melspectrogram(y, sr=sr, n_mels=mels)
        log_S = logamplitude(S, ref_power=np.max)

        detector = CENSURE()
        detector.detect(log_S)
        kp = detector.keypoints
        return detector, kp

def display_spec(S, sr):
    '''
    Displays a spectrogram back to the user
    :param S: (numpy.ndarray) | spectrogram
    :param sr: (int)          | sample rate
    :return: None
    '''
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')
    plt.tight_layout()
    plt.show()