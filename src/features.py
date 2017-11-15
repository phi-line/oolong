import numpy as np

# Gathering features from slice
from librosa import effects, feature, logamplitude
from skimage.feature import CENSURE

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