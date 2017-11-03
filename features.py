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
        y = effects.percussive(y)
        S = feature.melspectrogram(y, sr=sr, n_mels=mels)
        log_S = logamplitude(S, ref_power=np.max)
        detector = CENSURE()
        detector.detect(log_S)
        kp = detector.keypoints
        return detector, kp

def feat_censure(slice):
    from skimage.feature import CENSURE
    import matplotlib.pyplot as plt

    y, sr = tuple(slice)
    y = librosa.effects.percussive(y)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=256)
    log_S = librosa.logamplitude(S, ref_power=np.max)

    detector = CENSURE()

    detector.detect(log_S)
    kp = detector.keypoints
    xx, yy, zz = kd_feature(kp, 5.0, metric='manhattan')

    plt.pcolormesh(xx, yy, zz)#, cmap=plt.cm.gist_heat)
    plt.scatter(x=kp[:, 1], y=kp[:, 0], s=2 ** detector.scales, facecolor='white', alpha=.5)
    plt.axis('off')
    plt.show()

def kd_feature(scatter, bandwidth, xbins=100j, ybins=100j, **kwargs):
    """Build 2D kernel density estimate (KDE)."""
    from sklearn.neighbors import KernelDensity

    x = scatter[:, 1]; y = scatter[:, 0]

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins,
             y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)


def display_spec(S, sr):
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')
    plt.tight_layout()
    plt.show()

def main():
    song_folder = os.path.join(os.getcwd(), 'audio/')
    song_path = os.path.join(song_folder, 'smack_my_b.mp3')

    y, sr = librosa.load(path=song_path, offset=109.12, duration=3.529*2)
    y = librosa.effects.percussive(y)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=256)
    mel_slice = librosa.logamplitude(S, ref_power=np.max)
    display_spec(mel_slice, sr)

    feat_censure(mel_slice)

if __name__ == '__main__':
    main()