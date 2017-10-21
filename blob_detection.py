import matplotlib.pyplot as plt

import numpy as np
import librosa
import librosa.display
from scipy.misc import toimage
from skimage import io

def main():
    y, sr = librosa.load(path='Song.mp3', offset=108, duration=1.8333)#duration=8.8)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    slice = librosa.logamplitude(S, ref_power=np.max)

    y, sr = librosa.load(path='Song.mp3')
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    original = librosa.logamplitude(S, ref_power=np.max)

    image_manip(slice, original)
    # display_spec(a, sr)

def image_manip(slice, original):
    from skimage import data
    from skimage import transform as tf
    from skimage.feature import CENSURE
    from skimage.color import rgb2gray

    import matplotlib.pyplot as plt

    img_orig = rgb2gray(slice)
    img_warp = rgb2gray(original)

    detector = CENSURE()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    detector.detect(img_orig)

    ax[0].imshow(img_orig, cmap=plt.cm.gray)
    ax[0].scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
                  2 ** detector.scales, facecolors='none', edgecolors='r')
    ax[0].set_title("Sliced Image")

    detector.detect(img_warp)

    ax[1].imshow(img_warp, cmap=plt.cm.gray)
    ax[1].scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
                  2 ** detector.scales, facecolors='none', edgecolors='r')
    ax[1].set_title('Original Image')

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()


def display_spec(log_S, sr):
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.tight_layout()
    plt.show()

def normalize(v):
    norm=np.linalg.norm(v, ord=2)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def imsave(name, arr):
    im = toimage(arr)
    im.save(name)
    return name

    # y, sr = librosa.load(path='Song.mp3')
    # S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    # img2 = librosa.logamplitude(S, ref_power=np.max)

if __name__ == '__main__':
    main()