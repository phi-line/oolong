import matplotlib.pyplot as plt

import os
import numpy as np
import librosa
import librosa.display
from scipy.misc import toimage
from skimage.color import rgb2gray


# y, sr = librosa.load(path=path, offset=109, duration=8.7272)#duration=8.8)
# S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
# slice = librosa.logamplitude(S, ref_power=np.max)
# librosa.output.write_wav('slice.wav', y, sr)

def main():
    song_folder = os.path.join(os.getcwd(), 'audio/')
    song_path = os.path.join(song_folder, 'Song.mp3')

    y, sr = librosa.load(path=song_path, offset=109.12, duration=3.529*2)
    y = librosa.effects.percussive(y)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=256)
    mel_slice = librosa.logamplitude(S, ref_power=np.min)
    display_spec(mel_slice, sr)

    audio = os.path.join(song_folder, 'slice.wav')
    librosa.output.write_wav(audio, y, sr)
    image = os.path.join(song_folder, 'slice.png')
    slice = imsave(image, rgb2gray(mel_slice))


    y, sr = librosa.load(path=song_path, offset=109.12, duration=3.529*4)
    y = librosa.effects.percussive(y)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=256)
    mel_original = librosa.logamplitude(S, ref_power=np.min)
    # display_spec(mel_original, sr)

    audio = os.path.join(song_folder, 'original.wav')
    librosa.output.write_wav(audio, y, sr)
    image = os.path.join(song_folder, 'original.png')
    original = imsave(image, rgb2gray(mel_original))

    feat_censure(mel_slice, mel_original)
    # brute(slice, original)
    # flann(slice, original)

def feat_censure(slice, original):
    from skimage.feature import CENSURE
    from sklearn.neighbors import KernelDensity
    import matplotlib.pyplot as plt

    detector = CENSURE()

    detector.detect(slice)
    kp = detector.keypoints
    # hist_feature(detector.keypoints, slice)
    xx, yy, zz = kd_feature(kp, 4.0, metric='manhattan')

    plt.pcolormesh(xx, yy, zz, cmap=plt.cm.gist_heat)
    plt.scatter(x=kp[:, 1], y=kp[:, 0], s=2 ** detector.scales, facecolor='white', alpha=.5)
    plt.axis('off')
    plt.show()

def flann(slice, original):
    import cv2
    from matplotlib import pyplot as plt

    img1 = cv2.cvtColor(cv2.imread(slice), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.imread(original), cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    plt.imshow(img3, ), plt.show()

def hist_feature(scatter, slice):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    x = scatter[:, 1]; y = scatter[:, 0]

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    plt.scatter(x, y, c=z)
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

    # librosa.display.specshow(librosa.logamplitude(S, ref=np.max), y_axis='log')
    # plt.tight_layout()
    # plt.show()

def imsave(name, arr):
    im = toimage(arr)
    im.save(name)
    return name

    # y, sr = librosa.load(path='Song.mp3')
    # S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    # img2 = librosa.logamplitude(S, ref_power=np.max)

def play_by_seconds(y, sr):
    y, sr = librosa.load(path=mp3, offset=start_time, duration=end_time - start_time)
    librosa.output.write_wav('slice.wav', y, sr)

if __name__ == '__main__':
    main()