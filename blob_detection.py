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
    path = 'smbu.mp3'

    y, sr = librosa.load(path=path, offset=109.12, duration=3.529)
    y = librosa.effects.percussive(y)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=256)
    mel_slice = librosa.logamplitude(S, ref_power=np.min)
    # display_spec(mel_slice, sr)

    librosa.output.write_wav('slice.wav', y, sr)
    filename = os.path.join(os.getcwd(), 'slice.png')
    slice = imsave(filename, rgb2gray(mel_slice))


    y, sr = librosa.load(path=path, offset=109.12, duration=3.529*4)
    y = librosa.effects.percussive(y)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=256)
    mel_original = librosa.logamplitude(S, ref_power=np.min)
    # display_spec(mel_original, sr)

    librosa.output.write_wav('original.wav', y, sr)
    filename = os.path.join(os.getcwd(), 'original.png')
    original = imsave(filename, rgb2gray(mel_original))

    feat_censure(mel_slice, mel_original)
    # brute(slice, original)
    flann(slice, original)

def feat_censure(slice, original):
    from skimage.feature import CENSURE
    import matplotlib.pyplot as plt

    detector = CENSURE()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    detector.detect(slice)

    ax[0].imshow(slice, cmap=plt.cm.gray)
    ax[0].scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
                  2 ** detector.scales, facecolors='none', edgecolors='r')
    ax[0].set_title("Sliced Image")

    detector.detect(original)

    ax[1].imshow(original, cmap=plt.cm.gray)
    ax[1].scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
                  2 ** detector.scales, facecolors='none', edgecolors='r')
    ax[1].set_title('Original Image')

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

def brute(slice, original):
    import cv2
    from matplotlib import pyplot as plt

    img1 = cv2.cvtColor(cv2.imread(slice), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.imread(original), cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None)

    plt.imshow(img3), plt.show()

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