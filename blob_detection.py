import matplotlib.pyplot as plt

import numpy as np
import librosa
import librosa.display
from scipy.misc import toimage
from skimage import io
from skimage.color import rgb2gray


# y, sr = librosa.load(path=path, offset=109, duration=8.7272)#duration=8.8)
# S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
# slice = librosa.logamplitude(S, ref_power=np.max)
# librosa.output.write_wav('slice.wav', y, sr)

def main():
    path = 'smbu.mp3'

    y, sr = librosa.load(path=path, offset=109.12, duration=3.529)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    slice = librosa.logamplitude(S, ref_power=np.min)
    librosa.output.write_wav('slice.wav', y, sr)
    display_spec(slice, sr)

    y, sr = librosa.load(path=path, offset=109.12, duration=3.529*4)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    original = librosa.logamplitude(S, ref_power=np.min)
    librosa.output.write_wav('slice.wav', y, sr)
    # display_spec(original, sr)

    imsave("slice.jpg", rgb2gray(slice))
    imsave("original.jpg", rgb2gray(slice))

    image_manip(slice, original)
    # flann(slice, original)

def image_manip(slice, original):
    from skimage.feature import CENSURE

    import matplotlib.pyplot as plt

    img_orig = slice
    img_warp = original

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

def flann(slice, original):
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt

    img1 = cv2.imread('slice.', 0)  # queryImage
    img2 = cv2.imread('original', 0)  # trainImage

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
    matchesMask = [[0, 0] for i in xrange(len(matches))]

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

def play_by_seconds(y, sr):
    y, sr = librosa.load(path=mp3, offset=start_time, duration=end_time - start_time)
    librosa.output.write_wav('slice.wav', y, sr)

if __name__ == '__main__':
    main()