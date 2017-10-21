from __future__ import print_function

import sys #kwargs

import numpy as np
import scipy as scipy
import scipy as sp
# import keras only if we need DNN
import matplotlib.pyplot as plt

# Librosa is a music analysis tool
# https://github.com/librosa/librosa
import librosa
import librosa.display
import os
from collections import defaultdict

def main():
    # Track beats using time series input
    song = 'songs/song.mp3'

    song2 = 'songs/song4.mp3'

    # plt.plot(song1pattern)
    # plt.ylabel("Frames between beats")
    # plt.xlabel("Frames")
    #
    # plt.plot(song2pattern)
    # plt.ylabel("Frames between beats")
    # plt.xlabel("Frames")
    # plt.show()

    findPattern(song,10)
    findPattern(song2,10)


def mel_spectrogram(mp3 = "", display = True, start_time=0, end_time=0):
    '''
    this function displays a mel spectrogram .csv of the mp3 data
    :return:
    '''
    if len(sys.argv) >= 1:
        y, sr = librosa.load(path=mp3, offset=start_time, duration=end_time-start_time)

    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.logamplitude(S, ref_power=np.max)

    # Make a new figure
    plt.figure(figsize=(12,4))

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

    # Put a descriptive title on the plot
    plt.title('mel power spectrogram')

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    # Make the figure layout compact
    plt.tight_layout()

    # display
    if display:
        librosa.output.write_wav('slice.wav', y, sr)
        plt.show()


def play_by_frame(mp3, sr, start, end):
    name = os.path.splitext(mp3)[0]
    start_time = librosa.frames_to_time(start, sr=sr)
    end_time = librosa.frames_to_time(end, sr=sr)
    y, sr = librosa.load(path=mp3, offset=start_time, duration=end_time - start_time)
    librosa.output.write_wav(name+'_slice.wav', y, sr)

def play_by_seconds(mp3, start_time, end_time):
    name = os.path.splitext(mp3)[0]
    y, sr = librosa.load(path=mp3, offset=start_time, duration=end_time - start_time)
    librosa.output.write_wav(name+'_slice.wav', y, sr)

def combinations_from_array(array, comboNum):
    comboArray = []
    for i in range(len(array)-comboNum):
        arraycombo = []
        for k in range(comboNum):
            arraycombo.append(array[k+i])
        comboArray.append(arraycombo)
    return comboArray

def findCombo(comboArray,comboNum):
    comboDict = dict()
    for index in range(len(comboArray)):
        comboDict[index] = 0
    copyArray = comboArray
    length = len(copyArray)
    index = 0
    comboIndex = 1
    while index < length - comboNum:
        while comboIndex < length:
            if copyArray[index] == copyArray[comboIndex]:
                del copyArray[comboIndex]
                comboDict[index] += 1
                length = len(comboArray)
            comboIndex += 1
        index += 1
        comboIndex = index + 1
    return comboDict

def findPattern(song,combo):
    y, sr = librosa.load(song)
    onset_env = librosa.onset.onset_strength(y, sr=sr,
                                              aggregate=np.median)
    print("loaded onset")
    songArray = []

    numzeros = 0
    for i in onset_env:
        if i < 0.5:
            numzeros += 1
        else:
            songArray.append(numzeros)
            numzeros = 0

    songcombo = combinations_from_array(songArray, combo)
    comboDict = findCombo(songcombo, combo)
    print(songcombo)
    print(comboDict)
    max_value = max(comboDict.values())  # maximum value
    max_keys = [k for k, v in comboDict.items() if v == max_value]
    print(max_value, songcombo[max_keys[0]])
    patternIndex = max_keys[0] * combo  # pattern starts here
    songpattern = songArray[0:patternIndex + combo + 1]
    print(sum(songpattern))
    print(sum(songcombo[max_keys[0]]))

    play_by_frame(song, sr, sum(songpattern), sum(songpattern) + (sum(songcombo[max_keys[0]]) + 1) * combo * 2)
    print(songArray)

if __name__ == '__main__':
    main()
