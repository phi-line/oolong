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
import pandas as pd

def main():
    # Track beats using time series input
    song = 'song.mp3'
    y1, sr1 = librosa.load(song)
    onset_env1 = librosa.onset.onset_strength(y1, sr=sr1,
                                             aggregate=np.median)
    tempo1, beats1 = librosa.beat.beat_track(y=y1, sr=sr1)

    song2 = 'song3.mp3'
    y2, sr2 = librosa.load(song2)
    tempo2, beats2 = librosa.beat.beat_track(y=y2, sr=sr2)
    onset_env2 = librosa.onset.onset_strength(y2, sr=sr2,
                                             aggregate=np.median)
    song1pattern = []

    numzeros = 0
    for i in onset_env1:
        if i < 0.5:
            numzeros += 1
        else:
            song1pattern.append(numzeros)
            numzeros = 0

    np.savetxt("pattern1.csv", song1pattern, delimiter=",")

    song2pattern = []

    numzeros = 0
    for i in onset_env2:
        if i < 0.5:
            numzeros += 1
        else:
            song2pattern.append(numzeros)
            numzeros = 0


    # plt.plot(song1pattern)
    # plt.ylabel("Frames between beats")
    # plt.xlabel("Frames")
    #
    # plt.plot(song2pattern)
    # plt.ylabel("Frames between beats")
    # plt.xlabel("Frames")
    # plt.show()

    print(combinations_from_array(song1pattern, 3))
    print(check_combo(song1pattern,combinations_from_array(song1pattern, 3)))


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
    start_time = librosa.frames_to_time(start, sr=sr)
    end_time = librosa.frames_to_time(end, sr=sr)
    y, sr = librosa.load(path=mp3, offset=start_time, duration=end_time - start_time)
    librosa.output.write_wav('slice.wav', y, sr)

def play_by_seconds(mp3, start_time, end_time):
    y, sr = librosa.load(path=mp3, offset=start_time, duration=end_time - start_time)
    librosa.output.write_wav('slice.wav', y, sr)

def combinations_from_array(array, combo):
    comboArray = []
    for i in range(len(array)-combo):
        arraycombo = []

        for k in range(combo):
            arraycombo.append(array[k+i])
        comboArray.append(arraycombo)
    return comboArray

def check_combo(originalArray,comparisonArray):
    matchesArray = []
    for combo in comparisonArray:
        matches = 0
        if comparisonArray[]:
        matchesArray.append()
    return matchesArray
if __name__ == '__main__':
    main()
