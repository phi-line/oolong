from __future__ import print_function

import sys #kwargs

import numpy as np
import scipy as sp
# import keras only if we need DNN
import matplotlib.pyplot as plt

# Librosa is a music analysis tool
# https://github.com/librosa/librosa
import librosa
import librosa.display

def main():
    # Track beats using time series input

    song = 'song.mp3'
    y, sr = librosa.load(song)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)


    onset_env = librosa.onset.onset_strength(y, sr=sr,
                                             aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env,
                                           sr=sr)

    onset_max = np.argmax(onset_env)

    starting_beat = find_nearest(beats, onset_max)
    print(starting_beat)
    range = (beats[starting_beat], beats[starting_beat + 8])

    # start_time = librosa.frames_to_time(range[0], sr=sr)
    # end_time = librosa.frames_to_time(range[1], sr=sr)
    # print(start_time, end_time)
    # mel_spectrogram(mp3=song, start_time=start_time, end_time=end_time)

    play_by_seconds(song,108,120)
    print(librosa.time_to_frames(108, sr=sr))
    print(librosa.time_to_frames(120, sr=sr))


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx #array[idx]

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


if __name__ == '__main__':
    main()
