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
    if len(sys.argv) < 3:
        print ('Error: invalid number of arguments')
        print ('Usage: spectrogram.py type song.mp3')
        print ('Types: [mel] ')
        sys.exit()
    else:
        if sys.argv[1] == 'mel':
            mel_spectrogram(sys.argv[2], display=True)


def mel_spectrogram(mp3 = sys.argv[2], display = True):
    '''
    this function displays a mel spectrogram .csv of the mp3 data
    :return:
    '''
    if len(sys.argv) >= 1:
        y, sr = librosa.load(path=mp3)

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
        plt.show()

if __name__ == '__main__':
    main()
