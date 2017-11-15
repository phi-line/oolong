from __future__ import print_function
import librosa
import librosa.display

from song_classes import Slice, beatTrack

def slicer(song, n_beats=16, duration=0):
    '''
    Takes in a song and its segments and computes the largest total segment in the dictionary.
    To do this it sums up each of the dictionary entries using that disgusting(tm) comprehension below.
    The segment has to be larger than the given duration in order to be considered in the sum.
    It then takes the max dictionary entry and returns the segment with the bounds.

    :param song: (Song)       | song to slice
    :param duration: (float)  | min duration (in seconds)
    :return: slice (Slice)    | segmented slice
    '''
    largest_seg = max(song.segments.items(), key=lambda x: sum([z[1]-z[0] for z in x[1] if z[1]-z[0] >= duration]))[1]
    max_pair = tuple(max(largest_seg, key=lambda pair: pair[1]-pair[0]))

    slice = Slice(song.path, offset=max_pair[0], duration=max_pair[1])

    perc_y = librosa.effects.percussive(slice.y)
    beat_track = beatTrack(y=perc_y, sr=song.load.sr)

    end_frame = librosa.frames_to_samples(beat_track.beats[n_beats])[0]
    slice.y = slice.y[:end_frame]

    return slice

def segmentation(song, display=False):
    '''
    Takes in a song and then returns a class containing the spectrogram, bpm, and major segments
    It also fills the song's beatTrack and uses it in the segmentation algorithm.
    Algorithm written by: Brian McFee https://bmcfee.github.io/

    :param song: (Song)      | song to segment
    :param display: (bool)   | optional argument to display graph of segments using matPlotLib
    :return: seg_dict (dict) | dictionary of segments
    '''
    import numpy as np
    import scipy
    import matplotlib.pyplot as plt
    import sklearn.cluster

    y = song.load.y
    sr = song.load.sr
    beat_track = song.beat_track

    BINS_PER_OCTAVE = 12 * 3
    N_OCTAVES = 7
    C = librosa.amplitude_to_db(librosa.cqt(y=y, sr=sr,
                                            bins_per_octave=BINS_PER_OCTAVE,
                                            n_bins=N_OCTAVES * BINS_PER_OCTAVE),
                                ref=np.max)

    # To reduce dimensionality, we'll beat-synchronous the CQT
    tempo, beats = tuple(beat_track)

    Csync = librosa.util.sync(C, beats, aggregate=np.median)

    #####################################################################
    # Let's build a weighted recurrence matrix using beat-synchronous CQT
    # width=3 prevents links within the same bar
    # mode='affinity' here implements S_rep
    R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity',
                                          sym=True)

    # Enhance diagonals with a median filter (Equation 2)
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))

    ###################################################################
    # Now let's build the sequence matrix (S_loc) using mfcc-similarity

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    Msync = librosa.util.sync(mfcc, beats)

    path_distance = np.sum(np.diff(Msync, axis=1) ** 2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)

    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    ##########################################################
    # And compute the balanced combination

    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)

    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec) ** 2)

    A = mu * Rf + (1 - mu) * R_path

    #####################################################
    # Now let's compute the normalized Laplacian
    L = scipy.sparse.csgraph.laplacian(A, normed=True)

    # and its spectral decomposition
    evals, evecs = scipy.linalg.eigh(L)

    # We can clean this up further with a median filter.
    # This can help smooth over small discontinuities
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

    # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
    Cnorm = np.cumsum(evecs ** 2, axis=1) ** 0.5

    # If we want k clusters, use the first k normalized eigenvectors.
    k = 5

    X = evecs[:, :k] / Cnorm[:, k - 1:k]

    #############################################################
    # Let's use these k components to cluster beats into segments
    KM = sklearn.cluster.KMeans(n_clusters=k)

    seg_ids = KM.fit_predict(X)

    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

    bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

    bound_segs = list(seg_ids[bound_beats])

    bound_frames = beats[bound_beats]

    bound_frames = librosa.util.fix_frames(bound_frames,
                                           x_min=None,
                                           x_max=C.shape[1] - 1)

    bound_tuples = []
    for i in range(1, len(bound_frames)):
        bound_tuples.append((bound_frames[i-1], bound_frames[i]-1))
    bound_tuples = tuple(map(lambda x:librosa.frames_to_time(x),bound_tuples))

    pairs = zip(bound_segs, bound_tuples)
    seg_dict = dict()
    for seg, frame in pairs:
        seg_dict.setdefault(seg, []).append(frame)

    if display:
        import matplotlib.patches as patches
        plt.figure(figsize=(12, 4))
        colors = plt.get_cmap('Paired', k)

        bound_times = librosa.frames_to_time(bound_frames)
        freqs = librosa.cqt_frequencies(n_bins=C.shape[0],
                                        fmin=librosa.note_to_hz('C1'),
                                        bins_per_octave=BINS_PER_OCTAVE)

        librosa.display.specshow(C, y_axis='cqt_hz', sr=sr,
                                 bins_per_octave=BINS_PER_OCTAVE,
                                 x_axis='time')
        ax = plt.gca()

        for interval, label in zip(zip(bound_times, bound_times[1:]), bound_segs):
            ax.add_patch(patches.Rectangle((interval[0], freqs[0]),
                                           interval[1] - interval[0],
                                           freqs[-1],
                                           facecolor=colors(label),
                                           alpha=0.50))

        plt.tight_layout()
        plt.show()

    return seg_dict
