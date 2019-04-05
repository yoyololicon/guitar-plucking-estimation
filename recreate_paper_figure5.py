import numpy as np
from scipy.io import loadmat
from scipy import signal, fftpack
from scipy.optimize import brute, curve_fit
import matplotlib.pyplot as plt
import argparse
from aubio import onset, source

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='Recreate plucking experiment.')
parser.add_argument('infile', type=str, help='shoud be the file "recording_of_plucking_with_sudden_changes.wav"')
parser.add_argument('matfile', type=str, help='the training data matlab file.')
parser.add_argument('-fftsize', type=int, default=2 ** 19)
parser.add_argument('-segment_size', type=float, default=0.04, help='input segment size (in secondes).')
parser.add_argument('-M', type=int, default=25, help='maximum number of harmonics.')
parser.add_argument('-L', type=float, default=64.3, help='Length of strings.')


def segment_from_onsets(filename, dur):
    win_s = 1024
    hop_s = 256
    s = source(filename, 0, hop_s)
    sr = s.samplerate

    o = onset("hfc", win_s, hop_s, sr)
    o.set_threshold(0.21)
    o.set_silence(-30.)
    o.set_minioi_s(dur)
    raw_onsets = []
    chunks = []

    while 1:
        samples, read = s()
        if o(samples):
            raw_onsets.append(o.get_last())
        chunks.append(samples.copy())
        if read < hop_s:
            break

    y = np.concatenate(chunks)
    raw_segments = np.split(y, raw_onsets)[1:]
    segment_len = int(sr * dur)
    segments, onsets = [], []
    for seg, ost in zip(raw_segments, raw_onsets):
        if len(seg) < segment_len:
            pass
        else:
            onsets.append(ost)
            segments.append(seg[:segment_len])
    segments = np.stack(segments)
    segments /= np.abs(segments).max(1, keepdims=True)
    return segments, np.array(onsets) / sr, y, sr


def get_Z(f0, N, sr, M, B):
    _, harms = harmonics(N, sr, M, f0, B)
    Z = np.exp(np.arange(N)[:, None] * harms * 2j * np.pi / sr)
    return Z


def fft(x, N):
    x = fftpack.fft(x, N) / len(x)
    x = np.abs(x)
    x_db = 20 * np.log10(x)
    return x_db


def harmonics(nfft, sr, M, f, beta=0.):
    l = np.arange(1, M + 1)
    phi = f * l
    if beta > 0:
        phi *= np.sqrt(1. + beta * l ** 2)
    idx = np.rint(phi * nfft / sr).astype(np.int)
    return idx, phi


def harmonic_sum(x, bounds, M, sr):
    def best_f0(grid):
        max_cost = None
        best_f0 = 0.
        for f in grid:
            cost = np.sum(x[harmonics(len(x), sr, M, f)[0]])
            if max_cost is None:
                max_cost = cost
            elif cost > max_cost:
                max_cost = cost
                best_f0 = f
        return best_f0

    f0grid = np.arange(bounds[0], bounds[1], 0.1)
    pitch = best_f0(f0grid)
    f0grid = np.arange(pitch - 0.1, pitch + 0.1, 0.01)
    pitch = best_f0(f0grid)
    f0grid = np.arange(pitch - 0.01, pitch + 0.01, 0.001)
    pitch = best_f0(f0grid)
    return pitch


def find_f0_and_B(spec, M, sr):
    N = len(spec)

    # find spectral peaks
    raw_peaks = signal.argrelmax(spec[:N // 2])[0]
    raw_peak_values = spec[raw_peaks]

    # thresholding
    idx = np.where(raw_peak_values > raw_peak_values.max() - 30)
    peaks = raw_peaks[idx]
    peak_values = raw_peak_values[idx]

    # get dominant harmonic peaks
    peaks = np.sort(peaks[np.argsort(peak_values)[-M:]])
    peaks_freq = peaks / N * sr

    # get approximated f0
    freq_diff = np.diff(peaks_freq)
    med_freq = np.median(freq_diff)

    # remove false harmonics
    x = np.round(peaks_freq / med_freq).astype(int)
    new_peaks_freq = []
    for i in range(1, M + 1):
        idx = np.where(x == i)[0]
        if len(idx) > 1:
            # dist = np.abs(med_freq * i - peaks_freq[idx])
            dist = -raw_peak_values[idx]
            new_peaks_freq.append(peaks_freq[idx[np.argmin(dist)]])
        elif len(idx) == 1:
            new_peaks_freq.append(peaks_freq[idx[0]])
    peaks_freq = np.array(new_peaks_freq)

    x = np.round(peaks_freq / med_freq)
    #plt.plot(spec)
    #plt.vlines(np.round(peaks_freq / sr * N), -60, 10)
    #plt.ylim(-60, 10)
    #plt.xlim(0, N // 10)
    #plt.show()

    def func(m, f0, B):
        return m * f0 * np.sqrt(1. + B * m ** 2)

    param, _ = curve_fit(func, x, peaks_freq, bounds=(0., [1400., 1e-3]))
    return param[0], param[1]


def inharmonic_sum(x, M, sr, init_f0):
    N = len(x)

    def func(params):
        f0, B = params
        cost = np.sum(x[harmonics(N, sr, M, f0, B)[0]])
        return -cost

    rranges = (slice(init_f0 - 1, init_f0 + 1, 0.1), slice(0, 1e-3, 1e-4))
    init_f0, B = brute(func, rranges, finish=None)
    rranges = (slice(init_f0 - 0.1, init_f0 + 0.1, 0.01), slice(max(0, B - 1e-4), B + 1e-4, 1e-5))
    init_f0, B = brute(func, rranges, finish=None)
    rranges = (slice(init_f0 - 0.01, init_f0 + 0.01, 0.001), slice(max(0, B - 1e-5), B + 1e-5, 1e-6))
    init_f0, B = brute(func, rranges, finish=None)
    rranges = (slice(init_f0 - 0.001, init_f0 + 0.001, 0.0001), slice(max(0, B - 1e-6), B + 1e-6, 1e-7))
    init_f0, B = brute(func, rranges, finish=None)
    return init_f0, B


if __name__ == '__main__':
    args = parser.parse_args()
    model_params = loadmat(args.matfile)
    norm = model_params['normalizationConstant'][0]
    mu = model_params['mu']

    segs, onsets, y, sr = segment_from_onsets(args.infile, args.segment_size)

    std = segs.shape[1] / 2 / 2.5
    x = signal.hilbert(segs)
    window = signal.gaussian(x.shape[1], std)
    x *= window

    N = x.shape[1]
    spec = fft(x, args.fftsize)

    M = args.M
    L_open = args.L

    m = np.arange(1, M + 1)
    delta = 1

    P = np.linspace(0.1, 0.5, 1000)

    Cm = 2 * delta / (m[None, :] ** 2 * np.pi ** 2 * P[:, None] * (1 - P[:, None])) * np.abs(
        np.sin(m[None, :] * np.pi * P[:, None]))

    pos = []
    strings = []
    frets = []

    for i in range(x.shape[0]):
        tx = x[i]
        fx = spec[i]

        # init_f0 = harmonic_sum(fx, [75, 700], 5, sr)
        # f0, B = inharmonic_sum(fx, M, sr, init_f0)
        f0, B = find_f0_and_B(fx, M, sr)
        norm_f0 = f0 / norm[0]
        norm_B = B / norm[1]

        euclidean_dist = np.sqrt((np.log(norm_f0) - np.log(mu[:, 0])) ** 2 + (norm_B - mu[:, 1]) ** 2)
        idx = np.argmin(euclidean_dist)
        fret_estimate = idx % 13
        string_estimate = idx // 13 + 1

        Z = get_Z(f0, N, sr, M, B)
        alpha = np.linalg.lstsq(Z, tx)[0]
        # alpha = np.linalg.inv(Z.T @ Z) @ Z.T @ tx
        abs_a = np.abs(alpha)

        L = L_open * 2 ** (-fret_estimate / 12)
        D_LS_cost = np.sqrt(np.mean((10 * np.log10(abs_a / Cm)) ** 2, 1))
        pluck_pos = P[np.argmin(D_LS_cost)] * L

        pos.append(pluck_pos)
        strings.append(string_estimate)
        frets.append(fret_estimate)
        # print(string_estimate, fret_estimate, pluck_pos, f0, B)
        print(
            "Estimating string, fret and plucking position for segment {:d} out of {:d} segments".format(i + 1,
                                                                                                         x.shape[0]))

    time_axis = np.arange(len(y)) / sr
    fig, axes = plt.subplots(3, sharex=True, figsize=(8, 6))
    axes[0].plot(time_axis, y)
    axes[0].set_xlim([0, time_axis[-1]])
    axes[0].set_ylim([-1, 1])
    axes[0].set_ylabel('Ampl.')

    axes[1].plot(onsets, pos, 'x')
    axes[1].set_ylim([4, 32])
    axes[1].set_ylabel('$\hat{P}$[cm]')
    axes[1].minorticks_on()
    axes[1].grid(b=True, which='both', linestyle=':')

    for p in range(1, 7):
        axes[2].plot([0, len(y) / sr], [p, p], linewidth=1, color=[0.4] * 3)
    axes[2].set_ylim([0.1, 6.9])
    for ost, string, fret in zip(onsets, strings, frets):
        axes[2].text(ost - 0.2, string - 0.5, "%1.0f" % (fret), fontsize=18)
    axes[2].set_yticks(list(range(1, 7)))
    axes[2].set_ylabel('String Est.')
    axes[2].set_xlabel('Time [sec]')
    plt.show()
