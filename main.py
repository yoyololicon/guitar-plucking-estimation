import soundcard as sc
import numpy as np
from aubio import pitch, onset, pvoc
import matplotlib.pyplot as plt
import sys
from multiprocessing import Process, Queue
import queue

from audiolazy import freq2str
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from scipy import signal, fftpack
from scipy.optimize import curve_fit, brute


# print(sc.all_speakers(), sc.all_microphones(), sc.default_speaker(), sc.default_microphone())

class buffer():
    def __init__(self, size):
        self.buf = np.empty(size)
        self.size = size

    def __call__(self, x):
        data_size = len(x)
        assert data_size < self.size
        self.buf[:] = np.concatenate((self.buf[data_size:], x))
        return self.buf


def fft(x, N):
    x = fftpack.rfft(x, N) / len(x)
    x **= 2
    spec = np.concatenate((x[:1], x[1:-1:2] + x[2:-1:2]))
    return 10 * np.log10(spec)


def get_Z(f0, B, N, M, sr):
    harms = harmonics(f0, B, N, M, sr)[1]
    Z = np.exp(np.arange(N)[:, None] * harms * 2j * np.pi / sr)
    return Z


def harmonics(f, beta, N, M, sr):
    l = np.arange(1, M + 1)
    phi = f * l
    if beta > 0:
        phi *= np.sqrt(1. + beta * l ** 2)
    idx = np.rint(phi * N / sr).astype(np.int)
    return idx, phi


def find_f0_and_B(spec, M, sr, ref_f0):
    factor = len(spec) * 2 / sr

    # find spectral peaks
    raw_peaks = signal.argrelmax(spec)[0]
    raw_peak_values = spec[raw_peaks]

    # thresholding
    idx = np.where(raw_peak_values > raw_peak_values.max() - 30)
    peaks = raw_peaks[idx]
    peak_values = raw_peak_values[idx]

    # get dominant harmonic peaks
    peaks = np.sort(peaks[np.argsort(peak_values)[-M:]])
    peaks_freq = peaks / factor

    # get approximated f0
    freq_diff = np.diff(peaks_freq)
    med_freq = np.median(freq_diff)
    if ref_f0:
        med_freq = ref_f0

    # remove false harmonics
    x = np.round(peaks_freq / med_freq).astype(int)
    new_peaks_freq = []
    for i in range(1, M + 1):
        idx = np.where(x == i)[0]
        if len(idx):
            dist = -raw_peak_values[idx]
            dist2 = np.abs(med_freq * i - peaks_freq[idx])
            if np.argmin(dist) == np.argmin(dist2):
                new_peaks_freq.append(peaks_freq[idx[np.argmin(dist)]])
    peaks_freq = np.array(new_peaks_freq)
    x = np.round(peaks_freq / med_freq)

    #plt.plot(spec)
    #plt.vlines(np.round(peaks_freq * factor), -100, 0)
    #plt.ylim(-100, 0)
    #plt.xlim(0, len(spec) // 3)
    #plt.show()

    def func(m, f0, B):
        return m * f0 * np.sqrt(1. + B * m ** 2)
    print(inharmonic_sum(spec, M, sr, med_freq))
    param, _ = curve_fit(func, x, peaks_freq, bounds=([30., 0.], [1400., 1e-3]))
    return param[0], param[1]

def inharmonic_sum(x, M, sr, init_f0):
    N = len(x) * 2

    def func(params):
        f0, B = params
        cost = np.sum(x[harmonics(f0, B, N, M, sr)[0]])
        return -cost

    rranges = (slice(init_f0 - 2, init_f0 + 2, 0.1), slice(0, 1e-3, 1e-4))
    init_f0, B = brute(func, rranges, finish=None)
    rranges = (slice(init_f0 - 0.1, init_f0 + 0.1, 0.01), slice(max(0, B - 1e-4), B + 1e-4, 1e-5))
    init_f0, B = brute(func, rranges, finish=None)
    rranges = (slice(init_f0 - 0.01, init_f0 + 0.01, 0.001), slice(max(0, B - 1e-5), B + 1e-5, 1e-6))
    init_f0, B = brute(func, rranges, finish=None)
    rranges = (slice(init_f0 - 0.001, init_f0 + 0.001, 0.0001), slice(max(0, B - 1e-6), B + 1e-6, 1e-7))
    init_f0, B = brute(func, rranges, finish=None)
    return init_f0, B


def process(x, window, N, M, sr, ref_f0):
    x = x / np.abs(x).max()
    x = fft(x * window, N)
    return find_f0_and_B(x, M, sr, ref_f0)


# speaker = sc.get_speaker('2x2')
# mic = sc.get_microphone('2x2')
speaker = sc.default_speaker()
mic = sc.default_microphone()

sr = 44100
buffersize = 64
# hopsize = 256
winsize = 2048
fftsize = 2 ** 19

pitch_o = pitch('default', winsize, buffersize, sr)
onset_o = onset('hfc', winsize, buffersize, sr)
onset_o.set_threshold(0.2)
onset_o.set_silence(-60.)

# pv = pvoc(winsize, buffersize)  # phase vocoder
# pv.set_window('hanning')
temp = buffer(winsize)
# fftbuf = np.empty(fftsize)
window = signal.get_window('hann', winsize)


def f(q, *args):
    q.put(process(*args))


q = Queue(1)
buf = np.empty(winsize)
idx = -1
with mic.recorder(samplerate=sr, channels=[0], blocksize=buffersize) as mic2, speaker.player(samplerate=sr,
                                                                                             blocksize=buffersize) as sp:
    while 1:
        data = mic2.record(numframes=buffersize)
        sp.play(data)

        data = data.mean(1).astype(np.float32)

        pitch = pitch_o(data)[0]
        offset = onset_o(data)[0]
        # spec = pv(data).norm
        # x = temp(data)

        if offset:
            idx = 0

        if idx >= 0:
            buf[idx:idx + buffersize] = data[:winsize - idx]
            idx += buffersize
            if idx >= winsize:
                p = Process(target=f, args=(q, buf, window, fftsize, 25, sr, pitch))
                p.start()
                idx = -1

        try:
            f0, B = q.get(block=False)
            p.join(timeout=None)
            # print(freq2str(f0))
            sys.stdout.write("\nf0: %.2f, B: %.10f, %.2f" % (f0, B, pitch))
        except queue.Empty:
            pass

        # spec = fft(x * window, fftsize)
        # f0, B = find_f0_and_B(np.log(spec), 20, sr)

        # if offset:
        # pos = int((offset - 1) * hopsize) - onset_o.get_delay()
        # segment = buf_data[pos:]
        # print(len(segment))
        # print(process(segment, pitch), pitch)
        # plt.plot(buf_data[pos:])
        # plt.show()

        # print("hello", end='\r')
