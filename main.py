import soundcard as sc
import numpy as np
from aubio import pitch, onset, pvoc
import matplotlib.pyplot as plt
import sys
from multiprocessing import Process, Queue
import queue

from audiolazy import freq2str

from scipy import signal, fftpack
from scipy.optimize import curve_fit


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
    x = fftpack.rfft(x, N)
    x **= 2
    spec = np.concatenate(([x[0]], x[1:-1:2] + x[2:-1:2]))
    return np.log(spec)


def get_Z(f0, B, N, M, sr):
    harms = harmonics(f0, B, M)
    Z = np.exp(np.arange(N)[:, None] * harms * 2j * np.pi / sr)
    return Z


def harmonics(f, beta, M):
    l = np.arange(1, M + 1)
    phi = f * l
    if beta > 0:
        phi *= np.sqrt(1. + beta * l ** 2)
    return phi


def find_f0_and_B(spec, M, sr, ref_f0):
    factor = (len(spec) - 1) * 2 / sr

    # find spectral peaks
    peaks = signal.argrelmax(spec[:int(ref_f0 * (M + 1) * factor)])[0]
    peak_values = spec[peaks]

    # low pass
    # thresh = np.min(spec[:round(30 * factor)])
    # peaks = peaks[np.where(peak_values > thresh)]
    # peak_values = spec[peaks]

    peaks_freq = peaks / factor

    rough_freq = np.sort(peaks_freq[np.argsort(peak_values)[-M:]])
    # get approximated f0
    freq_diff = np.diff(rough_freq)
    med_freq = np.median(freq_diff)
    print(med_freq, ref_f0, rough_freq)
    if ref_f0:
        med_freq = ref_f0
    # get dominant harmonic peaks

    def func(m, f0, B):
        return m * f0 * np.sqrt(1. + B * m ** 2)

    # remove false harmonics
    x = np.round(peaks_freq / med_freq).astype(int)
    new_peaks_freq = []
    for i in range(1, M + 1):
        idx = np.where(x == i)[0]
        if len(idx) > 1:
            # dist = np.abs(med_freq * i - peaks_freq[idx])
            dist = -peak_values[idx]
            new_peaks_freq.append(peaks_freq[idx[np.argmin(dist)]])
        elif len(idx) == 1:
            new_peaks_freq.append(peaks_freq[idx[0]])

    peaks_freq = np.array(new_peaks_freq)
    x = np.round(peaks_freq / med_freq)
    print(np.round(np.diff(peaks_freq) * factor))

    plt.plot(spec)
    plt.vlines(np.round(peaks_freq * factor), -30, 20)
    plt.ylim(-30, 20)
    plt.xlim(0, len(spec) // 2)
    plt.show()
    param, _ = curve_fit(func, x, peaks_freq, bounds=(0., [1400., 1e-3]))

    return param[0], param[1]


def process(x, window, N, M, sr, ref_f0):
    x = x / np.abs(x).max()
    x = fft(x * window, N)
    return find_f0_and_B(x, M, sr, ref_f0)


# speaker = sc.get_speaker('2x2')
# mic = sc.get_microphone('2x2')
speaker = sc.default_speaker()
mic = sc.default_microphone()

sr = 44100
buffersize = 128
# hopsize = 256
winsize = 2048
fftsize = 2 ** 16

pitch_o = pitch('default', winsize, buffersize, sr)
onset_o = onset('hfc', winsize, buffersize, sr)
onset_o.set_threshold(0.1)
onset_o.set_silence(-50.)

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
                p = Process(target=f, args=(q, buf, window, fftsize, 20, sr, pitch))
                p.start()
                idx = -1

        try:
            f0, B = q.get(block=False)
            p.join(timeout=None)
            # print(freq2str(f0))
            sys.stdout.write("\rf0: %.2f, B: %.6f, %.2f" % (f0, B, pitch))
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
