import soundcard as sc
import numpy as np
from aubio import pitch, onset
import matplotlib.pyplot as plt
import sys

from scipy import signal, fftpack
from scipy.optimize import minimize_scalar, brute


# print(sc.all_speakers(), sc.all_microphones(), sc.default_speaker(), sc.default_microphone())

class buffer():
    def __init__(self, size):
        self.buf = np.empty(size, dtype=np.float32)
        self.size = size

    def __call__(self, x):
        data_size = len(x)
        assert data_size < self.size
        self.buf[:] = np.concatenate((self.buf[data_size:], x))
        return self.buf


def process(x, f0):
    N = len(x)
    x /= np.abs(x).max()
    x = signal.hilbert(x)
    x *= signal.hann(N)
    spec = np.log(np.abs(fftpack.fft(x, 4096)) ** 2)

    def func(params):
        f0, B = params
        cost = np.sum(spec[harmonics(N, sr, 15, f0, B)[0]])
        return -cost

    rranges = (slice(f0 - 1, f0 + 1, 1), slice(0, 1e-3, 1e-4))
    init_f0, B = brute(func, rranges, finish=None)
    rranges = (slice(f0 - 1, f0 + 1, 1), slice(B - 1e-4, B + 1e-4, 1e-5))
    init_f0, B = brute(func, rranges, finish=None)
    rranges = (slice(f0 - 1, f0 + 1, 1), slice(B - 1e-5, B + 1e-5, 1e-6))
    init_f0, B = brute(func, rranges, finish=None)
    rranges = (slice(f0 - 1, f0 + 1, 1), slice(B - 1e-6, B + 1e-6, 1e-7))
    init_f0, B = brute(func, rranges, finish=None)
    return B


def get_Z(f0, N, sr, M, B):
    harms = harmonics(N, sr, M, f0, B)
    Z = np.exp(np.arange(N)[:, None] * harms * 2j * np.pi / sr)
    return Z


def harmonics(nfft, sr, M, f, beta=0.):
    l = np.arange(1, M + 1)
    phi = f * l
    if beta > 0:
        phi *= np.sqrt(1. + beta * l ** 2)
    idx = np.rint(phi * nfft / sr).astype(np.int)
    return idx, phi


speaker = sc.get_speaker('2x2')
mic = sc.get_microphone('2x2')

sr = 44100
buffersize = 128
hopsize = 256
winsize = 2048

pitch_o = pitch('default', winsize, hopsize, sr)
onset_o = onset('default', winsize, hopsize, sr)
onset_o.set_threshold(0.5)

temp = buffer(max(winsize, onset_o.get_delay()))

with mic.recorder(samplerate=44100, channels=[0], blocksize=buffersize) as mic2, speaker.player(samplerate=44100,
                                                                                                blocksize=buffersize) as sp:
    while 1:
        data = mic2.record(numframes=hopsize)
        data = data.mean(1).astype(np.float32)

        pitch = pitch_o(data)[0]
        offset = onset_o(data)[0]
        buf_data = temp(data)
        if offset:
            pos = int((offset - 1) * hopsize) - onset_o.get_delay()
            segment = buf_data[pos:]
            print(len(segment))
            print(process(segment, pitch), pitch)
            #plt.plot(buf_data[pos:])
            #plt.show()

        sp.play(data)
        # print("hello", end='\r')
        # sys.stdout.write("\rpitch: %.2f, amp: %.2f" % (pitch, offset))
