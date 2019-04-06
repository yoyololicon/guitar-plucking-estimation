import soundcard as sc
import numpy as np
from aubio import pitch, onset, pvoc
import matplotlib.pyplot as plt
import sys
from multiprocessing import Process, Queue, Pipe
import queue
from numba import jit
from time import time
import random
from sklearn.preprocessing import MinMaxScaler

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
    limit = np.where(idx > N // 2 - 1)[0]
    if len(limit):
        idx = idx[:limit[0]]
        phi = phi[:limit[0]]
    return idx, phi


def process(x, sr, window, N, M, init_f0):
    x = x / np.abs(x).max()
    x = fft(x * window, N)

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


# speaker = sc.get_speaker('2x2')
# mic = sc.get_microphone('2x2')
speaker = sc.default_speaker()
mic = sc.default_microphone()

sr = 44100
buffersize = 128
winsize = 2048
fftsize = 2 ** 19

num_fret = 12
num_string = 6
pos_table = np.empty((num_string, num_fret + 1, 2))

pitch_o = pitch('default', winsize, buffersize, sr)
onset_o = onset('hfc', winsize, buffersize, sr)
onset_o.set_threshold(0.2)
onset_o.set_silence(-60.)

window = signal.get_window('hamming', winsize)
buf = np.empty(winsize)


def f(p):
    while 1:
        args = p.recv()
        p.send(process(*args))


parent_conn, child_conn = Pipe()
p = Process(target=f, args=(child_conn,))
pitch_list = []

idx = -1
string_idx = -1
random_fret = random.randint(0, num_fret)
mean_size = 5
data_list = [0] * mean_size
is_training = True

scaler = MinMaxScaler(copy=False)
p.start()
try:
    with mic.recorder(samplerate=sr, channels=[0], blocksize=buffersize) as mic2, speaker.player(samplerate=sr,
                                                                                                 blocksize=buffersize) as sp:
        while 1:
            data = mic2.record(numframes=buffersize)
            sp.play(data)

            data = data.mean(1).astype(np.float32)

            pitch = pitch_o(data)[0]
            offset = onset_o(data)[0]
            if offset:
                idx = 0

            if idx >= winsize:
                parent_conn.send((buf, sr, window, fftsize, 25, sum(pitch_list) / len(pitch_list)))
                pitch_list.clear()
                idx = -1
            elif idx >= 0:
                buf[idx:idx + buffersize] = data[:winsize - idx]
                idx += buffersize
                pitch_list.append(pitch)

            if parent_conn.poll():
                data = np.array(parent_conn.recv())
                if string_idx in range(num_string):
                    if len(data_list) < mean_size:
                        data_list.append(data)
                        print(len(data_list))
                elif string_idx > num_string:
                    data[0] = np.log(data[0])
                    data = scaler.transform(data[None, :])[0]
                    dist = np.sqrt(np.sum((pos_table - data) ** 2, 1))
                    best_pos = np.argmin(dist)
                    s, f = divmod(best_pos, num_fret + 1)
                    print(s + 1, f)

            if len(data_list) == mean_size:
                if string_idx in range(num_string):
                    mean_value = np.vstack(data_list).mean(0)
                    frets = random_fret - np.arange(num_fret + 1)
                    pos_table[string_idx, :] = np.power(2, frets[:, None] / [12, 6]) * mean_value
                    # pos_table[string_idx, :, 0] = np.power(2, frets / 12) * mean_value[0]
                    # pos_table[string_idx, :, 1] = np.power(2, frets / 6) * mean_value[1]

                string_idx += 1
                random_fret = random.randint(0, num_fret)
                data_list.clear()
                if string_idx in range(num_string):
                    print("Pluck {:d} string at {:d} fret.".format(string_idx + 1, random_fret))

            if string_idx == num_string:
                pos_table[:, :, 0] = np.log(pos_table[:, :, 0])
                pos_table = pos_table.reshape(-1, 2)
                pos_table = scaler.fit_transform(pos_table)
                string_idx += 1

                # sys.stdout.write("\rf0: %.2f, B: %.10f, %.2f" % (f0, B, pitch))

except KeyboardInterrupt:
    print("Finish")
