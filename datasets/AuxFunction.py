import numpy as np


def FFT(x):
    x = np.fft.fft(x)
    x = np.abs(x) / len(x)
    x = x[range(int(x.shape[0] / 2))]
    return x


def add_noise(x, snr):
    noise = np.random.randn(len(x))  # generate random noise
    signal_power = np.sum(np.power(x, 2)) / x.shape[0]
    noise_power = signal_power / np.power(10, (snr/10))
    noise = np.sqrt(noise_power / np.std(noise)) * noise
    noise_signal = x + noise
    return noise_signal