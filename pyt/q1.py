import numpy as np
import pandas as pd
import scipy
from pxx import pxx
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile
from scipy import signal as sig

def load_mat_simple(path):
    """Simple loader for classic MATLAB .mat files (v7 and earlier)."""
    from scipy.io import loadmat
    mat = loadmat(path, squeeze_me=True)
    return {k: v for k, v in mat.items() if not k.startswith('__')}


def extract_data(data):
    first_key = list(data.keys())[0]
    useful_data = data[first_key]

    print("Variables:", list(data.keys()))

    print(useful_data)

    x = useful_data['x'].item()
    dt = useful_data['dt'].item()

    print('x=', x)
    print('dt=', dt)


    return x, dt

def plot_spectrum(freq, psd):

    plt.subplot(1,2,1)

    plt.plot(freq, psd)
    plt.xlabel('frequency')
    plt.ylabel('Power spectral density')
    plt.title('freq vs psd')

    plt.subplot(1,2,2)
    plt.plot(freq, psd)
    plt.xscale('log')
    plt.xlabel('frequency')
    plt.ylabel('Power spectral density')
    plt.title('freq vs psd but log scale')
    plt.show()

def calc_keynote_f(freq, psd):

    pos_mask = freq > 0
    freq_pos = freq[pos_mask]
    psd_pos = psd[pos_mask]

    f0 = freq_pos[np.argmax(psd_pos)]
    print("keynote frequency:", f0, "Hz")



def save_as_wav(filename, signal, dt):
    """Saves the audio signal as a .wav file in the current directory."""
    fs = int(1 / dt)
    # Normalize the signal to 16-bit integer range for WAV format
    normalized_signal = np.int16(signal / np.max(np.abs(signal)) * 32767)
    scipy.io.wavfile.write(filename, fs, normalized_signal)
    print(f"\nAudio saved to {filename}")


def filter_signal(signal_data, dt, cutoff_freq, btype='lowpass', order=4):
    """
    Apply a Butterworth filter (lowpass or highpass) for noise reduction.

    Args:
        signal_data (np.array): The input audio signal.
        dt (float): The time step between samples.
        cutoff_freq (float): The cutoff frequency in Hz.
        btype (str): The type of filter, 'lowpass' or 'highpass'.
        order (int): The order of the filter.

    Returns:
        np.array: The filtered signal.
    """
    fs = 1 / dt  # Calculate sampling frequency
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    
    # Get the filter coefficients 
    b, a = sig.butter(order, normal_cutoff, btype=btype, analog=False)
    
    # Apply the filter
    filtered_signal = sig.filtfilt(b, a, signal_data)
    return filtered_signal


if __name__ == "__main__":

    file_cello = "cello.mat"

    file_trombone = "trombone.mat"


    data_cello = load_mat_simple(file_cello)

    data_trombone = load_mat_simple(file_trombone)

    cello, dt_cello = extract_data(data_cello)

    trombone, dt_trombone = extract_data(data_trombone)


    psd_cello, freq_cello = pxx(cello, dt_cello)

    psd_trombone, freq_trombone = pxx(trombone, dt_trombone)

    print("************CELLO************")
    print('psd=', psd_cello)
    print('freq=', freq_cello)

    print("************TROMBONE************")
    print('psd=', psd_trombone)
    print('freq=', freq_trombone)

    calc_keynote_f(freq_cello, psd_cello)
    calc_keynote_f(freq_trombone, psd_trombone)
    plot_spectrum(freq_trombone, psd_trombone)
    plot_spectrum(freq_cello, psd_cello)

    #save_as_wav("trombone_output.wav", trombone, dt_trombone)

    cello_filtered = filter_signal(cello, dt_cello, cutoff_freq=1500, btype='lowpass')


    psd_cello_filtered, freq_cello_filtered = pxx(cello_filtered, dt_cello)

    plot_spectrum(freq_cello_filtered, psd_cello_filtered)





    
