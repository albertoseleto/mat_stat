import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.io.wavfile
import sounddevice as sd
from scipy import signal as sig

from pxx import pxx


def load_mat_simple(path):
    """Simple loader for classic MATLAB .mat files (v7 and earlier)."""
    from scipy.io import loadmat

    mat = loadmat(path, squeeze_me=True)
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def extract_data(data):
    """Extract useful data from the loaded .mat file."""
    first_key = list(data.keys())[0]
    useful_data = data[first_key]

    print("Variables:", list(data.keys()))

    print(useful_data)

    x = useful_data["x"].item()
    dt = useful_data["dt"].item()

    print("x=", x)
    print("dt=", dt)

    return x, dt


# Question 1.2
def plot_spectrum(freq, psd, freq_filtered, psd_filtered):
    """Plot the power spectral density against frequency."""
    plt.style.use("ggplot")

    # Bigger figure but with more spacing between subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 5))

    # Top-left: original, linear scale
    ax = axes[0, 0]
    ax.plot(freq, psd)
    ax.set_xlabel("Frequency", fontsize=9)
    ax.set_ylabel("Power spectral density", fontsize=9)
    ax.set_title("Frequency vs Power Spectral Density", fontsize=11)
    ax.set_xlim(-50, 2000)

    # Top-right: original, log scale
    ax = axes[0, 1]
    ax.plot(freq, psd)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency", fontsize=9)
    ax.set_ylabel("Power spectral density", fontsize=9)
    ax.set_title("Frequency vs Power Spectral Density (Log Scale)", fontsize=11)
    ax.set_xlim(-50, 2000)
    ax.set_ylim(1e-12, 100)

    # Bottom-left: filtered, linear scale
    ax = axes[1, 0]
    ax.plot(freq_filtered, psd_filtered)
    ax.set_xlabel("Filtered frequency", fontsize=9)
    ax.set_ylabel("Power spectral density", fontsize=9)
    ax.set_title("Filtered vs Power Spectral Density", fontsize=11)
    ax.set_xlim(-50, 2000)

    # Bottom-right: filtered, log scale
    ax = axes[1, 1]
    ax.plot(freq_filtered, psd_filtered)
    ax.set_yscale("log")
    ax.set_xlabel("Filtered frequency", fontsize=9)
    ax.set_ylabel("Power spectral density (log scale)", fontsize=9)
    ax.set_title("Filtered vs Power Spectral Density (Log Scale)", fontsize=11)
    ax.set_xlim(-50, 2000)
    ax.set_ylim(1e-12, 100)

    # Extra spacing between subplots
    fig.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)

    plt.show()


# Question 1.1
def calc_keynote_f(freq, psd):
    """Calculate and print the keynote frequency from the PSD data."""
    pos_mask = freq > 0
    freq_pos = freq[pos_mask]
    psd_pos = psd[pos_mask]

    f0 = freq_pos[np.argmax(psd_pos)]  # Find frequency with maximum PSD
    print("Keynote Frequency: ", f0, "Hz")


def save_as_wav(filename, signal, dt):
    """Saves the audio signal as a .wav file in the current directory."""
    fs = int(1 / dt)
    # Normalize the signal to 16-bit integer range for WAV format
    normalized_signal = np.int16(signal / np.max(np.abs(signal)) * 32767)
    scipy.io.wavfile.write(filename, fs, normalized_signal)
    print(f"\nAudio saved to {filename}")


def filter_signal(signal_data, dt, cutoff_freq, btype="lowpass", order=4):
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
    print("psd=", psd_cello)
    print("freq=", freq_cello)

    print("************TROMBONE************")
    print("psd=", psd_trombone)
    print("freq=", freq_trombone)

    calc_keynote_f(freq_cello, psd_cello)
    calc_keynote_f(freq_trombone, psd_trombone)
    # plot_spectrum(freq_trombone, psd_trombone)
    # plot_spectrum(freq_cello, psd_cello)

    # save_as_wav("trombone_output.wav", trombone, dt_trombone)

    cello_filtered = filter_signal(cello, dt_cello, cutoff_freq=50, btype="highpass")
    psd_cello_filtered, freq_cello_filtered = pxx(cello_filtered, dt_cello)

    plot_spectrum(freq_cello, psd_cello, freq_cello_filtered, psd_cello_filtered)

    trombone_filtered = filter_signal(
        trombone, dt_trombone, cutoff_freq=50, btype="highpass"
    )
    psd_trmbone_filtered, freq_trombone_filtered = pxx(trombone_filtered, dt_trombone)
    plot_spectrum(
        freq_trombone, psd_trombone, freq_trombone_filtered, psd_trmbone_filtered
    )

    # Question 1.4
    print("\nPlaying ORIGINAL cello...")
    sd.play(cello, int(1 / dt_cello))
    sd.wait()

    print("Playing FILTERED cello...")
    sd.play(cello_filtered, int(1 / dt_cello))
    sd.wait()

    print("\nPlaying ORIGINAL trombone...")
    sd.play(trombone, int(1 / dt_trombone))
    sd.wait()

    print("Playing FILTERED trombone...")
    sd.play(trombone_filtered, int(1 / dt_trombone))
    sd.wait()
