import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from acf import acf
from scipy import signal as sig

# from kalman import kalman


def load_mat_simple(path):
    """Simple loader for classic MATLAB .mat files (v7 and earlier)."""
    from scipy.io import loadmat

    mat = loadmat(path, squeeze_me=True)
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def extract_data(data):
    """Extract useful data from the loaded .mat file."""
    Atrue = data["Atrue"]
    svedala_rec = data["svedala_rec"]

    # print("Atrue", Atrue)
    # print("svedala", svedala_rec)

    return Atrue, svedala_rec


def plot_series(data, covariance):
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Time Series Data", color="blue", alpha=0.7)
    plt.plot(covariance, label="Covariance (ACF)", color="red", linestyle="--")
    plt.title("Time Series Data and its Autocorrelation")
    plt.xlabel("Sample / Lag")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    file_svedala = "svedala.mat"

    data = load_mat_simple(file_svedala)

    Atrue, svedala_rec = extract_data(data)

    ACFed = acf(svedala_rec, 1000)
    print(len(svedala_rec))
    print(len(ACFed))

    plot_series(svedala_rec, ACFed)
