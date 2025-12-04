import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from acf import acf
from scipy import signal as sig

from filterpy.kalman import KalmanFilter
from kalman import kalman


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


def plot_series(data, filtered_data, covariance):
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 4))


    plt.subplot(2, 2, 1)

    plt.plot(data, label="Time Series Data", color="blue", alpha=0.7)
    plt.xlabel("Sample / Lag")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)

    plt.plot(covariance, label="Covariance (ACF)", color="red", linestyle="--")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)

    plt.plot(filtered_data, label="FilteredTime Series Data", color="blue", alpha=0.7)
    plt.xlabel("Sample / Lag")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)


    plt.show()
    
def plot_kalman_results(Xtt, Atrue, ft_hat):
    N = Xtt.shape[1]
    t = np.arange(N)
    plt.figure(figsize=(12, 8))

    # Plot a1
    plt.subplot(2, 1, 1)
    plt.plot(t, Xtt[0, :], 'b', label='Estimated a1', linewidth=1.2)
    plt.plot(t, Atrue[:, 0], 'r--', label='True a1', linewidth=1.2)
    plt.xlabel('Time (hours)')
    plt.ylabel('a1(t)')
    plt.legend()
    plt.title('Comparison of a1(t)')
    plt.grid(True)

    # Plot a2
    plt.subplot(2, 1, 2)
    plt.plot(t, Xtt[1, :], 'b', label='Estimated a2', linewidth=1.2)
    plt.plot(t, Atrue[:, 1], 'r--', label='True a2', linewidth=1.2)
    plt.xlabel('Time (hours)')
    plt.ylabel('a2(t)')
    plt.legend()
    plt.title('Comparison of a2(t)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot residuals
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(ft_hat)), ft_hat)
    plt.xlabel('Time (hours)')
    plt.ylabel('Residuals')
    plt.title('Estimated residuals')
    plt.grid(True)
    plt.show()

def apply_seasonal_filter(data, s):
    """
    Applies a filter to remove a seasonal component of period s.
    Equivalent to MATLAB's: filter([1 zeros(1,s-1) -1], 1, data)
    """
    # b coefficients: [1, 0, 0, ..., -1]
    b = np.zeros(s + 1)
    b[0] = 1
    b[s] = -1
    
    # a coefficient
    a = [1]
    
    # Apply the linear filter
    return sig.lfilter(b, a, data)


if __name__ == "__main__":
    file_svedala = "svedala.mat"

    data = load_mat_simple(file_svedala)

    Atrue, svedala_rec = extract_data(data)

    ACFed = acf(svedala_rec, 690)
    # confidence interval bounds sigma somethng

    


    print(len(svedala_rec))
    print(len(ACFed))


    filtered_svedala = apply_seasonal_filter(svedala_rec, 24)

    ACFed = acf(filtered_svedala, 690)

    plot_series(svedala_rec,filtered_svedala, ACFed)

    Xtt, Ytt1, ft_hat = kalman(filtered_svedala)
    plot_kalman_results(Xtt, Atrue, ft_hat)
