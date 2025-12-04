import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig

from acf import acf
from kalman import kalman


def load_mat_simple(path):
    """Simple loader for classic MATLAB .mat files (v7 and earlier)."""
    from scipy.io import loadmat

    mat = loadmat(path, squeeze_me=True)
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def extract_data(data):
    """Extract true parameters and reconstructed temperature series."""
    Atrue = data["Atrue"]
    svedala_rec = data["svedala_rec"]
    return Atrue, svedala_rec


def plot_series(data, filtered_data, covariance):
    """Plot original/filtered series and their covariance function."""
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 4))

    # Top-left: time series
    plt.subplot(2, 2, 1)
    plt.plot(data, label="Time Series Data", color="blue", alpha=0.7)
    plt.xlabel("Sample / Lag")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # Top-right: covariance (ACF)
    plt.subplot(2, 2, 2)
    plt.plot(covariance, label="Covariance (ACF)", color="red", linestyle="--")
    plt.xlabel("Lag")
    plt.ylabel("Autocovariance")
    plt.legend()
    plt.grid(True)

    # Bottom-left: filtered series
    plt.subplot(2, 2, 3)
    plt.plot(filtered_data, label="Filtered Time Series Data", color="blue", alpha=0.7)
    plt.xlabel("Sample / Lag")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_kalman_results(Xtt, Atrue, ft_hat):
    """Compare estimated a1,a2 with true ones and show residuals."""
    N = Xtt.shape[1]
    t = np.arange(N)

    plt.figure(figsize=(12, 8))

    # a1(t)
    plt.subplot(2, 1, 1)
    plt.plot(t, Xtt[0, :], "b", label="Estimated a1", linewidth=1.2)
    plt.plot(t, Atrue[:, 0], "r--", label="True a1", linewidth=1.2)
    plt.xlabel("Time (hours)")
    plt.ylabel("a1(t)")
    plt.legend()
    plt.title("Comparison of a1(t)")
    plt.grid(True)

    # a2(t)
    plt.subplot(2, 1, 2)
    plt.plot(t, Xtt[1, :], "b", label="Estimated a2", linewidth=1.2)
    plt.plot(t, Atrue[:, 1], "r--", label="True a2", linewidth=1.2)
    plt.xlabel("Time (hours)")
    plt.ylabel("a2(t)")
    plt.legend()
    plt.title("Comparison of a2(t)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Residuals
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(ft_hat)), ft_hat)
    plt.xlabel("Time (hours)")
    plt.ylabel("Residuals")
    plt.title("Estimated residuals f̂_t = Y_t - Ŷ_{t|t-1}")
    plt.grid(True)
    plt.show()

    # ACF of residuals (for whiteness check)
    ACF_res = acf(ft_hat, 200)
    plt.figure(figsize=(8, 3))
    plt.plot(ACF_res)
    plt.xlabel("Lag")
    plt.ylabel("Autocovariance")
    plt.title("ACF of residuals")
    plt.grid(True)
    plt.show()


def apply_seasonal_filter(data, s):
    """
    Remove a seasonal component of period s.
    Equivalent to Matlab: filter([1 zeros(1,s-1) -1], 1, data)
    """
    b = np.zeros(s + 1)
    b[0] = 1
    b[s] = -1
    a = [1]
    return sig.lfilter(b, a, data)


if __name__ == "__main__":
    file_svedala = "svedala.mat"

    data = load_mat_simple(file_svedala)
    Atrue, svedala_rec = extract_data(data)

    # ACF of original series
    ACF_raw = acf(svedala_rec, 690)
    print("Length of original series:", len(svedala_rec))
    print("Length of ACF_raw:", len(ACF_raw))

    # Remove daily periodicity (24h)
    filtered_svedala = apply_seasonal_filter(svedala_rec, 24)

    # ACF of filtered series
    ACF_filt = acf(filtered_svedala, 690)

    # Plots for Q2.1
    plot_series(
        data=svedala_rec,
        filtered_data=filtered_svedala,
        covariance=ACF_raw,
    )

    plot_series(
        data=filtered_svedala,
        filtered_data=filtered_svedala,
        covariance=ACF_filt,
    )

    # Kalman filter for Q2.2 / Q2.3
    Xtt, Ytt1, ft_hat = kalman(filtered_svedala, se2=1e-4, sf2=1.0)
    plot_kalman_results(Xtt, Atrue, ft_hat)
