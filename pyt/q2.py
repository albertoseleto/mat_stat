import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
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


def nice_stem(ax, x, y):
    """Clean stem plot: thin lines, small markers, no filled baseline."""
    markerline, stemlines, baseline = ax.stem(x, y, basefmt=" ")
    # Small neutral markers
    markerline.set_markersize(3)
    markerline.set_markerfacecolor("C1")  # or 'C0' if you prefer
    markerline.set_markeredgecolor("C1")
    # Thin stems
    stemlines.set_linewidth(0.8)
    # Hide baseline completely
    if baseline is not None:
        baseline.set_visible(False)


def plot_q21(original, filtered, acf_raw, acf_filt):
    """
    Q2.1 plots: original / filtered series and their ACFs
    in a single 2x2 figure.
    """
    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    # Top-left: original time series
    ax = axes[0, 0]
    ax.plot(original, color="C0", linewidth=0.8)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Value")
    ax.set_title("Original reconstructed series")
    ax.grid(True)

    # Top-right: ACF of original
    ax = axes[0, 1]
    lags_raw = np.arange(len(acf_raw))
    ax.plot(
        lags_raw,
        acf_raw,
        marker="o",
        markersize=3,
        markerfacecolor="none",
        markeredgecolor="C1",
        linewidth=0.6,
        color="C1",
    )
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("ACF of original series")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.grid(True)

    # Confidence interval (dashed lines) from `acf.py` styling
    signLvl = 0.05
    signScale = stats.norm.ppf(1 - signLvl / 2)
    N_raw = len(original)
    condInt = signScale / np.sqrt(N_raw)
    condInt_vec = condInt * np.ones_like(lags_raw)
    ax.plot(lags_raw, condInt_vec, "--", color="red", linewidth=0.8)
    ax.plot(lags_raw, -condInt_vec, "--", color="red", linewidth=0.8)
    # Fixed y-limits requested by user
    ax.set_ylim(-0.50, 1.10)
    # Caption in top-right corner explaining lines
    caption = "Dashed red: 95% Confidence Interval"
    ax.text(
        0.98,
        0.98,
        caption,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    # Bottom-left: filtered series
    ax = axes[1, 0]
    ax.plot(filtered, color="C0", linewidth=0.8)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Value")
    ax.set_title("Series after seasonal differencing (period 24)")
    ax.grid(True)

    # Bottom-right: ACF of filtered
    ax = axes[1, 1]
    lags_filt = np.arange(len(acf_filt))
    ax.plot(
        lags_filt,
        acf_filt,
        marker="o",
        markersize=3,
        markerfacecolor="none",
        markeredgecolor="C1",
        linewidth=0.6,
        color="C1",
    )
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("ACF of filtered series")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.grid(True)

    # Confidence interval (dashed lines) from `acf.py` styling
    signScale = stats.norm.ppf(1 - signLvl / 2)
    N_filt = len(filtered)
    condInt_f = signScale / np.sqrt(N_filt)
    condInt_vec_f = condInt_f * np.ones_like(lags_filt)
    ax.plot(lags_filt, condInt_vec_f, "--", color="red", linewidth=0.8)
    ax.plot(lags_filt, -condInt_vec_f, "--", color="red", linewidth=0.8)
    # Fixed y-limits requested by user
    ax.set_ylim(-0.50, 1.10)
    # Caption in top-right corner explaining lines
    caption = "Dashed red: 95% Confidence Interval"
    ax.text(
        0.98,
        0.98,
        caption,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    fig.tight_layout()
    plt.show()


def plot_kalman_results(Xtt, Atrue, ft_hat):
    """Compare estimated a1,a2 with true ones and show residuals and ACF."""
    N = Xtt.shape[1]
    t = np.arange(N)

    plt.style.use("ggplot")

    # a1(t) and a2(t) vs true
    plt.figure(figsize=(12, 6))

    # a1(t)
    plt.subplot(2, 1, 1)
    plt.plot(t, Xtt[0, :], "b", label="Estimated a1(t)", linewidth=1.0)
    plt.plot(t, Atrue[:, 0], "r--", label="True a1(t)", linewidth=1.0)
    plt.xlabel("Time (hours)")
    plt.ylabel("a1(t)")
    plt.title("Comparison of a1(t)")
    plt.legend()
    plt.grid(True)

    # a2(t)
    plt.subplot(2, 1, 2)
    plt.plot(t, Xtt[1, :], "b", label="Estimated a2(t)", linewidth=1.0)
    plt.plot(t, Atrue[:, 1], "r--", label="True a2(t)", linewidth=1.0)
    plt.xlabel("Time (hours)")
    plt.ylabel("a2(t)")
    plt.title("Comparison of a2(t)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # residuals over time
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(ft_hat)), ft_hat, color="C2", linewidth=0.8)
    plt.xlabel("Time (hours)")
    plt.ylabel("Residuals")
    plt.title(r"Estimated residuals $\hat{f}_t = Y_t - \hat{Y}_{t|t-1}$")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ACF of residuals (for whiteness check)
    # compute ACF without plotting inside acf()
    ACF_res = acf(ft_hat, 200, plotIt=False)
    lags_res = np.arange(len(ACF_res))

    plt.figure(figsize=(8, 3))
    plt.plot(
        lags_res,
        ACF_res,
        marker="o",
        markersize=3,
        markerfacecolor="none",
        markeredgecolor="C1",
        linewidth=0.6,
        color="C1",
    )
    plt.axhline(0, color="grey", linewidth=0.8)
    plt.xlabel("Lag (hours)")
    plt.ylabel("Autocorrelation")
    plt.title("ACF of residuals")
    signLvl = 0.05
    signScale = stats.norm.ppf(1 - signLvl / 2)
    N_res = len(ft_hat)
    condInt = signScale / np.sqrt(N_res)
    condInt_vec = condInt * np.ones_like(lags_res)
    plt.plot(lags_res, condInt_vec, "--", color="red", linewidth=0.8)
    plt.plot(lags_res, -condInt_vec, "--", color="red", linewidth=0.8)
    ax = plt.gca()
    caption = "Dashed red: 95% Confidence Interval"
    ax.text(
        0.98,
        0.98,
        caption,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # fhat is your residual vector
    # Create a styled normal probability (QQ) plot consistent with other figures
    (osm, osr), (slope, intercept, r) = stats.probplot(ACF_res, dist="norm", plot=None)
    plt.figure(figsize=(8, 3))
    plt.plot(
        osm,
        osr,
        marker="o",
        markersize=3,
        markerfacecolor="none",
        markeredgecolor="C1",
        linestyle="None",
        color="C1",
    )
    # Fit line
    line_x = np.array([np.min(osm), np.max(osm)])
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, color="C2", linewidth=0.8)
    plt.xlabel("Theoretical quantiles")
    plt.ylabel("Sample quantiles")
    plt.title("Normal Probability Plot")
    plt.grid(True)
    plt.tight_layout()
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
    ACF_raw = acf(svedala_rec, 999, plotIt=False)
    print("Length of original series:", len(svedala_rec))
    print("Length of ACF_raw:", len(ACF_raw))

    # Remove daily periodicity (24h)
    filtered_svedala = apply_seasonal_filter(svedala_rec, 24)

    # ACF of filtered series
    ACF_filt = acf(filtered_svedala, 999, plotIt=False)

    # Plots for Q2.1 (original + filtered + both ACFs)
    plot_q21(
        original=svedala_rec,
        filtered=filtered_svedala,
        acf_raw=ACF_raw,
        acf_filt=ACF_filt,
    )

    # Kalman filter for Q2.2 / Q2.3
    Xtt, Ytt1, ft_hat = kalman(filtered_svedala, se2=8e-3, sf2=85)
    plot_kalman_results(Xtt, Atrue, ft_hat)
