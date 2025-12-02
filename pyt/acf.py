import numpy as np
from scipy.stats import norm
from scipy.signal import correlate
import matplotlib.pyplot as plt

def acf(y, maxOrd, signLvl=0.05, plotIt=False, maOrder=0, includeZeroLag=True):
    """
    Python version of the MATLAB ACF estimator.

    Parameters
    ----------
    y : array_like
        Input signal.
    maxOrd : int
        Maximum lag.
    signLvl : float, optional
        Significance level for CI (default 0.05).
    plotIt : bool, optional
        Whether to plot the ACF and confidence interval.
    maOrder : int, optional
        MA order assumption (default 0).
    includeZeroLag : bool, optional
        Whether to include lag 0 in the plot.

    Returns
    -------
    rho : ndarray
        The ACF values from lag 0 to maxOrd.
    """

    y = np.asarray(y)

    if maOrder > maxOrd:
        raise ValueError("maOrder cannot be larger than maxOrd")
    if not (0 <= signLvl <= 1):
        raise ValueError("signLvl must be between 0 and 1")

    # Gaussian significance scale
    signScale = norm.ppf(1 - signLvl / 2)

    # Compute biased autocorrelation using numpy
    y_centered = y - np.mean(y)
    N = len(y)

    # full correlation
    rho_full = correlate(y_centered, y_centered, mode="full") / N
    lags = np.arange(-N + 1, N)

    # Keep only lags [0:maxOrd]
    mid = len(rho_full) // 2
    rho = rho_full[mid:mid + maxOrd + 1]

    # Normalize like MATLAB: rho/max(rho)
    rho = rho / np.max(rho)

    # ---- Plotting ----
    if plotIt:
        if includeZeroLag:
            range_lags = np.arange(0, maxOrd + 1)
            rho_plot = rho
            max_range = 1.1
            start_lag = 0
        else:
            range_lags = np.arange(1, maxOrd + 1)
            rho_plot = rho[1:]
            max_range = np.max(np.abs(rho_plot)) * 1.2
            start_lag = 1

        plt.stem(range_lags, rho_plot, use_line_collection=True)
        plt.xlabel("Lag")
        plt.ylabel("Amplitude")

        # Confidence interval
        condInt = signScale / np.sqrt(N)
        condInt *= np.sqrt(1 + 2 * np.sum(rho[:maOrder]**2))
        condInt_vec = condInt * np.ones_like(range_lags)

        plt.plot(range_lags, condInt_vec, "--")
        plt.plot(range_lags, -condInt_vec, "--")

        if (not includeZeroLag) and max_range < condInt:
            plt.axis([start_lag, maxOrd, -condInt*1.2, condInt*1.2])
        else:
            plt.axis([start_lag, maxOrd, -max_range, max_range])

        plt.show()

    return rho
