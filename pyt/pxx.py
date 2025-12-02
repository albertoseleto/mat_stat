import numpy as np

def pxx(x, dt):
    """
    Compute a non-parametric estimate of the power spectral density (PSD)
    of signal x, matching the MATLAB function.

    Parameters
    ----------
    x : array_like
        Input signal.
    dt : float
        Sampling interval.

    Returns
    -------
    psd : ndarray
        Power spectral density, FFT-shifted.
    freq : ndarray
        Corresponding frequency axis (Hz), centered at zero.
    """
    x = np.asarray(x)
    x = x - np.mean(x)
    N = len(x)

    # PSD = |FFT|^2 / N  (fftshift for centered spectrum)
    psd = np.fft.fftshift(np.abs(np.fft.fft(x))**2) / N

    # Sampling frequency
    fsample = 1.0 / dt

    # Frequency axis from -fs/2 to +fs/2
    freq = fsample * np.linspace(-0.5, 0.5, N)

    return psd, freq
