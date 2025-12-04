import numpy as np
import numpy as np

def kalman(y, se2=0.000003, sf2=0.1, lag=2):
    """
    Kalman filter for time-varying AR(2) model.
    Returns estimated states, predicted observations, and residuals.
    """
    y = np.asarray(y).flatten()
    N = len(y)
    Xtt = np.zeros((2, N))
    Ytt1 = np.zeros(N)
    VYYtt1 = np.zeros(N)
    VXXtt = np.diag([0.1, 0.1])
    A = np.eye(2)

    # Initialize first 'lag' states
    Xtt[:, :lag] = np.tile(np.array([0.0, 0.0]).reshape(-1, 1), (1, lag))

    for k in range(lag, N):
        C = np.array([-y[k-1], -y[k-2]]).reshape(1, -1)
        # Prediction
        Xtt1 = A @ Xtt[:, k-1]
        VXXtt1 = A @ VXXtt @ A.T + se2 * np.eye(2)
        Ytt1[k] = C @ Xtt1
        VYYtt1[k] = C @ VXXtt1 @ C.T + sf2
        # Update
        CXYtt1 = VXXtt1 @ C.T
        Kt = CXYtt1 / VYYtt1[k]
        Xtt[:, k] = Xtt1 + Kt.flatten() * (y[k] - Ytt1[k])
        VXXtt = VXXtt1 - Kt @ C @ VXXtt1

    # Residuals
    ft_hat = y[lag:] - Ytt1[lag:]
    return Xtt, Ytt1, ft_hat