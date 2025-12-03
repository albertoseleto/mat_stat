import numpy as np
def kalman(series, ):
    # N = length(??)
    N = len(series)

    se2 = 10 # estimate  / measure      # measurement noise variance 2x2 shaped?
    sf2 = 0.6 # how much the model will change, estimate it      # state noise variance

    # Xtt = zeros(??, N)
    Xtt = np.zeros((2, N))

    Ytt1 = np.zeros(N)        # predicted measurements
    VYYtt1 = np.zeros(N)      # predicted measurement variances

    lag = 2                 # initial lag for state

    # Xtt(:,1:lag) = repmat([??]', 1, lag)
    Xtt[:, :lag] = np.tile(np.array([0.0, 0.0]).reshape(-1, 1), (1, lag))

    # VXXtt = diag([??])
    VXXtt = np.diag([??])  # initial covariance matrix

    # A = eye(??)
    A = np.eye(2)


    for k in range(lag, N):
        # C = [??]
        C = np.array([??]).reshape(1, -1)

        # ---------- PREDICTION ----------
        # Xtt1 = A * Xtt(:,k-1)
        Xtt1 = A @ Xtt[:, k-1]

        # VXXtt1 = A * VXXtt * A' + sf2 * I
        VXXtt1 = A @ VXXtt @ A.T + sf2 * np.eye(A.shape[0])

        # Ytt1(k) = C * Xtt1
        Ytt1[k] = C @ Xtt1

        # VYYtt1(k) = C * VXXtt1 * C' + se2
        VYYtt1[k] = C @ VXXtt1 @ C.T + se2

        # ---------- UPDATE ----------
        # CXYtt1 = VXXtt1 * C'
        CXYtt1 = VXXtt1 @ C.T

        # Kt = CXYtt1 / VYYtt1(k)
        Kt = CXYtt1 / VYYtt1[k]

        # Xtt(:,k) = Xtt1 + Kt * (y(k) - Ytt1(k))
        Xtt[:, k] = Xtt1 + Kt.flatten() * (??[k] - Ytt1[k])

        # VXXtt = VXXtt1 - Kt * C * VXXtt1
        VXXtt = VXXtt1 - Kt @ C @ VXXtt1
    return 