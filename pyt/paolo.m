%% Load data
load('svedala.mat');  % loads svedala_rec and Atrue

% Use seasonally adjusted series
y = filter([1 zeros(1,s-1) -1], 1, svedala_rec);

N = length(y);

%% ----- PARAMETERS -----
lag = 2;                % AR(2)
Xtt = zeros(2,N);       % state estimates
Ytt1 = zeros(1,N);      % predicted observations
VYYtt1 = zeros(1,N);    % predicted observation variance

% Process noise variance (Σe) controls time-varying vs constant
se2 = 0.000003;             % small -> nearly constant, larger -> time-varying
sf2 = 0.1;             % observation noise variance

% Initialize state and covariance
Xtt(:,1:lag) = repmat([0;0],1,lag);
VXXtt = diag([0.1 0.1]);
A = eye(2);             % state transition matrix

%% ----- KALMAN FILTER LOOP -----
for k = lag+1:N
    C = [-y(k-1) -y(k-2)];
    
    % PREDICTION
    Xtt1 = A*Xtt(:,k-1);
    VXXtt1 = A*VXXtt*A' + se2*eye(2);
    Ytt1(k) = C*Xtt1;
    VYYtt1(k) = C*VXXtt1*C' + sf2;
    
    % UPDATE
    CXYtt1 = VXXtt1*C';
    Kt = CXYtt1 / VYYtt1(k);
    Xtt(:,k) = Xtt1 + Kt*(y(k) - Ytt1(k));
    VXXtt = VXXtt1 - Kt*VYYtt1(k)*Kt';
end

%% ----- Compare with true parameters -----
a1_true = Atrue(:,1);
a2_true = Atrue(:,2);

figure;
subplot(2,1,1)
plot(1:N, Xtt(1,:), 'b', 'LineWidth',1.2); hold on
plot(1:N, a1_true, 'r--', 'LineWidth',1.2);
xlabel('Time (hours)')
ylabel('a_1(t)')
legend('Estimated','True')
title('Comparison of a_1(t)')
grid on

subplot(2,1,2)
plot(1:N, Xtt(2,:), 'b', 'LineWidth',1.2); hold on
plot(1:N, a2_true, 'r--', 'LineWidth',1.2);
xlabel('Time (hours)')
ylabel('a_2(t)')
legend('Estimated','True')
title('Comparison of a_2(t)')
grid on

%% ----- Compute residuals -----
ft_hat = y(lag+1:end) - Ytt1(lag+1:end);
ft_hat = ft_hat(:);                     % ensure column vector
t_res = (1:length(ft_hat))';            % x-axis vector matches ft_hat

% Plot residuals
figure;
plot(t_res, ft_hat)
xlabel('Time (hours)')
ylabel('\hat{f}_t','Interpreter','latex')
title('Estimated residuals')
grid on

%% ----- Plot ACF of residuals -----
maxLag = 200;  % reduce memory usage
[rho,lags] = xcorr(ft_hat-mean(ft_hat), maxLag, 'coeff');

figure;
stem(lags, rho, 'filled')
xlabel('Lag (hours)')
ylabel('Autocorrelation')
title('ACF of residuals')
grid on