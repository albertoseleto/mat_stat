% skeleton code for Kalman filter

N = length(??);
se2 = ??;
sf2 = ??;

Xtt = zeros(??,N);
Ytt1 = zeros(1,N);
VYYtt1 = zeros(1,N);
lag=??;
Xtt(:,1:lag) = repmat([??]',1,lag);

VXXtt = diag([??]);
A = eye(??);


for k=lag+1:N
    C = [??];
    
    %predict
    Xtt1 = ??;
    VXXtt1 = ??;
    Ytt1(k) = ??;
    VYYtt1(k) = ??;

    %update
    CXYtt1 = ??;
    Kt = ??;
    Xtt(:,k) = ??;
    VXXtt = ??;

end