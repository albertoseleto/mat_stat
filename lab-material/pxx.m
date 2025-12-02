function [psd, freq] = pxx(x, dt)
% compute a non-parametric estimate of the power spectral density of x 

x = x - mean(x);
N = length(x); 

psd = fftshift(abs(fft(x)).^2) / N;

fsample = 1 / dt; 

freq = fsample * linspace(-0.5, 0.5, N);

end