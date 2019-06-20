%import  table2array
figure();
plot(x)
y=fft(x);
n = length(y);
power = abs(y(1:floor(n/2))).^2; % power of first half of transform data
maxfreq = 1/2;                   % maximum frequency
freq = (1:n/2)/(n/2)*maxfreq;    % equally spaced frequency grid
period = 1./freq;
figure();
plot(period,power,'bo');