%import  table2array
clc;
% figure();
% plot(x)
y=fft(x);
n = length(y);
power = abs(y(1:floor(n/2))).^2; % power of first half of transform data
maxfreq = 1/2;                   % maximum frequency
freq = (1:n/2)/(n/2)*maxfreq;    % equally spaced frequency grid
period = 1./freq;
% figure();
% plot(period,power,'bo');
%%×ÓÐòÁÐ·Ö¶Î
T = 188;
% z = smooth(x);
z = x;
wi =int16( T + 0.06*T );
i = 1;
maxx=zeros(1,length(z));
maxy=zeros(1,length(z));
while i < length(z) - wi 
    [m,p] = max(z(i:i+wi));
    maxx(p+i-1) = p+i;
    maxy(p+i-1) = m;
    i = wi + i;
end
t = [1:1:length(z)];
Ix=find(maxx~=0);
Iy=maxy(Ix);
figure();
plot(t./60,z)
hold on
plot(Ix./60,Iy,'rx')

figure()
plot(smooth([Ix(2:length(Ix))-Ix(1:length(Ix)-1)]./60))