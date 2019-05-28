clc
clear
N = 3257;
M = importfile('./force/video_data.csv', 2, N);
a = table2array(M);
x = smooth(a(:,1),'moving');
y = smooth(a(:,2),'moving');
t = 1:N - 1;
t = t';
plot3(t,x,y)
