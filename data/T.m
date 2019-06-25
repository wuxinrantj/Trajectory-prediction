clc;
y = smooth(x,51);
plot(y)
hold on
plot(find(diff(sign(diff(y)))==-2)+1,y(find(diff(sign(diff(y)))==-2)+1),'*r')
hold off
