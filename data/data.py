import pandas as pd
import csv


l = []
title = ['timestamp', 'value']
i = 0
with open('./force/video_data.csv', 'rt') as f:
    cr = csv.reader(f)
    for row in cr:
        row[0] = i + 1540796400
        row[1] = i
        row[2] = float(row[2])
        row[3] = float(row[3])
        i += 1
        l.append(row)
    print(l)
with open('./force/x.csv','wt') as f2:
    cw = csv.writer(f2)
    cw.writerow(title)
    for item in l:
        cw.writerow(item[1:3])
with open('./force/y.csv','wt') as f3:
    cw = csv.writer(f3)
    cw.writerow(title)
    for item in l:
        cw.writerow(item[1:4:2])
with open('./force/x_timeseries.csv','wt') as f4:
    cw = csv.writer(f4)
    cw.writerow(title)
    for item in l:
        cw.writerow(item[0:3:2])
with open('./force/y_timeseries.csv','wt') as f5:
    cw = csv.writer(f5)
    cw.writerow(title)
    for item in l:
        cw.writerow(item[0:4:3])