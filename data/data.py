import pandas as pd
import csv


l = []
with open('video_data.csv', 'rt') as f:
    cr = csv.reader(f)
    for row in cr:
        row [1] = int(row[1]) + 1540796400
        l.append(row)  # 将test.csv内容读入列表l，每行为其一个元素，元素也为list
    print(l)
with open('1.csv','wt') as f2:
   cw = csv.writer(f2)
   #采用writerow()方法
   for item in l:
      cw.writerow(item) #将列表的每个元素写到csv文件的一行