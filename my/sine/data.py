import pandas as pd
import csv
import xlwt
import pandas as pd
import numpy as np
from datetime import datetime

l = []
title = ['Month', '#Passengers']
i = 0
j = 0
string = []
with open('date.csv','wt') as f2:
    cw = csv.writer(f2)
    cw.writerow(title)
    for i in range(1949,2249,1):
        for j in range(1,13,1):
            string.append(''.join(str(i) + '-' + str(j)))
            cw.writerow(''.join(str(i) + '-' + str(j)))

dt2 = [datetime.strftime(x,'%Y-%m') for x in list(pd.date_range(start=194901, end=224901, freq='M'))]
print(dt2)