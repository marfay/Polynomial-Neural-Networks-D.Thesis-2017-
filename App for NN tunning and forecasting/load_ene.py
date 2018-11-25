import numpy as np
def load_data():
    date = []
    time = []
    i = 0
    with open('energo_time.txt', 'r') as f:
        for line in f:
            data = line.split()
            time.append(data[0])
            i+=1
    i=0
    with open('energo_date.txt', 'r') as f:
        for line in f:
            data = line.split()
            date.append(data[0])
            i += 1

    return date,time
