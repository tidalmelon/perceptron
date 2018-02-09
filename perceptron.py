# -*- coding: utf-8 -*-
import numpy as np

ITERATE_NUM = 0
ALPHA = 0.0
FEATURE_NUM = 0
SAMPLE_NUM = 0
WEIGHTS = []
X = []
Y = []

    
def LoadData():
    global ITERATE_NUM
    global ALPHA
    global FEATURE_NUM
    global SAMPLE_NUM
    global WEIGHTS
    global X
    global Y

    fname = 'train.dat'
    with open(fname) as f:
        line = f.readline().strip()
        arr = line.split()
        FEATURE_NUM = int(arr[0]) + 1 # 偏移特征
        SAMPLE_NUM = int(arr[1])
        ALPHA = float(arr[2])
        ITERATE_NUM = int(arr[3])
        
        line = f.readline().strip()
        arr = line.split()
        arr = [float(e) for e in arr]
        WEIGHTS = np.array(arr)

        t = []
        y = []
        for i in range(SAMPLE_NUM):
            line = f.readline().strip()
            arr = line.split()
            arr = [float(e) for e in arr]
            arr.insert(0, 1.0)
            t.append(arr[0: -1])
            y.append(int(arr[-1]))
            
        X = np.array(t)
        Y = np.array(y)

def PerceptronTrain():
    global WEIGHTS
    global X
    global Y
    for i in range(ITERATE_NUM):
        # 激活函数计算
        output = np.sum(X * WEIGHTS, axis=1)
        output = np.array(map(lambda x: 1 if x>0 else 0, output))
        
        # 计算权重的增加量detal:  w_new = w + delta = w + alpha * (y - output) * x
        delta = ALPHA * (Y - output).reshape(SAMPLE_NUM, 1) * X
        delta = np.sum(delta, axis=0)

        # 更新参数
        WEIGHTS += delta
        print WEIGHTS


if __name__ == '__main__':
    LoadData()
    PerceptronTrain()
