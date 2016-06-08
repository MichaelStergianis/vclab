import scipy.io as sio
import numpy as np
import random

class load_mat:
    def __init__(self, file_name):
        s = sio.loadmat(file_name)
        self.x = np.array(s['X'], dtype=np.float32)
        self.y = np.zeros(shape=[len(s['y']), 10], dtype=np.float32)
        for i in range(len(s['y'])):
            self.y[i, (s['y'][i]-1)] = 1.0

    def next_batch(self, size):
        arrayRetList = []
        labelsRetList = []
        listSize = self.x.shape[0]
        assert size <= listSize
        for i in range(size):
            index = random.randint(0, listSize-1)
            arrayRetList.append(self.x[index])
            labelsRetList.append(self.y[index])
        return (arrayRetList, labelsRetList)

    def get_data(self):
        return (self.x, self.y)
