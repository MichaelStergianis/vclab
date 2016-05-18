#! /usr/bin/env python3

import os
import DataGenerator
import random
import copy
import numpy as np

class DataImport:
    def __init__(self):
        self.arrays = 0
        self.labels = 0

    def parse_whole(self, file):
        with open(file, "r") as f:
            array = []
            lbls = []
            for line in f:
                line = line[:-1].split(",")
                assert len(line) == 5
                num_list = []
                for i in line[:-1]:
                    num_list.append(float(i))
                array.append(num_list)
                label = []
                ## # make one of many labels 1 this is a naive way of labeling
                ## for i in range(1024):
                ##     label.append(0)
                ## label[ (float(line[-1]) - 1) ] = 1.0
                label.append(float(line[-1]))
                lbls.append(label)
            self.labels = np.array(copy.deepcopy(lbls), dtype=np.float32)
            self.arrays = np.array(copy.deepcopy(array), dtype=np.float32)

    def parse_bitwise(self, file):
        with open(file, "r") as f:
            array = []
            lbls  = []
            for line in f:
                line = line[:-1].split(",")
                assert len(line) == 5
                num_list = []
                for i in line[:-1]:
                    for j in self._bitwise(float(i), 8):
                        num_list.append(j)
                array.append(num_list)
                label = []
                for i in self._bitwise(float(line[-1]), 10):
                    label.append(i)
                ### label.append(float(line[-1]))
                lbls.append(label)
#                label = np.zeros((1021,), dtype=np.float32)
#                label[ (float(line[-1]) - 1) ] = 1.0
#                lbls.append(label)
            self.labels = np.array(lbls, dtype=np.float32)
            self.arrays = np.array(array, dtype=np.float32)
    """
    _bitwise

    Returns a list of the binary values of the number starting with the least
    significant bit. Only works on integers

    number -- the number to be put into bitwise fashion
    """
    def _bitwise(self, number, num_bits):
        ret_list = []
        for i in range(0, num_bits):
            temp = number & 2**i
            temp = temp >>i
            ret_list.append(float(temp))
        return ret_list

    def next_batch(self, size):
        # randomly select size numbers to provide
        # return two lists as a tuple
        arrayRetList = []
        labelsRetList = []
        listSize = len(self.arrays)
        assert size <= listSize
        for i in range(size):
            index = random.randint(0, listSize-1)
            arrayRetList.append(self.arrays[index])
            labelsRetList.append(self.labels[index])
        return (arrayRetList, labelsRetList)

    def verify(self):
        assert len(self.arrays) == len(self.labels)
        for i in range(len(self.arrays)):
            sum = 0
            for j in range(len(self.arrays[0])):
                sum += self.arrays[i][j]
            if self.labels[i][sum - 1] != 1:
                print("Array does not match label")


    def printIndex(self, index):
        print("Array = [["  + str(self.arrays[index][0][0]) + ", " + str(self.arrays[index][0][1])
            + "\n         " + str(self.arrays[index][1][0]) + ", " + str(self.arrays[index][1][1])
            + "]]" + "\nsum = " + str(self.labels[index]))

"""
    AdvancedDataImport

    The purpose of this class will be to import data in much the same way as DataImport
    with one main difference. It will append rows to the arrays tensor performing operations
    on the original values.
"""
class AdvancedDataImport(DataImport):
    def __init__(self):
        DataImport.__init__(self)

    def parse_whole(self, f):
        DataImport.parse_whole(self, f)
        # t.arr will be the copy of t.arrays that we store, we will then remake t.arrays
        self.arr = self.arrays
        self.arrays = np.zeros((self.arr.shape[0], 9 * self.arr.shape[1]))
        for i in range(self.arr.shape[0]):
            for j in range(self.arr.shape[1]):
                self.arrays[i, j] = self.arr[i, j]
        self.arr = None
        for i in self.arrays:
            # find multiplicative values
            for j in range(1, 5):
                for k in range(j-1, 4):
                    i[j*4+k] = i[k] * i[j-1]
            # do the same for divisive
            for j in range(5, 9):
                for k in range(0, 4):
                    i[j*4+k] = i[k] / i[j-5]
        for i in self.arrays:
            for j in range(36):
                if not (j == 5 or j == 34):
                    i[j] = 0.0

class TrainData(DataImport):
    def __init__(self, bitwise=False):
        DataImport.__init__(self)
        if bitwise:
            self.parse_bitwise("data/ran_nums_train.csv")
        else:
            self.parse_whole("data/ran_nums_train.csv")

class TestData(DataImport):
    def __init__(self, bitwise=False):
        DataImport.__init__(self)
        if bitwise:
            self.parse_bitwise("data/ran_nums_test.csv")
        else:
            self.parse_whole("data/ran_nums_test.csv")

class AdvancedTrainData(AdvancedDataImport):
    def __init__(self):
        AdvancedDataImport.__init__(self)
        self.parse_whole("data/ran_nums_train.csv")

class AdvancedTestData(AdvancedDataImport):
    def __init__(self):
        AdvancedDataImport.__init__(self)
        self.parse_whole("data/ran_nums_test.csv")


