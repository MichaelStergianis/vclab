#! /usr/bin/env python3

import os
import DataGenerator
import random

class DataImport:
    def __init__(self):
        self.arrays = []
        self.labels = []

    def parse_whole(self, file):
        with open(file, "r") as f:
            for line in f:
                line = line[:-1].split(",")
                assert len(line) == 5
                num_list = []
                for i in line[:-1]:
                    num_list.append(int(i))
                self.arrays.append(num_list)
                label = []
                for i in range(1024):
                    label.append(0)
                label[ (int(line[-1]) - 1) ] = 1
                self.labels.append(label)

    """
    _bitwise

    Returns a list of the binary values of the number starting with the least
    significant bit. For now it is only useable for values up to 255, but that
    is intentional given our dataset

    number -- the number to be put into bitwise fashion
    """
    def _bitwise(self, number):
        ret_list = []
        for i in range(0, 8):
            temp = number & 2**i
            temp = temp >>i
            ret_list.append(temp)
        return ret_list

    
    def parse_bitwise(self, file):
        with open(file, "r") as f:
            for line in f:
                line = line[:-1].split(",")
                assert len(line) == 5
                num_list = []
                for i in line[:-1]:
                    num_list.append(self._bitwise(int(i)))
                self.arrays.append(num_list)
                label = []
                for i in range(1021):
                    label.append(0)
                label[ (int(line[-1]) - 1) ] = 1
                self.labels.append(label)


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
