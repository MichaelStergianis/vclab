#! /usr/bin/env python3

import os
import DataGenerator
import random

class DataImport:
    def __init__(self):
        self.arrays = []
        self.labels = []

    def parse(self, file):
        with open(file, "r") as f:
            for line in f:
                line = line[:-1].split(",")
                assert len(line) == 5
                self.arrays.append([[int(line[0]), int(line[1])], 
                                    [int(line[2]), int(line[3])]])
                self.labels.append(int(line[4]))

    def next_batch(self, size):
        # randomly select size numbers to provide
        # return two lists as a tuple
        arrayRetList = []
        labelsRetList = []
        listSize = len(self.arrays)
        assert size <= listSize
        for i in range(size):
            index = random.randint(0, listSize)
            arrayRetList.append(self.arrays[index])
            labelsRetList.append(self.labels[index])
        return (arrayRetList, labelsRetList)

    def verify(self):
        assert len(self.arrays) == len(self.labels)
        for i in range(len(self.arrays)):
            sum = 0
            for j in range(2):
                for k in range(2):
                    sum += self.arrays[i][j][k]
            if sum != self.labels[i]:
                print("Array does not match sum")

    def printIndex(self, index):
        print("Array = [["  + str(self.arrays[index][0][0]) + ", " + str(self.arrays[index][0][1])
            + "\n         " + str(self.arrays[index][1][0]) + ", " + str(self.arrays[index][1][1])
            + "]]" + "\nsum = " + str(self.labels[index]))

class TrainData:
    def __init__(self):
        if not os.path.isdir("data"):
            DataGenerator.main(55000, 10000)
        self.d = DataImport()
        self.d.parse("data/ran_nums_train.csv")
        self.d.verify()

    def next_batch(self, size):
        return self.d.next_batch(size)

class TestData:
    def __init__(self):
        if not os.path.isdir("data"):
            DataGenerator.main(55000, 10000)
        d = DataImport()
        d.parse("data/ran_nums_test.csv")
        d.verify()

    def next_batch(self, size):
        return self.d.next_batch(size)
