#! /usr/bin/env python3
import numpy
import random
import time
import os

if not os.path.isdir("./data"):
    os.mkdir("./data")

random.seed(time.clock())

# create training data
with open("data/ran_nums_train.csv", "w") as f:
    for i in range(55000):
        sum = 0
        for i in range(4):
            temp = random.randint(0, 255)
            sum += temp
            s = str(temp)
            f.write(s)
            f.write(",")
        s = str(sum)
        f.write(s)
        f.write("\n")

# create testing data
with open("data/ran_nums_test.csv", "w") as f:
    for i in range(10000):
        sum = 0
        for i in range(4):
            temp = random.randint(0, 255)
            sum += temp
            s = str(temp)
            f.write(s)
            f.write(",")
        s = str(sum)
        f.write(s)
        f.write("\n")

