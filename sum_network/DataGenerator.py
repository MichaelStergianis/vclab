#! /usr/bin/env python3
import numpy
import random
import time
import os

def file_write(f, weights, offset):
    sum = 0
    for i in range(4):
        temp = random.randint(0, 255)
        sum += (temp * weights[i])
        s = str(temp)
        f.write(s)
        f.write(",")
    s = str(sum)
    f.write(s)
    f.write("\n")

def file_write_prod(f):
    sum = 0
    nums = []
    for i in range(4):
        nums.append(random.randint(0, 255))
        s = str(nums[i])
        f.write(s)
        f.write(",")
    for i in range(3):
        sum += (nums[i] * nums[i+1])
    s = str(sum)
    f.write(s)
    f.write("\n")


def main(trainNum, testNum):
    if not os.path.isdir("./data"):
        os.mkdir("./data")
    
    random.seed(time.clock())

    weights = [1,1,1,1]
    offset = 0
    
    # create training data
    with open("data/ran_nums_train.csv", "w") as f:
        for i in range(trainNum):
            file_write(f, weights, offset)
            #file_write_prod(f)
    
    # create testing data
    with open("data/ran_nums_test.csv", "w") as f:
        for i in range(testNum):
            file_write(f, weights, offset)
            #file_write_prod(f)
    
main(55000, 10000)
