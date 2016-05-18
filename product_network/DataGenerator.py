#! /usr/bin/env python3
import numpy
import random
import time
import os

"""
    label_function

    This function determines the value for the label. In this example we are
    going to generate labels on these assumptions
    x4 != 0
    return_value = (x1 * x2 * weight) - (x3 / x4)
"""
def label_function(elems, weight):
    return ( elems[0] * elems[1] )
    #return ( (elems[0] * elems[1] * weight) - (elems[2] / elems[3]) )


def file_write(f, weights, offset):
    sum = 0
    elems = []
    for i in range(4):
        temp = random.uniform(0, 255)
        # x4 cannot be zero
        while temp == 0.0:
            temp = random.uniform(0, 255)
        elems.append(temp)
        s = str(temp)
        f.write(s)
        f.write(",")
    s = str(label_function(elems, 1))
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
    offset = 12
    
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
