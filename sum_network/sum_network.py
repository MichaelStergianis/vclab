#! /usr/bin/env python3

import tensorflow as tf
import numpy
from data_type import DataType as dt

# if no data exists, let's create it for now
# will probably delete this later for more flexibility
if not os.path.isdir("./data"):
    os.system("sum_gen.py")

# read in the training data
training_data_list = []
with open("data/ran_nums_train.csv") as f:
    while line = f.readline():
        line = line.split(",")
	training_data_list.append(dt([line[0], line[1], line[2], line[3]], line[4]))

print(training_data_list[1])
