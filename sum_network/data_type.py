#! /usr/bin/env python3

import numpy as np
import tensorflow as tf

class DataType:
    def __init__(self, nums, label):
        assert(len(nums) == 4)
        assert(type(label) == int)
        self.nums = tf.constant(nums, shape=(2,2))
        self.label = label
    def __str__(self):
        return str(self.nums)
