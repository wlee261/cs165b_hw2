#!/usr/bin/env python3

import math
#import numpy as np
from time import time
from collections import Counter

class Data:
    def __init__(self):
        self.features = []	# list of lists (size: number_of_examples x number_of_features)
        self.labels = []	# list of strings (lenght: number_of_examples)

def read_data(path):
    
    data = Data()
    feature1 = []
    feature2 = []
    feature3 = []
    feature4 = []
    traininglist = []
    training_file = open(path, 'r')
    for x in training_file:
        traininglist.append(x)
    training_file.close()
    for i in range(len(traininglist)):
        line_in_traininglist = traininglist[i].split(",")
        feature1.append(line_in_traininglist[0])
        feature2.append(line_in_traininglist[1])
        feature3.append(line_in_traininglist[2])
        feature4.append(line_in_traininglist[3])
        data.labels.append(line_in_traininglist[4].rstrip())
    
    data.features.append(feature1)
    data.features.append(feature2)
    data.features.append(feature3)
    data.features.append(feature4)
    # TODO: function that will read the input file and store it in the data structure
    # use the Data class defined above to store the information
    return data

def dot_kf(u, v):
    """
    The basic dot product kernel returns u*v.

    Args:
        u: list of numbers
        v: list of numbers

    Returns:
        u*v
    """
    # TODO: implement the kernel function
    return dot(u,v)

def poly_kernel(d):
    """
    The polynomial kernel.

    Args:
        d: a number

    Returns:
        A function that takes two vectors u and v,
        and returns (u*v+1)^d.
    """
    def kf(u, v):
        # TODO: implement the kernel function
        return 
    return kf

def exp_kernel(s):
    """
    The exponential kernel.

    Args:
        s: a number

    Returns:
        A function that takes two vectors u and v,
        and returns exp(-||u-v||/(2*s^2))
    """
    def kf(u, v):
        # TODO: implement the kernel function
        return
    return kf

class Perceptron:
    def __init__(self, kf, lr):
        """
        Args:
            kf - a kernel function that takes in two vectors and returns
            a single number.
        """
        self.MissedPoints = []
        self.MissedLabels = []
        self.kf = kf
        self.lr = lr

    def train(self, data):
        # TODO: Main function - train the perceptron with data
        errors = 0
        converged = false
        while converged = false:
            for i in range(len(data.labels)):
                if(
        return
        

    def update(self, point, label):

        """
        Updates the parameters of the perceptron, given a point and a label.

        Args:
            point: a list of numbers
            label: either 1 or -1

        Returns:
            True if there is an update (prediction is wrong),
            False otherwise (prediction is accurate).
        """
        # TODO
        return is_mistake

    def predict(self, point):
        """
        Given a point, predicts the label of that point (1 or -1).
        """
        # TODO
        return

    def test(self, data):
        predictions = []
        # TODO: given data and a perceptron - return a list of integers (+1 or -1).
        # +1 means it is Iris Setosa
        # -1 means it is not Iris Setosa
        return predictions


# Feel free to add any helper functions as needed.
def main():
    return
    



if __name__ == '__main__':
    main()
    
