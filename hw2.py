import math
import numpy as np
from time import time
from collections import Counter

class Data:
    def __init__(self):
        self.features = []	# list of lists (size: number_of_examples x number_of_features)
        self.labels = []	# list of strings (lenght: number_of_examples)

def read_data(path):
    
    data = Data()
    traininglist = []
    training_file = open(path, 'r')
    for x in training_file:
        traininglist.append(x)
    training_file.close()
    for i in range(len(traininglist)):
        data.features.append([])
        line_in_traininglist = traininglist[i].split(",")
        data.features[i].append(float(line_in_traininglist[0]))
        data.features[i].append(float(line_in_traininglist[1]))
        data.features[i].append(float(line_in_traininglist[2]))
        data.features[i].append(float(line_in_traininglist[3]))
        data.labels.append(line_in_traininglist[4].rstrip())
        if(data.labels[i] == "Iris-setosa"):
            data.labels[i] = 1
        else:
            data.labels[i] = 0

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
    return np.dot(u,v)

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
        converged = False
        self.MissedPoints.append(data.features[0])
        self.MissedLabels.append(data.labels[0])
        while converged == False:
            converged = True
            #print(self.MissedPoints)
            for i in range(len(data.labels)):
                if(self.update(data.features[0], data.labels[i]) == True):
                    converged = False
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
        if(self.predict(point) != label):
            self.MissedLabels.append(label)
            self.MissedPoints.append(point)
            is_mistake = True
        else:
            is_mistake = False
        return is_mistake

    def getScore(self, point):
        score = 0
        for x in range(len(self.MissedLabels)):
            score =  score + self.MissedLabels[x]*self.kf(point, self.MissedPoints[x])
        return score



    def predict(self, point):
        """
        Given a point, predicts the label of that point (1 or -1).
        """
        score = self.getScore(point)
        if score > 0:
            label = 1
        else:
            label = 0
        # TODO
        return label

    def test(self, data):
        predictions = []
        point = [0, 0, 0, 0]
        # TODO: given data and a perceptron - return a list of integers (+1 or -1).
        # +1 means it is Iris Setosa
        # -1 means it is not Iris Setosa
        for x in range(len(data.labels)):
            predictions.append(self.predict(data.features[x]))

        return predictions


# Feel free to add any helper functions as needed.
def main():
    p = Perceptron(dot_kf, 0)
    traindata = Data()
    testdata = Data()
    traindata = read_data("hw2_train.txt")
    testdata = read_data("hw2_test.txt")
    p.train(traindata)
    result = []
    result = test(traindata)
    correct = 0
    total = 0
    for i in range(len(traindata)):
        if result[i] == traindata.labels[i]:
            correct = correct + 1
        total = total + 1
    print(correct/total)
    return
    



if __name__ == '__main__':
    main()