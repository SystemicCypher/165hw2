# Starter code for CS 165B HW2
import numpy as np

# Utility functions
def centroid_calculator(train_data, feature_count, label, feature_size):
    # Initializations
    point_sum = [0.0] * feature_count
    centroid = [0.0] * feature_count

    start = 0 + feature_size[label] * label
    end = feature_size[label] * (label + 1)

    # Calculation
    # Sum of every point 
    for i in range(start, end):
        for j in range(0, feature_count):
            point_sum[j] += train_data[i][j]

    # Mean point - "centroid"
    for i in range(0, feature_count):
        centroid[i] = point_sum[i]/feature_size[label]
    return centroid

def coordSub(pointA, pointB, size, midflag):
    coSum = [0.0] * size
    for x in range(size):
        coSum[x] = pointA[x] - pointB[x]
        if midflag == 1:
            coSum[x] = coSum[x]/2.0
    return coSum

def coordSq(point):
    summ = 0.0
    for value in point:
        summ += value**2
    return summ

def vecMult(w, x):
    output = 0.0
    for i in range(len(w)):
        output += w[i] * x[i]
    return output


# Uses the classifier
def classify(point, classif_w, classif_bias):    
    A = False
    B = False
    C = False

    if (vecMult(classif_w[0], point) + classif_bias[0]) >= 0:
        A = True
    else:
        B = True
    
    if A:
        if (vecMult(classif_w[1], point) + classif_bias[1]) >= 0:
            A = True
        else:
            C = True
            A = False
    else:
        if (vecMult(classif_w[2], point) + classif_bias[2]) >= 0:
            B = True
        else:
            C = True
            B = False

    if A:
        y_hat = "A"
    elif B:
        y_hat = "B"
    else:
        y_hat = "C"
    return y_hat

# Trains and creates the classifier
def basic_linear_classifier_train(train_data, feature_count_train, feature_size_train):
    centroids = []
    for i in range(3):
        centroids.append(centroid_calculator(train_data, feature_count_train, i, feature_size_train))
    
    bias = [[0.0 for i in range(3)] for j in range(3)]
    w = [[0.0 for i in range(3)] for j in range(3)]
    
    for i in range(3):
        for j in range(3):
            if i == j:
                w[i][j] = 0
            else:
                w[i][j] = coordSub(centroids[i], centroids[j], feature_count_train, 0)

    for i in range(3):
        for j in range(3):
            bias[i][j] = -0.5 * (coordSq(centroids[i]) - coordSq(centroids[j]))
    
    classif_bias = []
    classif_w = []
    
    for i in range(2):
        for j in range(i+1, 3):
            classif_bias.append(bias[i][j])
            classif_w.append(w[i][j])


    return classif_w, classif_bias
    


# Tests the classifier
def basic_linear_classifier_test(test_data, w, bias):
    classified = []
    for point in test_data:
        classified.append(classify(point, w, bias))
    return classified

def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition. 
    You are you are permitted to use the numpy library but you must write 
    your own code for the linear classifier. 

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values 

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED
        
        Example:
            return {
                "tpr": true_positive_rate,
                "fpr": false_positive_rate,
                "error_rate": error_rate,
                "accuracy": accuracy,
                "precision": precision
            }
    """ 
    # Initializations
    TPR = 0.0
    FPR = 0.0
    errRate = 0.0
    accuracy = 0.0
    precision = 0.0

    # Get dimensionality of data and class sizes
    infoTr = training_input[0]
    infoTes = testing_input[0]

    #class_count_train = len(infoTr[1:]) # the number of classes = 3
    feature_count_train = infoTr[0] # the dimension of features 3-d 4-d etc
    feature_size_train = infoTr[1:] # the number of items in the class


    #class_count_test = len(infoTes[1:])
    #feature_count_test = infoTes[0]
    feature_size_test = infoTes[1:]

    # Get the remaining data into the proper arrays
    train_data = training_input[1:]
    test_data = testing_input[1:]

    # Train classifier
    w, bias = basic_linear_classifier_train(train_data, feature_count_train, feature_size_train)
    test_output = basic_linear_classifier_test(test_data, w, bias)


    # Compare and get the stats to return
    trueAs = 0.0
    trueBs = 0.0
    trueCs = 0.0
    falseB_A = 0.0
    falseB_C = 0.0
    falseA_B = 0.0
    falseA_C = 0.0
    falseC_A = 0.0
    falseC_B = 0.0
    totalSize = feature_size_test[0] + feature_size_test[1] + feature_size_test[2]

    for i in range(feature_size_test[0]):
        if test_output[i] == "A":
            trueAs += 1.0
        elif test_output[i] == "B":
            falseB_A += 1.0
        else:
            falseC_A += 1.0

    for i in range(feature_size_test[0], feature_size_test[0] + feature_size_test[1]):
        if test_output[i] == "B":
            trueBs += 1.0
        elif test_output[i] == "A":
            falseA_B += 1.0
        else:
            falseC_B += 1.0

    for i in range(feature_size_test[0] + feature_size_test[1], feature_size_test[0] + feature_size_test[1] + feature_size_test[2]):
        if test_output[i] == "C":
            trueCs += 1.0
        elif test_output[i] == "A":
            falseC_A += 1.0
        else:
            falseC_B += 1.0

    tprA = trueAs / feature_size_test[0]
    tprB = trueBs / feature_size_test[1]
    tprC = trueCs / feature_size_test[2]
    TPR = (tprA + tprB + tprC) / 3.0
    #print TPR
    fprA = (falseA_B + falseA_C) / (feature_size_test[1] + feature_size_test[2])
    fprB = (falseB_A + falseB_C) / (feature_size_test[0] + feature_size_test[2])
    fprC = (falseC_B + falseC_A) / (feature_size_test[0] + feature_size_test[1])
    FPR = (fprA + fprB + fprC) / 3.0
    #print FPR
    errA = (feature_size_test[0] - trueAs)/feature_size_test[0]
    errB = (feature_size_test[1] - trueBs)/feature_size_test[1]
    errC = (feature_size_test[2] - trueCs)/feature_size_test[2]
    errRate = (errA + errB + errC) / 3.0
    #print errRate
    accuracy = (trueAs + trueBs + trueCs)/totalSize
    #print accuracy
    precA = trueAs / (trueAs + falseA_B + falseA_C)
    precB = trueBs / (trueBs + falseB_A + falseB_C)
    precC = trueCs / (trueCs + falseC_A + falseC_B)
    precision = (precA + precB + precC) / 3.0
    #print precision
    TPR = round(TPR, 2)
    FPR = round(FPR, 2)
    errRate = round(errRate, 2)
    accuracy = round(accuracy, 2)
    precision = round(precision, 2)

    return {
            "tpr": TPR,
            "fpr": FPR,
            "error_rate": errRate,
            "accuracy": accuracy,
            "precision": precision
            }


#######
# The following functions are provided for you to test your classifier.
######
def parse_file(filename):
    """
    This function is provided to you as an example of the preprocessing we do
    prior to calling run_train_test
    """
    with open(filename, "r") as f:
        data = [[float(y) for y in x.strip().split(" ")] for x in f]
        data[0] = [int(x) for x in data[0]]

        return data

if __name__ == "__main__":
    """
    You can use this to test your code.
    python hw2.py [training file path] [testing file path]
    """
    import sys

    training_input = parse_file(sys.argv[1])
    testing_input = parse_file(sys.argv[2])

    run_train_test(training_input, testing_input)

