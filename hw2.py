from __future__ import division
# Starter code for CS 165B HW2
import numpy as np


def train_classifier(data, dimension, valuesPerClass):
    classA = []
    classB = []
    classC = []
    
    for i in range(valuesPerClass[0]):
        classA.append(data[i])
    
    for i in range(valuesPerClass[0], valuesPerClass[0] + valuesPerClass[1]):
        classB.append(data[i])
    
    for i in range(valuesPerClass[0] + valuesPerClass[1], valuesPerClass[0] + valuesPerClass[1] + valuesPerClass[2]):
        classC.append(data[i])
    
    A = np.array(classA)
    B = np.array(classB)
    C = np.array(classC)

    centroidA = np.mean(A, axis=0)
    centroidB = np.mean(B, axis=0)
    centroidC = np.mean(C, axis=0)
    #print centroidA, centroidB, centroidC

    wAB = centroidA - centroidB
    wAC = centroidA - centroidC
    wBC = centroidB - centroidC
    #print wA,wB,wC

    w = [wAB, wAC, wBC]
    w = np.array(w)
    #print w

    biasAB = -0.5 * (centroidA**2 - centroidB**2)
    biasAB = -0.5 * np.dot((centroidA - centroidB),(centroidA + centroidB))
    biasAC = -0.5 * (centroidA**2 - centroidC**2)
    biasAC = -0.5 * np.dot((centroidA - centroidC),(centroidA + centroidC))
    biasBC = -0.5 * (centroidB**2 - centroidC**2)
    biasBC = -0.5 * np.dot((centroidB - centroidC),(centroidB + centroidC))
    #print biasAB, biasAC, biasBC

    bias = [[biasAB, biasAC, biasBC]]
    bias = np.array(bias)
    #print bias

    return w, bias 

def classify(point, w, bias, dimension):
    coord = np.array([point]).reshape(((dimension, 1)))
    
    AB_val = (np.dot(w[0], coord)[0] + bias[0][0])
    AC_val = (np.dot(w[1], coord)[0] + bias[0][1])
    BC_val = (np.dot(w[2], coord)[0] + bias[0][2])
    

    if AB_val >= 0:
        if AC_val >= 0:
            y_hat = "A"
        else:
            y_hat = "C"
    else:
        if BC_val >= 0:
            y_hat = "B"
        else:
            y_hat = "C"
        
    

    return y_hat

def test_and_classify(data, w, bias, dimension):
    classified = []
    for point in data:
        classified.append(classify(point, w, bias, dimension))
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
    train_info = training_input[0]
    test_info = testing_input[0]

    train_dimension = train_info[0]
    number_training_points_perClass = train_info[1:]

    test_dimension = test_info[0]
    number_testing_points_perClass = test_info[1:]
    

    train_data = training_input[1:]
    test_data = testing_input[1:]
    #train = np.array(train_data)
    #test = np.array(test_data)

    w, bias = train_classifier(train_data, train_dimension, number_training_points_perClass)
    test_output = test_and_classify(test_data, w, bias, test_dimension)

    #for i in test_output:
    #    print i
    

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

    total_size = number_testing_points_perClass[0] + number_testing_points_perClass[1] + number_testing_points_perClass[2]

    for i in range(number_testing_points_perClass[0]):
        if test_output[i] == "A":
            trueAs += 1.0
        elif test_output[i] == "B":
            falseB_A += 1.0
        else:
            falseC_A += 1.0

    for i in range(number_testing_points_perClass[0], number_testing_points_perClass[0] + number_testing_points_perClass[1]):
        if test_output[i] == "B":
            trueBs += 1.0
        elif test_output[i] == "A":
            falseA_B += 1.0
        else:
            falseC_B += 1.0

    for i in range(number_testing_points_perClass[0] + number_testing_points_perClass[1], number_testing_points_perClass[0] + number_testing_points_perClass[1] + number_testing_points_perClass[2]):
        if test_output[i] == "C":
            trueCs += 1.0
        elif test_output[i] == "A":
            falseA_C += 1.0
        else:
            falseB_C += 1.0

    tprA = trueAs / (trueAs + falseB_A + falseC_A)
    tprB = trueBs / (trueBs + falseA_B + falseC_B)
    tprC = trueCs / (trueCs + falseB_C + falseA_C)
    TPR = (tprA + tprB + tprC) / 3.0
    #print TPR
    fprA = (falseA_B + falseA_C) / (falseA_B + falseA_C + trueBs + falseC_B + falseB_C + trueCs)
    fprB = (falseB_A + falseB_C) / (falseB_A + falseB_C + trueAs + falseC_A + falseA_C + trueCs)
    fprC = (falseC_B + falseC_A) / (falseC_B + falseC_A + falseB_A + trueBs + trueAs + falseA_B)
    FPR = (fprA + fprB + fprC) / 3.0
    #print FPR
    errA = (falseA_B + falseA_C + falseB_A + falseC_A)/((falseA_B + falseA_C + trueBs + falseC_B + falseB_C + trueCs) + (trueAs + falseB_A + falseC_A))
    errB = (falseB_A + falseB_C + falseA_B + falseC_B)/((falseB_A + falseB_C + trueAs + falseC_A + falseA_C + trueCs) + (trueBs + falseA_B + falseC_B))
    errC = (falseC_A + falseC_B + falseA_C + falseB_C)/((falseC_B + falseC_A + falseB_A + trueBs + trueAs + falseA_B) + (trueCs + falseB_C + falseA_C) )
    errRate = (errA + errB + errC) / 3.0
    #print errRate
    #accA = 
    P_A =  trueAs + falseB_A + falseC_A
    P_B =  trueBs + falseA_B + falseC_B
    P_C =  trueCs + falseB_C + falseA_C
    N_A = falseA_B + falseA_C + trueBs + falseC_B + falseB_C + trueCs
    N_B = falseB_A + falseB_C + trueAs + falseC_A + falseA_C + trueCs
    N_C = falseC_B + falseC_A + falseB_A + trueBs + trueAs + falseA_B
    #print P_A, P_B, P_C, N_A, N_B, N_C
    accuracyA = (P_A/(P_A + N_A))*tprA + (N_A/(P_A + N_A))*(1 - fprA)
    accuracyB = (P_B/(P_B + N_B))*tprB + (N_B/(P_B + N_B))*(1 - fprB)
    accuracyC = (P_C/(P_C + N_C))*tprC + (N_C/(P_C + N_C))*(1 - fprC)
    accuracy = (accuracyA + accuracyB + accuracyC) / 3.0

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

    print run_train_test(training_input, testing_input)

