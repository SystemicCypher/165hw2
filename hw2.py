# Starter code for CS 165B HW2

def centroid_calculator(train_data, feature_count, label, label_count):
    # Initializations
    point_sum = [0.0] * feature_count
    centroid = [0.0] * feature_count

    # Calculation
    # Sum of every point 
    for i in range(0+label_count*label,label_count*(label+1)):
        for j in range(0,feature_count):
            point_sum[j] += train_data[i][j]

    # Mean point - "centroid"
    for i in range(0, feature_count):
        centroid[i] = point_sum[i]/feature_count

    return (label, centroid)

def coordSub(pointA, pointB, size):
    coSum = [0.0 for i in range(size)]
    for x in range(size):
        coSum[x] = pointA[x] - pointB[x]
        coSum[x] = coSum[x]/2.0
    return coSum

def basic_linear_classifier_train(train_data, train_info):
    centroids = []
    total_labels = -1
    for i in train_info:
        total_labels += 1
    for i in range(0, total_labels):
        centroids.append(centroid_calculator(train_data, train_info[0], i, train_info[i]))
    
    #distance = [[0.0 for i in range(total_labels)] for j in range(total_labels)]
    classifier = [[0.0 for i in range(total_labels)] for j in range(total_labels)]
    
    for i in range(0, total_labels):
        for j in range(0, total_labels):
            if i == j:
                classifier[i][j] = 0
            else:
                classifier[i][j] = coordSub(centroids[i], centroids[j], train_info[0])


    return classifier
    



def basic_linear_classifier_test(test_data):
    return 3

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

    # Get the remaining data into the proper arrays
    train_data = training_input[1:]
    test_data = testing_input[1:]

    # Train classifier
    train_output = basic_linear_classifier_train(train_data, infoTr)
    test_output = basic_linear_classifier_test(test_data)

    # Compare and get the stats to return


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

