# Starter code for CS 165B HW2

def centroid_calculator(train_data, feature_count, class_count, label, feature_size):
    # Initializations
    point_sum = [0.0] * feature_count
    centroid = [0.0] * class_count

    start = 0 + feature_size[label] * label
    end = feature_size[label] * (label + 1)

    # Calculation
    # Sum of every point 
    for i in range(start, end):
        for j in range(0, class_count):
            point_sum[j] += train_data[i][j]

    # Mean point - "centroid"
    for i in range(0, feature_count):
        centroid[i] = point_sum[i]/feature_count
    
    return centroid

def coordSub(pointA, pointB, size):
    coSum = [0.0] * size
    for x in range(size):
        coSum[x] = pointA[x] - pointB[x]
        coSum[x] = coSum[x]/2.0
    return coSum

def basic_linear_classifier_train(train_data, feature_count_train, class_count_train, feature_size_train):
    centroids = []

    for i in range(0, class_count_train):
        centroids.append(centroid_calculator(train_data, feature_count_train, class_count_train, i, feature_size_train))

    
    classifier = [[0.0 for i in range(class_count_train)] for j in range(class_count_train)]
    
    for i in range(0, class_count_train):
        for j in range(0, class_count_train):
            if i == j:
                classifier[i][j] = 0
            else:
                classifier[i][j] = coordSub(centroids[i], centroids[j], feature_count_train)
    return classifier
    



def basic_linear_classifier_test(test_data, classifier):

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

    class_count_train = len(infoTr[1:])
    feature_count_train = infoTr[0]
    feature_size_train = infoTr[1:]


    #class_count_test = len(infoTes[1:])
    #feature_count_test = infoTes[0]
    #feature_size_test = infoTr[1:]

    # Get the remaining data into the proper arrays
    train_data = training_input[1:]
    test_data = testing_input[1:]

    # Train classifier
    train_output = basic_linear_classifier_train(train_data, feature_count_train, class_count_train, feature_size_train)
    #test_output = basic_linear_classifier_test(test_data, train_output)

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

