##############################################
# Programmer: Alexa Williams
# Class: CptS 322-01, Fall 2024
# Programming Assignment #7
# 11/11/2024
# 
# Description: This file contains some 
#   utility functions needed for 
#   various things in this project
##############################################

import numpy as np
import math
from mysklearn import myevaluation
from mysklearn import myclassifiers

def compute_holdout_partitions(table, index, random, random_state):
    """Computes the holdout partitions for a given table
    
    Args:
        table
        index
        random
        random_state
    
    Returns: 
        randomized[0:split_index], randomized[split_index:]
    """
    if random_state is not None:
        np.random.seed(random_state)
    randomized = table[:]
    n = len(table)
    if random:
        for i in range(n):
            j = np.random.randint(0, n)
            randomized[i], randomized[j] = randomized[j], randomized[i]
    split_index = index
    return randomized[0:split_index], randomized[split_index:]

def euclidean_distance(point1, point2):
    """Computes the Euclidean distance between the two points given
    
    Args:
        point1, point2
    
    Returns: 
        double: the euclidean distance
    """

    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

def discretizer_100(item):
    """discretizer function where >= 100 is high and < 100 is low
    
    Args:
        item
    
    Returns: 
        string: "high" or "low"
    """

    if item >= 100:
        item = "high"
    else:
        item = "low"
    return item

def mpg_to_rating_discretizer_2(mpg):
    """discretizer function that maps mpg to its official rating when a list is passed in
    
    Args:
        list: [mpg]
    
    Returns: 
        int: rating number
    """

    if mpg[0] >= 45:
        return 10
    elif 37 <= mpg[0] <=44:
        return 9
    elif 31 <= mpg[0] <= 36:
        return 8
    elif 27 <= mpg[0] <= 30:
        return 7
    elif 24 <= mpg[0] <= 26:
        return 6
    elif 20 <= mpg[0] <= 23:
        return 5
    elif 17 <= mpg[0] <= 19:
        return 4
    elif 15 <= mpg[0] <= 16:
        return 3
    elif mpg[0] == 14:
        return 2
    else:
        return 1

def mpg_to_rating_discretizer(mpg):
    """discretizer function that maps mpg to its official rating when a value is passed in
    
    Args:
        mpg
    
    Returns: 
        int: rating number
    """

    if mpg >= 45:
        return 10
    elif 37 <= mpg <=44:
        return 9
    elif 31 <= mpg <= 36:
        return 8
    elif 27 <= mpg <= 30:
        return 7
    elif 24 <= mpg <= 26:
        return 6
    elif 20 <= mpg <= 23:
        return 5
    elif 17 <= mpg <= 19:
        return 4
    elif 15 <= mpg <= 16:
        return 3
    elif mpg == 14:
        return 2
    else:
        return 1
        

def calculate_accuracy(y_actual, y_pred):
    """function to calculate the accuracy of actual vs predicted values by dividing the amount correct by the total
    
    Args:
        list: y_actual
        list: y_pred
    
    Returns: 
        double: accuracy
    """
    correct = 0
    total = len(y_actual)
    
    for actual, pred in zip(y_actual, y_pred):
        if actual == pred:
            correct += 1
    
    accuracy = correct / total
    return accuracy

def min_max_normalize(value, min_val, max_val):
    """function to normalize a value with simple min max normalization technique
    
    Args:
        value
        min_va
        max_val
    
    Returns: 
        normalized value
    """
    return (value - min_val) / (max_val - min_val)

def random_subsample(X, y, k=10, test_size=0.33):
    """Performs k iterations of random sub-sampling and returns average accuracy and error rate.
    
    Args:
        X
        y
        k
        test_size
    
    Returns: 
        knn_avg_acc, knn_error_rate, dummy_avg_acc, dummy_error_rate
    """

    knn_accuracy, dummy_accuracy = [], []
    
    for _ in range(k):
        X_train, X_test, y_train, y_test = myevaluation.train_test_split(X, y, test_size=test_size)
        
        knn = myclassifiers.MyKNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        knn_accuracy.append(myevaluation.accuracy_score(y_test, y_pred_knn))
        
        dummy = myclassifiers.MyDummyClassifier()
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)
        dummy_accuracy.append(myevaluation.accuracy_score(y_test, y_pred_dummy))
    
    knn_avg_acc = compute_mean(knn_accuracy)
    knn_error_rate = 1 - knn_avg_acc
    dummy_avg_acc = compute_mean(dummy_accuracy)
    dummy_error_rate = 1 - dummy_avg_acc
    
    return knn_avg_acc, knn_error_rate, dummy_avg_acc, dummy_error_rate

def cross_val_predict(X, y, k=10):
    """Performs k-fold (or stratified) cross-validation and returns average accuracy and error rate.

    Args:
        X
        y
        k
    
    Returns: 
        knn_avg_acc, knn_error_rate, dummy_avg_acc, dummy_error_rate, knn_y_actual, knn_y_pred, dummy_y_actual, dummy_y_pred
    """
    folds = myevaluation.kfold_split(X, n_splits=k, random_state = 0, shuffle = True)
    knn_accuracy, dummy_accuracy = [], []
    knn_y_actual = []
    knn_y_pred = []
    dummy_y_actual = []
    dummy_y_pred = []

    for i in range(len(folds)):
        X_train, X_test, y_train, y_test = get_train_test_data(X, y, folds[i])

        knn = myclassifiers.MyKNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        knn_accuracy.append(myevaluation.accuracy_score(y_pred_knn, y_test))
        for i in range(len(y_test)):
            knn_y_actual.append(y_test[i])
            knn_y_pred.append(y_pred_knn[i])
        
        dummy = myclassifiers.MyDummyClassifier()
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)
        dummy_accuracy.append(myevaluation.accuracy_score(y_pred_dummy, y_test))
        for i in range(len(y_test)):
            dummy_y_actual.append(y_test[i])
            dummy_y_pred.append(y_pred_dummy[i])
        
    knn_avg_acc = compute_mean(knn_accuracy)
    knn_error_rate = 1 - knn_avg_acc
    dummy_avg_acc = compute_mean(dummy_accuracy)
    dummy_error_rate = 1 - dummy_avg_acc

    return knn_avg_acc, knn_error_rate, dummy_avg_acc, dummy_error_rate, knn_y_actual, knn_y_pred, dummy_y_actual, dummy_y_pred

def get_train_test_data(X, y, fold):
    """
    Extracts the train and test sets for a given fold.

    Args:
        X
        y
        fold

    Returns:
        X_train, X_test, y_train, y_test
    """
    train_indices, test_indices = fold

    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test


def bootstrap_method(X, y, k=10):
    """Performs k iterations of bootstrap sampling and returns average accuracy and error rate.

    Args:
        X
        y
        fold

    Returns:
        knn_avg_acc, knn_error_rate, dummy_avg_acc, dummy_error_rate
    """
    knn_accuracy, dummy_accuracy = [], []

    for _ in range(k):
        X_sample, X_out_of_bag, y_sample, y_out_of_bag = myevaluation.bootstrap_sample(X, y)
        
        knn = myclassifiers.MyKNeighborsClassifier()
        knn.fit(X_sample, y_sample)
        y_pred_knn = knn.predict(X_out_of_bag)
        knn_accuracy.append(myevaluation.accuracy_score(y_pred_knn, y_out_of_bag))
        
        dummy = myclassifiers.MyDummyClassifier()
        dummy.fit(X_sample, y_sample)
        y_pred_dummy = dummy.predict(X_out_of_bag)
        dummy_accuracy.append(myevaluation.accuracy_score(y_pred_dummy, y_out_of_bag))

    knn_avg_acc = compute_mean(knn_accuracy)
    knn_error_rate = 1 - knn_avg_acc
    dummy_avg_acc = compute_mean(dummy_accuracy)
    dummy_error_rate = 1 - dummy_avg_acc

    return knn_avg_acc, knn_error_rate, dummy_avg_acc, dummy_error_rate

def compute_mean(values):
    """computes mean of a list of numeric values

    Args:
        values

    Returns:
        value of the mean
    """
    return sum(values) / len(values)

def calegorical_distance(x1, x2):
    """Compute a categorical distance
    
    Args:
        x1 (list): The first instance (e.g., test instance).
        x2 (list): The second instance (e.g., train instance).
    
    Returns:
        float: The computed distance between x1 and x2.
    """
    distance = 0
    for i, (v1, v2) in enumerate(zip(x1, x2)):
        distance += 0 if v1 == v2 else 1

    return distance ** 0.5

def knn_nb_classifiers(X, y):
    folds = myevaluation.kfold_split(X, n_splits=10, random_state = 1, shuffle = True)
    knn_accuracy, nb_accuracy, dummy_accuracy = [], [], []
    knn_y_actual = []
    knn_y_pred = []
    nb_y_actual = []
    nb_y_pred = []
    dummy_y_actual = []
    dummy_y_pred = []


    for i in range(len(folds)):
        X_train, X_test, y_train, y_test = get_train_test_data(X, y, folds[i])

        knn = myclassifiers.MyKNeighborsClassifier(n_neighbors=10)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        knn_accuracy.append(myevaluation.accuracy_score(y_pred_knn, y_test))
        for i in range(len(y_test)):
            knn_y_actual.append(y_test[i])
            knn_y_pred.append(y_pred_knn[i])
        
        nb = myclassifiers.MyNaiveBayesClassifier()
        nb.fit(X_train, X_test)
        y_pred_nb = nb.predict(X_test)
        nb_accuracy.append(myevaluation.accuracy_score(y_pred_nb, y_test))
        for i in range(len(y_test)):
            nb_y_actual.append(y_test[i])
            nb_y_pred.append(y_pred_nb[i])
        
        dummy = myclassifiers.MyDummyClassifier()
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)
        dummy_accuracy.append(myevaluation.accuracy_score(y_pred_dummy, y_test))
        for i in range(len(y_test)):
            dummy_y_actual.append(y_test[i])
            dummy_y_pred.append(y_pred_dummy[i])
        
    knn_avg_acc = compute_mean(knn_accuracy)
    knn_error_rate = 1 - knn_avg_acc

    dummy_avg_acc = compute_mean(dummy_accuracy)
    dummy_error_rate = 1 - dummy_avg_acc

    nb_avg_acc = compute_mean(nb_accuracy)
    nb_error_rate = 1 - nb_avg_acc

    knn_binary_ps = myevaluation.binary_precision_score(knn_y_actual, knn_y_pred)
    nb_binary_ps = myevaluation.binary_precision_score(nb_y_actual, nb_y_pred)
    dummy_binary_ps = myevaluation.binary_precision_score(dummy_y_actual, dummy_y_pred)

    knn_recall = myevaluation.binary_recall_score(knn_y_actual, knn_y_pred)
    nb_recall = myevaluation.binary_recall_score(nb_y_actual, nb_y_pred)
    dummy_recall = myevaluation.binary_recall_score(dummy_y_actual, dummy_y_pred)

    knn_f1 = myevaluation.binary_f1_score(knn_y_actual, knn_y_pred)
    nb_f1 = myevaluation.binary_f1_score(nb_y_actual, nb_y_pred)
    dummy_f1 = myevaluation.binary_f1_score(dummy_y_actual, dummy_y_pred)

    return knn_avg_acc, knn_error_rate, nb_avg_acc, nb_error_rate, knn_y_actual, knn_y_pred, nb_y_actual, nb_y_pred, knn_binary_ps, nb_binary_ps, knn_recall, nb_recall, knn_f1, nb_f1, dummy_avg_acc, dummy_error_rate, dummy_binary_ps, dummy_recall, dummy_f1, dummy_y_actual, dummy_y_pred

def tree_classifier(X, y, header):
    folds = myevaluation.kfold_split(X, n_splits=10, random_state = 1, shuffle = True)
    tree_accuracy = []
    y_actual = []
    y_pred = []
    y_pred_full = []
    mytree = myclassifiers.MyDecisionTreeClassifier(header)
    
    for i in range(len(folds)):
        X_train, X_test, y_train, y_test = get_train_test_data(X, y, folds[i])

        
        mytree.fit(X_train, y_train)
        y_pred = mytree.predict(X_test)
        tree_accuracy.append(myevaluation.accuracy_score(y_pred, y_test))
        for i in range(len(y_test)):
            y_actual.append(y_test[i])
            y_pred_full.append(y_pred[i])
        
    tree_avg_acc = compute_mean(tree_accuracy)
    tree_error_rate = 1 - tree_avg_acc

    tree_binary_ps = myevaluation.binary_precision_score(y_actual, y_pred_full)
    tree_recall = myevaluation.binary_recall_score(y_actual, y_pred_full)
    tree_f1 = myevaluation.binary_f1_score(y_actual, y_pred_full)

    return tree_avg_acc, tree_error_rate, tree_binary_ps, tree_recall, tree_f1, y_actual, y_pred_full

def tree_classifier_train_all(X, y, header):
    mytree = myclassifiers.MyDecisionTreeClassifier(header)
    mytree.fit(X, y)
    return mytree

def forest_classifier(X, y):
    folds = myevaluation.kfold_split(X, n_splits=10, random_state = 1, shuffle = True)
    tree_accuracy = []
    y_actual = []
    y_pred = []
    y_pred_full = []
    mytree = myclassifiers.MyRandomForestClassifier()
    
    for i in range(len(folds)):
        X_train, X_test, y_train, y_test = get_train_test_data(X, y, folds[i])

        
        mytree.fit(X_train, y_train)
        y_pred = mytree.predict(X_test)
        tree_accuracy.append(myevaluation.accuracy_score(y_pred, y_test))
        for i in range(len(y_test)):
            y_actual.append(y_test[i])
            y_pred_full.append(y_pred[i])
        
    tree_avg_acc = compute_mean(tree_accuracy)
    tree_error_rate = 1 - tree_avg_acc

    tree_binary_ps = myevaluation.binary_precision_score(y_actual, y_pred_full)
    tree_recall = myevaluation.binary_recall_score(y_actual, y_pred_full)
    tree_f1 = myevaluation.binary_f1_score(y_actual, y_pred_full)

    return tree_avg_acc, tree_error_rate, tree_binary_ps, tree_recall, tree_f1, y_actual, y_pred_full, mytree


        