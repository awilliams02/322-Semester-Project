##############################################
# Programmer: Alexa Williams and Mia Procel
# Class: CptS 322-01, Fall 2024
# Stress Detection
# 12/9/2024
# 
# Description: This file contains the 
#   classes for mysklearn classifiers
##############################################

from mysklearn import myutils
import numpy as np
from collections import Counter
import math
import copy

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        if type(X_test[0]) in [int, float]:
            feature_type = "Numeric"
        else:
            feature_type = "Categorical"

        distances = []
        neighbor_indices = []
        
        for test_instance in X_test:
            if(feature_type == "Numeric"):
                dists = [(myutils.euclidean_distance(test_instance, train_instance), idk )for idk, train_instance in enumerate(self.X_train)]
            else: 
                dists = [(myutils.calegorical_distance(test_instance, train_instance), idk )for idk, train_instance in enumerate(self.X_train)]

            dists = sorted(dists)[:self.n_neighbors]

            distances.append([dist for dist, _ in dists])
            neighbor_indices.append([idx for _, idx in dists])
        
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        _, neighbor_indeces = self.kneighbors(X_test)

        for indeces in neighbor_indeces:
            neighbor_labels = [self.y_train[i] for i in indeces]
            most_common = Counter(neighbor_labels).most_common(1)[0][0]
            y_predicted.append(most_common)
        
        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        
        self.most_common_label = Counter(y_train).most_common(1)[0][0]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        
        return [self.most_common_label] * len(X_test)

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        """Fits a Naive Bayes classifier to X_train and y_train."""
        if isinstance(y_train[0], list):
            y_train = [label[0] for label in y_train]
        self.priors = {}
        self.posteriors = {}

        total_samples = len(y_train)
        for label in set(y_train):
            self.priors[label] = y_train.count(label) / total_samples

        feature_count = len(X_train[0])
        for label in self.priors:
            self.posteriors[label] = [{} for _ in range(feature_count)]
            label_indices = [i for i, lbl in enumerate(y_train) if lbl == label]
            for feature_idx in range(feature_count):
                feature_values = [X_train[i][feature_idx] for i in label_indices]
                unique_values = set(feature_values)
                for value in unique_values:
                    value_count = feature_values.count(value)
                    self.posteriors[label][feature_idx][value] = value_count / len(feature_values)

                # Laplace smoothing...    
                for value in set([item[feature_idx] for item in X_train]):
                    if value not in self.posteriors[label][feature_idx]:
                        self.posteriors[label][feature_idx][value] = 1e-9  

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        """Makes predictions for test instances in X_test."""
        """Makes predictions for test instances in X_test."""
        y_predicted = []
        
        for instance in X_test:
            max_posterior = None
            best_class = None
            
            for label in self.priors:
                log_posterior = math.log(self.priors[label])
                
                for feature_idx, feature_value in enumerate(instance):
                    log_posterior += math.log(self.posteriors[label][feature_idx][feature_value])
                
                if max_posterior is None or log_posterior > max_posterior:
                    max_posterior = log_posterior
                    best_class = label
            
            y_predicted.append(best_class)
        
        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, h):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.header = copy.deepcopy(h)

    def select_attribute(self, instances, attributes):
        """Selects the attribute with the lowest entropy (highest information gain)."""
        min_entropy = float('inf')
        best_attribute = None

        for attribute in attributes:
            # Calculate the entropy of the split for this attribute
            weighted_entropy = self.calculate_weighted_entropy(instances, attribute)
            
            # Check if this attribute has the lowest entropy found so far
            if weighted_entropy < min_entropy:
                min_entropy = weighted_entropy
                best_attribute = attribute

        return best_attribute
    def majority_vote(self, instances):
        class_labels = [instance[-1] for instance in instances]
        label_counts = Counter(class_labels)
        majority_class, _ = label_counts.most_common(1)[0]
    
        return majority_class
    
    def tdidt(self, current_instances, available_attributes, attribute_domains):
        
        split_attribute = self.select_attribute(current_instances, available_attributes)
        available_attributes.remove(split_attribute) # can't split on this attribute again
        att = "att"
        index = str(self.header.index(split_attribute))
        attribute = att + index
        tree = ["Attribute", attribute]

        partitions = self.partition_instances(current_instances, split_attribute, attribute_domains)
        for att_value in sorted(partitions.keys()):
            att_partition = partitions[att_value]
            if len(att_partition) > 0 and self.all_same_class(att_partition):
                # Case 1: All class labels in this partition are the same -> Leaf node
                leaf_label = att_partition[0][-1]  # All have the same class label
                tree.append(["Value", att_value, ["Leaf", leaf_label, len(att_partition), len(current_instances)]])
            elif len(att_partition) == 0 or len(available_attributes) == 0:
                # Case 2 or 3: No more attributes or instances -> Majority vote leaf node
                majority_class = self.majority_vote(current_instances)
                tree.append(["Value", att_value, ["Leaf", majority_class, len(att_partition), len(current_instances)]])
            else:
                # Recursive case: build subtree
                subtree = self.tdidt(att_partition, available_attributes.copy(), attribute_domains)
                tree.append(["Value", att_value, subtree])
                
        return tree        
        
    def calculate_weighted_entropy(self, instances, attribute):
        """Calculates the weighted average entropy for a given attribute."""
        partitions = self.partition_instances(instances, attribute, 
                                                attribute_domains={attr: list(set([row[self.header.index(attr)] for row in instances])) for attr in self.header})

        total_instances = len(instances)
        weighted_entropy = 0.0

        for att_value, partition in partitions.items():
            if len(partition) > 0:
                partition_entropy = self.entropy([row[-1] for row in partition])
                weighted_entropy += (len(partition) / total_instances) * partition_entropy

        return weighted_entropy

    def entropy(self, class_labels):
        """Calculates the entropy for a list of class labels."""
        total = len(class_labels)
        if total == 0:
            return 0

        label_counts = Counter(class_labels)
        entropy = 0.0
        for count in label_counts.values():
            probability = count / total
            entropy -= probability * np.log2(probability)

        return entropy

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train      
        available_attributes = []
        train = [([X_train[i]] if isinstance(X_train[i], str) else X_train[i]) + [y_train[i]] for i in range(len(X_train))]
        attribute_domains = {self.header[i]: list(set([row[i] for row in X_train])) for i in range(len(self.header))}
        available_attributes = list(self.header).copy()
        self.tree = self.tdidt(train, available_attributes, attribute_domains)

    def partition_instances(self, instances, attribute, attribute_domains):
        """Partitions instances by the values of the chosen attribute."""
        att_index = self.header.index(attribute)
        partitions = {val: [] for val in attribute_domains[attribute]}
        
        for instance in instances:
            partitions[instance[att_index]].append(instance)
        
        return partitions

    def all_same_class(self, instances):
        """Checks if all instances have the same class label."""
        first_label = instances[0][-1]
        return all(instance[-1] == first_label for instance in instances)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        """Predicts class labels for X_test instances based on the fitted decision tree."""
        predictions = []
        for instance in X_test:
            predictions.append(self.predict_instance(instance, self.tree))
        return predictions

    def predict_instance(self, instance, subtree):
        """Recursively traverses the tree for an instance to make a prediction."""
        if subtree[0] == "Leaf":
            return subtree[1]

        attribute = subtree[1]
        attribute_index = self.header.index(attribute)

        for value_node in subtree[2:]:
            if value_node[0] == "Value" and instance[attribute_index] == value_node[1]:
                return self.predict_instance(instance, value_node[2])

        return None  # Edge case if no path is found (shouldn't happen)
    
    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.
        """
        if attribute_names is None:
            attribute_names = [f"att{i}" for i in range(len(self.X_train[0]))]
        
        self.generate_rules(self.tree, "", attribute_names, class_name)

    def generate_rules(self, subtree, rule, attribute_names, class_name):
        """Recursively generates and prints the decision rules for each path to a leaf."""
        if subtree[0] == "Leaf":
            print(f"{rule} THEN {class_name} = {subtree[1]}")
            return
        
        attribute_index = int(subtree[1][3:])
        attribute_name = attribute_names[attribute_index]

        for value_node in subtree[2:]:
            if value_node[0] == "Value":
                new_rule = f"{rule}IF {attribute_name} == {value_node[1]}" if rule == "" else f"{rule} AND {attribute_name} == {value_node[1]}"
                self.generate_rules(value_node[2], new_rule, attribute_names, class_name)
                
    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this

class MyRandomForestClassifier:
    """Represents a random forest classifier.

    Attributes:
        n_estimators (int): The number of decision trees in the forest.
        max_features (int): The maximum number of features to consider when splitting.
        bootstrap (bool): Whether to use bootstrap samples for training the trees.
        trees (list of MyDecisionTreeClassifier): The individual decision tree classifiers.
        X_train (list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features).
        y_train (list of obj): The target y values (parallel to X_train).
                The shape of y_train is n_train_samples.
    
    Notes:
        Loosely based on sklearn's RandomForestClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    def __init__(self, n_estimators=100, max_features=None, bootstrap=True):
        """Initializer for MyRandomForestClassifier.

        Args:
            n_estimators (int): Number of decision trees in the forest.
            max_features (int or None): Maximum number of features to consider for splits. 
                Defaults to None (all features considered).
            bootstrap (bool): Whether to use bootstrap samples for training the trees.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits the random forest classifier to the training data.

        Args:
            X_train (list of list of obj): The list of training instances.
            y_train (list of obj): The target values.
        
        Notes:
            For each tree, a bootstrap sample is drawn from the training data if bootstrap=True.
            Each tree is trained on a random subset of features (controlled by max_features).
        """
        self.X_train = X_train
        self.y_train = y_train
        n_samples = len(X_train)
        n_features = len(X_train[0])
        max_features = self.max_features or n_features

        for _ in range(self.n_estimators):
            # Bootstrap sample
            if self.bootstrap:
                sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_sample = [X_train[i] for i in sample_indices]
                y_sample = [y_train[i] for i in sample_indices]
            else:
                X_sample, y_sample = X_train, y_train

            # Random subset of features
            feature_indices = np.random.choice(n_features, size=max_features, replace=False)
            X_sample = [[x[i] for i in feature_indices] for x in X_sample]

            # Train a decision tree
            tree = MyDecisionTreeClassifier([f"att{i}" for i in feature_indices])
            tree.fit(X_sample, y_sample)
            self.trees.append((tree, feature_indices))

    def predict(self, X_test):
        """Makes predictions for the test data.

        Args:
            X_test (list of list of obj): The list of test instances.

        Returns:
            y_predicted (list of obj): The predicted target values.
        """
        predictions = []

        for tree, feature_indices in self.trees:
            # Restrict test data to the tree's features
            X_subset = [[x[i] for i in feature_indices] for x in X_test]
            predictions.append(tree.predict(X_subset))

        # Combine predictions by majority vote
        y_predicted = []
        for i in range(len(X_test)):
            votes = [pred[i] for pred in predictions]
            majority_vote = Counter(votes).most_common(1)[0][0]
            if majority_vote is None:
                majority_vote = self.y_train[0]
            y_predicted.append(majority_vote)

        return y_predicted
