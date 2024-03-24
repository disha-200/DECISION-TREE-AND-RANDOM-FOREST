import collections
import math
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from recursive_decisiontree_copy import *


class RandomForest:
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.trees = []
        # self.max_depth = max_depth
        # self.min_samples_split = min_samples_split
        self.finaltrain_predictions = collections.defaultdict(list)
        self.finaltest_predictions = collections.defaultdict(list)

    def fit(self, train_data, test_data):

        # n_samples, n_features = X.shape
        bootstrap_train = self.bootstrap_data(train_data)
        # print("bootstrap_train", bootstrap_train)

        for i in range(self.n_trees):

            # Train a decision tree on the bootstrap sample
            tree = DecisionTree(bootstrap_train[i])
            attribute_list = list(bootstrap_train[i].iloc[:, :-1].columns)
            # maintaining an attribute list with all attribute names - here getting columns excluding last column -
            # dataset specific
            tree.getvalues_forattributes(bootstrap_train[i], attribute_list)
            tree.root = tree.Construct_tree(bootstrap_train[i], attribute_list)

            X_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1]
            # testing each tree
            train_predictions = []
            test_predictions = []
            test_predictions = tree.find_Accuracy(X_test, y_test)
            # print('test preds', test_predictions)# is a list of predictions value for
            # each instance
            self.finaltest_predictions[i] = test_predictions

        # maxvotes_trainpred = self.maximum_voting(self.finaltrain_predictions, data.iloc[:, -1].unique())
        # print("values in test data", test_data.iloc[:, -1].unique())
        maxvotes_testpred = self.maximum_voting(self.finaltest_predictions, test_data.iloc[:, -1].unique())
        # print('max_votes_pred',maxvotes_testpred)
        # forest_testaccuracy = self.forest_accuracy(test_data.iloc[:, -1].reset_index(drop = True), maxvotes_testpred)
        forest_testaccuracy, precision, recall, f1 = self.evaluate(test_data.iloc[:, -1].reset_index(drop=True),
                                                                   maxvotes_testpred)

        return forest_testaccuracy, precision, recall, f1
        # print("Accuracy of forest: ", forest_testaccuracy)
        # print("Accuracy of precision: ", precision)
        # print("Accuracy of recall: ", recall)
        # print("Accuracy of f1: ", f1)
        # print("---------------")

    def bootstrap_data(self, train_data):
        n_samples = train_data.shape[0]
        # print("n smaples", n_samples)
        bootstrap_data = []
        for i in range(self.n_trees):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            # print('indices',indices)
            bootstrap_data.append(train_data.iloc[indices].reset_index(drop=True))
            # print('data')
            # print(bootstrap_data)
        return bootstrap_data

    def maximum_voting(self, predictions, class_value_list):

        # Get the number of instances
        num_instances = len(predictions[0])

        # Initialize a list to store the maximum voting for each instance
        max_voting = []

        # Loop through each instance
        for i in range(num_instances):
            # Initialize a dictionary to store the counts for each class
            counts = {value: 0 for value in class_value_list}
            # print('counts')
            # print(counts)


            # Loop through each tree
            for each_tree in predictions:
                # Get the class for the current instance in the current tree
                class_label = predictions[each_tree][i]

                # Increment the count for the current class label
                counts[class_label] += 1

            # Get the class label with the maximum count
            max_count_class = max(counts, key=counts.get)

            # Add the maximum count to the max_voting list
            max_voting.append(max_count_class)

        # Print the maximum voting for each instance
        return max_voting

    def forest_accuracy(self, original_testpred, maxvotes_testpred):
        correct = 0
        total = len(original_testpred)
        for i in range(total):
            if original_testpred[i] == maxvotes_testpred[i]:
                correct += 1
        accuracy = correct / total
        return accuracy

    def evaluate(self, y_true, y_pred):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                if y_true[i] == 1:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if y_true[i] == 1:
                    false_negatives += 1
                else:
                    false_positives += 1
        accuracy = (true_positives + true_negatives) / len(y_true)
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        return accuracy, precision, recall, f1

class Node:
    def __init__(self):
        self.children = []
        self.attributename = None  # attribute
        self.leaf = None
        self.final_value = None  # classid

    def setasLeaf(self, final_value):
        self.children = []
        self.attributename = None  # attribute
        self.leaf = True
        self.final_value = final_value  # classid

    def setasDecisionNode(self, attributename):
        self.children = []
        self.attributename = attributename  # attribute
        self.leaf = False
        self.final_value = None  # classid


class DecisionTree:
    def __init__(self, data, max_depth=5):
        self.root = None
        self.attribute_value_list = dict()
        self.max_depth = max_depth
        self.current_depth = 0
        self.flag = 1  # make flag = 0 for gini criterion

    def getvalues_forattributes(self,data, attribute_list):
        for attribute in attribute_list:
            values = list(set(data[attribute]))
            self.attribute_value_list[attribute] = values
        return

    def Construct_tree(self,data, attribute_list, min_instances=5):
        node = Node()
        if (len(data['target'].unique()) <= 1) or len(data) < min_instances:  # dataset specific
            node.setasLeaf(data.iloc[0]['target'])
            return node
        if (len(attribute_list) == 0) or (self.current_depth == self.max_depth):
            node.setasLeaf(data.target.mode()[0])
            return node

        self.current_depth += 1
        bestsplit_Attribute = self.find_Split(data, attribute_list, self.flag)
        # print("Best attribute found ",bestsplit_Attribute)
        node.setasDecisionNode(bestsplit_Attribute)
        # attribute_list.remove(bestsplit_Attribute)

        for val in self.attribute_value_list[bestsplit_Attribute]:
            split_data = data[data[bestsplit_Attribute] == val]
            if split_data.size == 0:
                node.setasLeaf(data.target.mode()[0])
                return node
            else:
                childTree = self.Construct_tree(split_data, attribute_list, min_instances)
                node.children.append(childTree)

        return node

    def entropy_Value_info(self, data_rows):
        entropy = 0.0
        base = 2.0
        target_column = data_rows.iloc[:, -1]  # chooses last column -- specific to dataset
        if len(data_rows) == 0:
            return 0.0
        totalno_rows = len(target_column)
        # values_target = set(target_column)
        values_target = target_column.unique()
        counts = {value: 0 for value in values_target}
        for row in target_column:
            counts[row] += 1
        probability = [counts[value] / totalno_rows for value in values_target]
        probability = [p for p in probability if p != 0]
        for p in probability:
            entropy -= p * math.log(p, base)
        return entropy

    def entropy_Value_gini(self, data_rows):
        final_entropy = 0.0
        target_column = data_rows.iloc[:, -1]  # chooses last column -- specific to dataset
        if len(data_rows) == 0:
            return 0.0
        totalno_rows = len(target_column)
        # values_target = set(target_column)
        values_target = target_column.unique()
        counts = {value: 0 for value in values_target}
        for row in target_column:
            counts[row] += 1
        probability = [counts[value] / totalno_rows for value in values_target]
        probability = [p for p in probability if p != 0]
        for p in probability:
            final_entropy += (p**2)
        entropy = 1 - final_entropy
        return entropy


    def Information_gain(self, data, attribute_lists_left, entropy_set):
        max_informationgain = 0
        attributeon_split = ' '
        # selecting root_n number of attributes from entire attribute list
        n = len(attribute_lists_left)
        root_n = int(math.sqrt(n))
        root_n_attribute_list = random.sample(attribute_lists_left, root_n)

        for single_attribute in root_n_attribute_list:
            if np.issubdtype(data[single_attribute].dtype, np.number):
                max_value = data[single_attribute].max()
                min_value = data[single_attribute].min()
                split_number = (max_value + min_value) / 2
                count_of_attribute_values = [0] * 2
                count_of_attribute_values[0] = len(data[data[single_attribute] <= split_number])
                count_of_attribute_values[1] = len(data[data[single_attribute] > split_number])

                entropy_ofattribute = 0.0

                entropy_ofvalue = self.entropy_Value_info(data[data[single_attribute] <= split_number])
                entropy_ofattribute += (count_of_attribute_values[0] * entropy_ofvalue)

                entropy_ofvalue = self.entropy_Value_info(data[data[single_attribute] > split_number])
                entropy_ofattribute += (count_of_attribute_values[0] * entropy_ofvalue)

                entropy_ofattribute = entropy_ofattribute / (sum(count_of_attribute_values))

                information_gain = entropy_set - entropy_ofattribute

                if information_gain > max_informationgain:
                    max_informationgain = information_gain
                    attributeon_split = single_attribute

                # unique_values = []
                # count_of_attribute_values = {}
                #
                # for value in data[single_attribute]:
                #     if value not in count_of_attribute_values:
                #         unique_values.append(value)
                #         count_of_attribute_values[value] = 1
                #     else:
                #         count_of_attribute_values[value] += 1
                # count_of_attribute_values = collections.Counter(data[single_attribute])
                # unique_values = list(count_of_attribute_values.keys())

        return max_informationgain, attributeon_split


    def gini(self, data, attribute_lists_left, entropy_set):
        min_gini = float('inf')
        # min_gini = entropy_set
        attributeon_split = ' '
        # selecting root_n number of attributes from entire attribute list
        n = len(attribute_lists_left)
        root_n = int(math.sqrt(n))
        root_n_attribute_list = random.sample(attribute_lists_left, root_n)
        for single_attribute in root_n_attribute_list:
            unique_values = []
            count_of_attribute_values = {}

            for value in data[single_attribute]:
                if value not in count_of_attribute_values:
                    unique_values.append(value)
                    count_of_attribute_values[value] = 1
                else:
                    count_of_attribute_values[value] += 1
            # count_of_attribute_values = collections.Counter(data[single_attribute])
            # unique_values = list(count_of_attribute_values.keys())

            entropy_ofattribute = 0.0
            for i in range(len(unique_values)):
                entropy_ofvalue = self.entropy_Value_gini(data[data[single_attribute] == unique_values[i]])
                entropy_ofattribute += (count_of_attribute_values[unique_values[i]] * entropy_ofvalue)

            entropy_ofattribute = entropy_ofattribute / (sum(count_of_attribute_values.values()))

            gini = entropy_ofattribute

            if gini < min_gini:
                min_gini = gini
                attributeon_split = single_attribute

        return min_gini, attributeon_split


    def find_Split(self, data, attributelist, flag):
        if flag == 1:
            parent_entropy = self.entropy_Value_info(data)
            # print("parent entropy", parent_entropy)
            max_infogain, split_attribute = self.Information_gain(data, attributelist, parent_entropy)

            if max_infogain == 0:
                return attributelist[0]
            # print("max infogain",max_infogain)
            return split_attribute

        if flag == 0:
            parent_entropy = self.entropy_Value_gini(data)
            # print("parent entropy", parent_entropy)
            min_gini, split_attribute = self.gini(data, attributelist, parent_entropy)

            if min_gini == 0:
                return attributelist[0]
            # print("min gini", min_gini)
            return split_attribute


    def predict_Class(self, currentNode, datapoint):
        if currentNode.leaf:
            return currentNode.final_value
        thisattribute = currentNode.attributename
        return self.predict_Class(currentNode.children[datapoint[thisattribute]], datapoint)

    def find_Accuracy(self, X, Y):
        pred = []
        # print(len(X),len(Y))
        for i in range(len(X)):
            pred.append(self.predict_Class(self.root, X.iloc[i]))
            # print(pred[i])
        return pred

    def flush(self):
        self.root = None
        self.attribute_value_list = dict()



def K_fold_datasets(dataset, k):
    class_frame = dataset.groupby('target')
    label_folds = {}

    for label, rows in class_frame:
        # Divide rows into k folds
        folds = np.array_split(rows, k)

        # Add folds to dictionary
        label_folds[label] = folds

    # Combine folds across labels
    all_folds = []
    for i in range(k):
        fold = []
        for label in label_folds:
            fold_data = label_folds[label][i]
            if isinstance(fold_data, str):
                fold_data = pd.DataFrame([fold_data])
            fold.append(fold_data)
        all_folds.append(pd.concat(fold))

    return all_folds


data = pd.read_csv("hw3_wine.csv", delimiter='\t')
data = pd.concat([data.iloc[:, 1:], data.iloc[:, 0]], axis=1)
data = data.rename(columns={'# class': 'target'})
data_wo_target = data.drop(labels='target',axis=1,inplace=False)
k = 10
num_trees = [1, 5, 10, 20, 30, 40, 50]
full_folds = K_fold_datasets(data, k)

forest_testaccuracy_plot, precision_plot, recall_plot, f1_plot = [], [], [], []

for tree in num_trees:

    forest_testaccuracy_folds, precision_folds, recall_folds, f1_folds = [], [], [], []

    for i in range(k):
        current_test_fold = full_folds[i]
        current_train_folds = pd.DataFrame()
        for j in range(k):
            if j != i:
                current_train_folds = pd.concat([current_train_folds, full_folds[j]])

        forest = RandomForest(tree)
        forest_accuracy, forest_precision, forest_recall, forest_f1 = forest.fit(current_train_folds, current_test_fold)

        # metrics for one forest for each fold
        forest_testaccuracy_folds.append(forest_accuracy)
        precision_folds.append(forest_precision)
        recall_folds.append(forest_recall)
        f1_folds.append(forest_f1)

    # mean for each no of trees for one entire k = 10 folds
    mean_accuracy = np.mean(forest_testaccuracy_folds)
    mean_precision = np.mean(precision_folds)
    mean_recall = np.mean(recall_folds)
    mean_f1 = np.mean(f1_folds)

    forest_testaccuracy_plot.append(mean_accuracy)
    precision_plot.append(mean_precision)
    recall_plot.append(mean_recall)
    f1_plot.append(mean_f1)


plt.plot(num_trees, forest_testaccuracy_plot)
plt.title('Accuracy vs. Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.show()

plt.plot(num_trees, precision_plot)
plt.title('Precision vs. Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Precision')
plt.show()

plt.plot(num_trees, recall_plot)
plt.title('Recall vs. Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Recall')
plt.show()

plt.plot(num_trees, f1_plot)
plt.title('F1 vs. Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('F1')
plt.show()








