import collections
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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
    def __init__(self, data):
        self.root = None
        self.attribute_value_list = dict()

    def getvalues_forattributes(self,data, attribute_list):
        for attribute in attribute_list:
            values = list(set(data[attribute]))
            self.attribute_value_list[attribute] = values
        return

    def Construct_tree(self,data, attribute_list):
        node = Node()
        if (len(data['target'].unique()) <= 1):  # dataset specific
            node.setasLeaf(data.iloc[0]['target'])
            return node
        if (len(attribute_list) == 0):
            node.setasLeaf(data.target.mode()[0])
            return node

        bestsplit_Attribute = self.find_Split(data, attribute_list)
        # print("Best attribute found ",bestsplit_Attribute)
        node.setasDecisionNode(bestsplit_Attribute)
        attribute_list.remove(bestsplit_Attribute)

        for val in self.attribute_value_list[bestsplit_Attribute]:
            split_data = data[data[bestsplit_Attribute] == val]
            if split_data.size == 0:
                node.setasLeaf(data.target.mode()[0])
                return node
            else:
                childTree = self.Construct_tree(split_data, attribute_list)
                node.children.append(childTree)

        return node

    def entropy_Value(self, data_rows):
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


    def Information_gain(self, data, attribute_lists_left, entropy_set):
        max_informationgain = 0
        attributeon_split = ' '
        for single_attribute in attribute_lists_left:
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
                entropy_ofvalue = self.entropy_Value(data[data[single_attribute] == unique_values[i]])
                entropy_ofattribute += (count_of_attribute_values[unique_values[i]] * entropy_ofvalue)

            entropy_ofattribute = entropy_ofattribute / (sum(count_of_attribute_values.values()))

            information_gain = entropy_set - entropy_ofattribute

            if information_gain > max_informationgain:
                max_informationgain = information_gain
                attributeon_split = single_attribute

        return max_informationgain, attributeon_split

    def find_Split(self, data, attributelist):
        parent_entropy = self.entropy_Value(data)
        max_infogain, split_attribute = self.Information_gain(data, attributelist, parent_entropy)

        if max_infogain == 0:
            return attributelist[0]

        return split_attribute

    def predict_Class(self, currentNode, datapoint):
        if currentNode.leaf:
            return currentNode.final_value
        thisattribute = currentNode.attributename
        return self.predict_Class(currentNode.children[datapoint[thisattribute]], datapoint)

    def find_Accuracy(self, X, Y):
        pred = np.empty(len(X), dtype=bool)
        # print(len(X),len(Y))
        for i in range(len(X)):
            pred[i] = (self.predict_Class(self.root, X.iloc[i]) == Y.iloc[i])
        return np.count_nonzero(pred) / np.size(pred)

    def flush(self):
        self.root = None
        self.attribute_value_list = dict()


data = pd.read_csv('house_votes_84.csv')
data_wo_target = data.drop(labels='target',axis=1,inplace=False)
trainaccuracy=[]
testaccuracy=[]

for index in range(100):
    tree = DecisionTree(data)
    attribute_list = list(data.iloc[:, :-1].columns)
    # maintaining an attribute list with all attribute names - here getting columns excluding last column -
    # dataset specific
    tree.getvalues_forattributes(data, attribute_list)

    # splitting the data into X and y - y is given last column - dataset specific
    X_train, X_test, y_train, y_test = train_test_split(data_wo_target,data.iloc[:, -1],test_size=0.20,shuffle=True)
    joined_data = X_train.join(y_train)
    tree.root = tree.Construct_tree(joined_data, attribute_list)
    # print(tree.root.attributename)
    # print('root')
    train_accuracy = (tree.find_Accuracy(X_train, y_train)*100)
    test_accuracy = (tree.find_Accuracy(X_test, y_test)*100)

    trainaccuracy.append(train_accuracy)
    testaccuracy.append(test_accuracy)

    tree.flush()

print("Mean of training accuracy",np.mean(trainaccuracy))
print("STD of training accuracy",np.std(trainaccuracy))
print("Mean of testing accuracy",np.mean(testaccuracy))
print("STD of testing accuracy",np.std(testaccuracy))

plt.hist(trainaccuracy,bins=5)
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title("Training Accuracy")
plt.show()
plt.hist(testaccuracy,bins=5)
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title("Testing Accuracy")
plt.show()








