# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:37:46 2019

@author: spong
"""
import sklearn.datasets as ds
import random
import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class node:
    
    def __init__(self):
        self.weights = []
        self.threshold = 0
        self.activation = 0
        self.error_rate = 0
    
    def define_weights(self, data):
        for item in data:
            self.weights.append(random.uniform(-1,1))
        self.weights.append(random.uniform(-1,1))
        #last weight will always be bias weight
        
    def fire_or_nahh(self, data):
        i = 0
        total = 0
        if(self.weights == []):
            self.define_weights(data)
        
        for item in data:
            total = total + (item * self.weights[i])
            i = i + 1
        total = total + ((-1) * self.weights[i]) #bias weight
        
        self.activation = 1 / (1 + math.exp((-1) * total))
            
        return self.activation
# =============================================================================
#         if (total > self.threshold):
#             return True
#         else:
#             return False
#         
# =============================================================================

###############################################################################
class node_layer:
    
    def __init__(self, number, output):
        self.output = output
        self.input = False
        self.nodes = []
        for num in range(number):
            self.nodes.append(node())
        
    def run_layer(self, data):
        returnData = []
        
        for cNode in self.nodes:
            returnData.append(cNode.fire_or_nahh(data))
        return returnData
##############################################################################
class neural_net:
    
    def __init__(self):
        self.layers = []
        
    def add_layer(self, nodes, output = False):
        if self.layers == []:
            self.layers.append(node_layer(nodes, output))
            self.layers[0].input = True
        else:
            self.layers.append(node_layer(nodes, output))
        
        
    def predict(self, data):
        predictions = []
        for item in data:
            output = item
            for layer in self.layers:
                output = layer.run_layer(output)
            
            predictions.append(output)
        return predictions
    
    def train(self, data_train, data_targets, learn_rate = 0.1):
        data_length = len(data_targets)
        accuracy = 0
        last_loop_acc = 0
        no_change_count = 0
        iterations = 0
        loop = True
        graph_x = []
        graph_y = []
            
        while (loop == True):
        #for counter in range(iterations):
            i = 0
            for item in data_train:
                test = []
                test.append(item)# breaks one line of data set, treats as full set
                target = self.target_convert(data_targets[i])
                prediction = self.predict(test)
                prediction = self.convert_prediction(prediction)# sets activation values
                j = 0
                if (prediction != data_targets[i]):
                    for layer in reversed(self.layers):# find error rates
                        
                        if layer.output == True:# output and hidden layers require different error rate calulations
                            for node in layer.nodes:
                                node.error_rate = node.activation * (1 - node.activation) * (node.activation - target[j])
                                
                        else:
                            k = 0
                            for node in layer.nodes:
                                error_sum = 0
                                for pre_node in self.layers[j].nodes:# look at next layer for certian values
                                    error_sum = error_sum + (pre_node.weights[k] * pre_node.error_rate)
                                node.error_rate = node.activation * (1 - node.activation) * error_sum
                                k = k + 1
                        j = j + 1 
                    j = -1 # used for accessing previous layer info
                    for layer in self.layers:# update weights
                        
                        for node in layer.nodes:
                            k = 0 # keeps track of which node we are looking at
                            if layer.input == True:# input layer uses input data to calulate weight changes
                                for weight in node.weights:
                                    if (k + 1) != len(node.weights):
                                        node.weights[k] = node.weights[k] - (learn_rate * node.error_rate * test[0][k])
                                    else:
                                        node.weights[k] = node.weights[k] - (learn_rate * node.error_rate * -1)
                                    k = k + 1
                            else:
                                for weight in node.weights:
                                    if (k + 1) != len(node.weights):
                                        node.weights[k] = node.weights[k] - (learn_rate * node.error_rate * self.layers[j].nodes[k].activation)
                                    else:
                                        node.weights[k] = node.weights[k] - (learn_rate * node.error_rate * -1)
                                    k = k + 1
                        j = j + 1
                i = i + 1    
                
                
####################### Validate Accuraccy             
            prediction = self.predict(data_train)
            prediction = self.convert_prediction(prediction)
         
            counter = 0
            total_right = 0
            accuracy = 0
            
            for value in data_targets:
                if value == prediction[counter]:
                    total_right = total_right + 1
                counter = counter + 1
                
            accuracy = (total_right * 100)/data_length
            if (accuracy > 80):
                loop == False
            if(accuracy == last_loop_acc):
                no_change_count = no_change_count + 1
                if (no_change_count > 10000):
                    loop = False
            else:
                no_change_count = 0
            last_loop_acc = accuracy
            if (iterations % 250 == 0):
                graph_x.append(iterations)
                graph_y.append(accuracy)  
                print(accuracy, iterations, no_change_count)
            iterations =  iterations + 1
            
            
        print("Total iterations:", iterations, "Final accuracy: ", accuracy, "%")
        plt.plot(graph_x,graph_y)
        plt.xlabel = "Iterations"
        plt.ylabel = "Accuracy"
        plt.show()
            
    def target_convert(self, target):
        if target == 0:
            return [1,0,0]
        if target == 1:
            return [0,1,0]
        if target == 2:
            return [0,0,1]
        else:
            return [0,0,0]
        
    def convert_prediction(self, data):
        convert = []
        for item in data:
            if item[0] > item[1] and item [0] > item[2]:
                convert.append(0)
            if item[1] > item[0] and item [1] > item[2]:
                convert.append(1)
            if item[2] > item[1] and item [2] > item[0]:
                convert.append(2)
        return convert
        
###############################################################################

iris = ds.load_iris()

learner = neural_net()

learner.add_layer(4)
learner.add_layer(3, True)

train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target, train_size=0.8, test_size=0.2, random_state=321)

learner.train(train_data, train_target)

prediction = learner.predict(test_data)
prediction = learner.convert_prediction(prediction)


print(prediction)
print(test_target)